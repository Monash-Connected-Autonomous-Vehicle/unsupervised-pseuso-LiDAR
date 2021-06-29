import yaml
import importlib
import sys
import os
import time
from   inspect import getmembers, isclass

import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np

from   PIL import Image

from utils.kitti_dataset import UnSupKittiDataset
from losses import Losses


class Trainer:
    def __init__(self, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init models based on config
        self.depth_model = self.load_from_config(config, model_type='depth')
        self.pose_model  = self.load_from_config(config, model_type='pose')
        
        if self.depth_model == None or self.pose_model == None:
            assert("Config file format is incorrect: Take a look at example \
                    config files")

        # init training and optimizer variables
        self.batch_size          = config['action']['batch_size']
        self.learning_rate       = config['action']['optimizer']['depth']['lr']
        self.scheduler_step_size = config['action']['scheduler']['step_size']
        self.gamma               = config['action']['scheduler']['gamma']
        self.shuffle_dataset     = config['datasets']['augmentation']['shuffle']
        self.mode                = config['action']['mode']
        self.parameters_train  = list(self.depth_model.parameters())
        self.parameters_train += list(self.pose_model.parameters())

        # init train transforms
        # TODO: Add composite transforms
        transform = transforms.ToTensor()
        
        # init dataset
        self.dataset = UnSupKittiDataset(config, transforms=transform)

        # create a dataset splits (70, 15, 15) -> (train, val, test)
        random_seed      = config['action']['random_seed']
        validation_split = config['action']['split'][1]
        test_split       = config['action']['split'][2]
        dataset_size = len(self.dataset)
        indices      = list(range(dataset_size))
        split    = int(np.floor(validation_split * dataset_size))

        if self.shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices  = indices[split:], indices[:split]
        train_indices, test_indices = train_indices[split:], train_indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler  = SubsetRandomSampler(test_indices)

        self.train_loader     = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=train_sampler, num_workers=0)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                            sampler=valid_sampler)
        self.test_loader       = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=test_sampler)

        # init optimiser and LR shcheduler
        # TODO: if not config >> from scratch, load optimiser for continual training
        self.model_optimizer = optim.Adam(self.parameters_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step_size, self.gamma)

        # init losses
        self.criterion = Losses()
    
    def load_from_config(self, config, model_type='depth'):
        '''
            Returns the models from a config file

            # TODO:if checkpoint exists, load weights
            # else init weights (xavier: uniformly distributed)
        '''
        module = importlib.import_module('models.'+ model_type + '.' + \
                                        config['model'][model_type]['file'])

        model_name =  config['model'][model_type]['name']
        model = None
        for name, obj in getmembers(module, isclass):
            if name == model_name:
                model = obj
        
        # init model and weigths
        model = model()
        model.init_weights()
        return model

    def set_train(self):
        print('Training...')
        self.depth_model.train()
        self.pose_model.train()

    def set_eval(self):
        print('Evaluating...')
        self.depth_model.eval()
        self.pose_model.eval()

    def train(self):
        self.num_epochs = config['action']['num_epochs']
        self.epoch      = 0
        self.step       = 0 
        self.start_time = time.time()

        # run epoch
        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            break
            # save model

    def run_epoch(self):

        self.set_train()

        # process batch
        for batch_indx, samples in enumerate(self.train_loader):

            self.model_optimizer.zero_grad()

            outputs, loss = self.process_batch(samples)
            loss.backward()
            self.model_optimizer.step()
            break
        
        self.model_lr_scheduler.step()

        # validate after each epoch?

    def process_batch(self, samples):
        tgt        = samples['tgt'].to(self.device) # T(B, 3, H, W)
        ref_imgs   = [img.to(self.device) for img in samples['ref_imgs']] # [T(B, 3, H, W), T(B, 3, H, W)]
        intrinsics = samples['intrinsics'].to(self.device)
        extrinsics = samples['extrinsics'].to(self.device)

        disp = self.depth_model(tgt) # [T(B, 1, H, W), T(B, 1, H_re, W_re), ....rescaled)
        poses = self.pose_model(tgt, ref_imgs) # T(B, 2, 6)

        # forward + backward + optimize
        loss = self.criterion.multiview_appearence_matching(tgt, ref_imgs, disp, poses, intrinsics)

        return [disp, poses], loss


with open('configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

trainer = Trainer(config)
trainer.train()