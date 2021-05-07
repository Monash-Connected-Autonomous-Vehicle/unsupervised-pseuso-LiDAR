import yaml
import importlib
import sys
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

from dataset.kitti_dataset import UnSupKittiDataset

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

        # TODO: if checkpoint exists, load weights
        # else init weights (xavier: uniformly distributed)

        # init train transforms
        # TODO: Add composit transforms
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
                                           sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                            sampler=valid_sampler)
        self.test_loader       = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=test_sampler)

        # init optimiser and LR shcheduler
        self.model_optimizer = optim.Adam(self.parameters_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step_size, self.gamma)

        # init losses
    
    def load_from_config(self, config, model_type='depth'):
        '''
            Returns the models from a config file
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
        return model

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def train(self):
        # run epoch

        # save model
        pass

    def run_epoch(self):
        # process batch

        # validate after each epoch?
        pass

    def process_batch(self):
        pass

with open('configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

trainer = Trainer(config)