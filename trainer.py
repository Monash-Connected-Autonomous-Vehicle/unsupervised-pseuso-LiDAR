import yaml
import importlib
import sys
import os
import time
from   inspect import getmembers, isclass
import matplotlib.pyplot as plt
import numpy as np
from   PIL import Image
import wandb


import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data.sampler import SubsetRandomSampler

from dataloaders import UnSupKittiDataset
from losses import Losses
from evaluate import compute_errors


class Trainer:
    def __init__(self, config):

        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = './models/pretrained/'+ config['model']['name'] +'.pth'

        # init training and optimizer variables
        self.batch_size          = config['action']['batch_size']
        self.learning_rate       = config['action']['optimizer']['depth']['lr']
        self.scheduler_step_size = config['action']['scheduler']['step_size']
        self.gamma               = config['action']['scheduler']['gamma']
        self.shuffle_dataset     = config['datasets']['augmentation']['shuffle']
        self.mode                = config['action']['mode']
        self.train_from_scratch  = config['action']['from_scratch']
        self.num_epochs          = config['action']['num_epochs']
        self.epoch      = 0
        self.step       = 0 

        # init models based on config
        self.depth_model = self.load_from_config(config, model_type='depth')
        self.pose_model  = self.load_from_config(config, model_type='pose')
        
        if self.depth_model == None or self.pose_model == None:
            assert("Config file format is incorrect: Take a look at example \
                    config files")

        # init model, weigths, and params
        # TODO: Check if this works
        self.parameters_train  = list(self.depth_model.parameters())
        self.parameters_train += list(self.pose_model.parameters())

        # init optimiser and LR shcheduler
        self.model_optimizer = optim.Adam(self.parameters_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step_size, self.gamma)

        # init losses and acc.
        self.criterion = Losses()
        self.loss      = None
        self.valid_acc = 0
        # load checkpoint
        if self.train_from_scratch:
            print("Training from scratch..")
            self.save_chkpnt()
        else:
            self.load_chkpnt()

        # init train transforms
        # TODO: Add composite transforms
        transform = transforms.ToTensor()
        
        # init dataset
        self.dataset = UnSupKittiDataset(config, transforms=transform)

        # create a dataset splits (70, 15, 15) -> (train, val, test)
        random_seed      = config['action']['random_seed']
        validation_split = config['action']['split'][1]
        # test_split       = config['action']['split'][2]
        dataset_size = len(self.dataset)
        indices      = list(range(dataset_size))
        split    = int(np.floor(validation_split * dataset_size))

        if self.shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices  = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader      = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=train_sampler, num_workers=0)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                            sampler=valid_sampler)
        
        # Start a new run, tracking hyperparameters in config
        wandb.init(project="unsup-depth-estimation", config=config)
            
    def save_chkpnt(self):
        self.checkpoint = { 'epoch': self.epoch, 
                            'dpth_mdl_state_dict': self.depth_model.state_dict(),
                            'pose_mdl_state_dict': self.pose_model.state_dict(),
                            'optimizer_state_dict': self.model_optimizer.state_dict(),
                            'loss': self.loss,
                            'valid_acc': self.valid_acc
                        }

        # save file
        torch.save(self.checkpoint, self.save_path)
        
    def load_chkpnt(self):
        print("Loading Pretrained model..")

        # find checkpoint and load model
        self.checkpoint = torch.load(self.save_path)
        self.depth_model.load_state_dict(self.checkpoint['dpth_mdl_state_dict'])
        self.pose_model.load_state_dict(self.checkpoint['pose_mdl_state_dict'])
        self.model_optimizer .load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.epoch = self.checkpoint['epoch']
        self.valid_acc = self.checkpoint['valid_acc']

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

        model = model()
        if not self.train_from_scratch:
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

        self.set_train()
        self.start_time = time.time()

        # run epoch
        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            break
    
    @torch.no_grad()
    def validate(self):

        self.set_eval()

        # calculate evaluation accuracy
        # with torch.no_grad():
        for batch_indx, samples in enumerate(self.validation_loader):
            outputs, self.loss = self.process_batch(samples)
            
            # calculate the accuracy
            gt = samples['groundtruth']

            # compute_error
            silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = compute_errors(gt, outputs[0])
            
            # compare againt checkpoint
            if abs_rel > self.valid_acc:
                self.valid_acc = abs_rel
                
                # save checkpoint
                self.save_chkpnt()
                
    def run_epoch(self):

        # process batch
        for batch_indx, samples in enumerate(self.train_loader):

            self.model_optimizer.zero_grad()
            
            outputs, self.loss = self.process_batch(samples)
            sum(self.loss).backward()
            self.model_optimizer.step()   

            wandb.log({"loss":sum(self.loss), "mul_app_loss": self.loss[0], \
                    "smoothness_loss":self.loss[1]}, step=batch_indx)
    
        self.model_lr_scheduler.step()

        # validate after each epoch?
        # self.validate()
        # wandb.log({'acc': self.valid_acc}, step=self.epoch)

    def process_batch(self, samples):
        tgt        = samples['tgt'].to(self.device) # T(B, 3, H, W)
        ref_imgs   = [img.to(self.device) for img in samples['ref_imgs']] # [T(B, 3, H, W), T(B, 3, H, W)]
        intrinsics = samples['intrinsics'].to(self.device)
        extrinsics = samples['extrinsics'].to(self.device)

        disp = self.depth_model(tgt) # [T(B, 1, H, W), T(B, 1, H_re, W_re), ....rescaled)
        poses = self.pose_model(tgt, ref_imgs) # T(B, 2, 6)

        # forward + backward + optimize
        loss = self.criterion.forward(tgt, ref_imgs, disp, poses, intrinsics)
        
        return [disp, poses], loss


with open('configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

trainer = Trainer(config)
trainer.train()