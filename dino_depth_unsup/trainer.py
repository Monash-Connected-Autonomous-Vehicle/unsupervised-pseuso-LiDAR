import warnings
from torchvision.transforms.transforms import ToPILImage
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
from tqdm import tqdm


import torch
from   torch.utils.data import Dataset, Sampler
from   torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data.sampler import SubsetRandomSampler

from dataloaders import dino_dataset
from losses import Losses
from utils.transforms import UnNormalize
from geometry.pose_geometry import *


class SequentialIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Trainer:
    def __init__(self, config):

        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_path = './pretrained/'+ config['model']['name'] +'.pth'

        # init training and optimizer variables
        self.batch_size          = config['action']['batch_size']
        self.learning_rate       = config['action']['optimizer']['depth']['lr']
        self.scheduler_step_size = config['action']['scheduler']['step_size']
        self.gamma               = config['action']['scheduler']['gamma']
        self.shuffle_dataset     = config['datasets']['augmentation']['shuffle']
        self.mode                = config['action']['mode']
        self.MLOps               = config['action']['MLOps']
        self.train_from_scratch  = config['action']['from_scratch']
        self.num_epochs          = config['action']['num_epochs']
        self.num_workers         = config['action']['num_workers']
        self.log_freq            = config['action']['log_freq']
        self.epoch      = 0
        self.step       = 0 

        # init models based on config
        self.depth_model = self.load_from_config(config, model_type='depth')

        if self.depth_model == None: # or self.pose_model == None:
            assert("Config file format is incorrect: Take a look at example \
                    config files")

        # init model, weigths, and params
        # TODO: Check if this works
        parameters_train  = list(self.depth_model.parameters())

        # init optimiser and LR shcheduler
        self.model_optimizer = optim.Adam(parameters_train, self.learning_rate)

        # init losses and acc.
        self.criterion = Losses()
        self.loss      = None
        self.valid_acc = 0
        
        # load checkpoint
        if self.train_from_scratch:
            self.save_chkpnt()
        else:
            self.load_chkpnt()

        # init train transforms
        # TODO: Add color jitters and research
        # normalisation transforms
        self.unnormalize  =  UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        
        # init dataset
        self.dataset = dino_dataset(transforms=transform)

        # create a dataset splits 
        random_seed      = config['action']['random_seed']
        validation_split = config['action']['split'][1]

        self.train_loader = self.create_loaders(random_seed, validation_split)

        # sample to test warp
        self.warp_sample = self.create_warp_sample()

        # Start a new run, tracking hyperparameters in config
        if self.MLOps:

            # weights and biases init
            wandb.init(project="unsup-depth-estimation", config=config)

            columns=["id", "image", "gt", "depth_pred"]
            self.test_table = wandb.Table(columns=columns)
            self.row_id     = 0

            wandb.watch(self.depth_model,  log_freq=self.log_freq)

    def save_chkpnt(self):
        print("Saving checkpoint..")

        self.checkpoint = { 'epoch': self.epoch, 
                            'dpth_mdl_state_dict': self.depth_model.state_dict(),
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
        self.model_optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
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
        if self.train_from_scratch:
            model.init_weights()
        return model.to(self.device)
    
    def create_loaders(self, random_seed=None, valid_split_ratio=None):
        train_loader      = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return train_loader

    def set_train(self):
        self.depth_model.train()
        self.pose_model.train()

    def set_eval(self):
        self.depth_model.eval()
        self.pose_model.eval()

    def create_warp_sample(self):
        return next(iter(self.train_loader))

    def log_depth_predictions(self, samples, outputs):

        # get samples
        image      = np.transpose(self.unnormalize(samples['tgt'][0].squeeze()).cpu().detach().numpy(), (1, 2, 0))
        gt         = samples['groundtruth'][0].squeeze().cpu().detach().numpy()
        depth_pred = disp_to_depth(outputs[0][0][0].squeeze().cpu().detach().numpy())

        self.test_table.add_data(self.row_id, wandb.Image(image), wandb.Image(gt), wandb.Image(depth_pred))
        self.row_id += 1

    @torch.no_grad()
    def log_warps(self, indx):
        self.set_eval()

        # pass through model
        outputs = self.process_batch(self.warp_sample, warp_test=True, semi_sup_pose=True)
        depth   = disp_to_depth(outputs[0][0])

        poses   = outputs[1]
        poses   = poses[:, 0, :]

        ref_imgs   = [ref_img.to(self.device) for ref_img in self.warp_sample['ref_imgs']]
        intrinsics = self.warp_sample['intrinsics'].to(self.device)

        # create warp
        projected_img = inverse_warp(ref_imgs[0], depth, poses, intrinsics)[1]
        projected_img = np.transpose((projected_img.squeeze()).cpu().detach().numpy(), (1, 2, 0))
        projected_img = 0.5 + (projected_img * 0.5) # remove normalization

        d = depth[0][0].cpu().detach().numpy()

        warp_file_name = './images/warping/' + str(indx) + '.png'
        depth_name = './images/depth/' + str(indx) + '.png'
        plt.imsave(warp_file_name, projected_img)
        plt.imsave(depth_name, d)

        self.set_train()

    def train(self):

        self.set_train()
        self.start_time = time.time()

        # run epoch
        for self.epoch in range(self.num_epochs):
            self.run_epoch()
            break
        
        if self.MLOps:
            # log predictions table to wandb
            wandb.log({"test_predictions" : self.test_table})

    def run_epoch(self):

        # process batch
        for batch_indx, samples in tqdm(enumerate(self.train_loader), unit='images',
                                        total=len(self.train_loader), desc=f"Epoch {self.epoch} BATCH"):

            self.model_optimizer.zero_grad()
            
            outputs, self.loss = self.process_batch(samples, semi_sup_pose=False)
            sum(self.loss).backward()
            self.model_optimizer.step() 

            if self.epoch < 1 and (batch_indx + 1) < 10000:
              self.log_warps(batch_indx)

            break

            if self.MLOps:

                wandb.log({"loss":sum(self.loss), "mul_app_loss": self.loss[0], \
                        "smoothness_loss":self.loss[1]})
                
                if (batch_indx + 1) % self.log_freq == 0:
                    self.log_depth_predictions(samples, outputs)
                    # self.log_warps(batch_indx)
            

        # self.model_lr_scheduler.step()

        # validate after each epoch
        # self.validate()

        # save checkpoint
        self.save_chkpnt()

    def process_batch(self, samples, warp_test=False, semi_sup_pose=False):
        tgt        = samples['tgt'].to(self.device) # T(B, 3, H, W)
        ref_imgs   = samples['ref_imgs'].to(self.device)

        disp = self.depth_model(tgt) # [T(B, 1, H, W), T(B, 1, H_re, W_re), ....rescaled)

        if semi_sup_pose:
            poses = samples["oxts"].unsqueeze(1).to(self.device)

        if warp_test:
            return [disp, poses]
        else:
            # forward + backward 
            loss = self.criterion.forward(tgt, ref_imgs, disp, poses)
            return [disp, poses], loss