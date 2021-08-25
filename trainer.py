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
from   torch.utils.data import Dataset
from   torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data.sampler import SubsetRandomSampler

from dataloaders import UnSupKittiDataset
from losses import Losses
from utils.transforms import UnNormalize
from evaluate import compute_errors
from geometry.pose_geometry import inverse_warp, disp_to_depth




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
        self.log_freq            = config['action']['log_freq']
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
            self.save_chkpnt()
        else:
            self.load_chkpnt()

        # init train transforms
        # TODO: Add color jitters and research
        # normalisation transforms
        self.unnormalize  =  UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        transform = [
            transforms.ToTensor(),
            transforms.Resize((384, 1280)), # packnet standard
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        
        # init dataset
        self.dataset = UnSupKittiDataset(config, transforms=transform)
        self.warp_sample = self.create_warp_sample()

        # create a dataset splits 
        random_seed      = config['action']['random_seed']
        validation_split = config['action']['split'][1]

        self.train_loader, self.validation_loader = self.create_loaders(random_seed, validation_split)

        # Start a new run, tracking hyperparameters in config
        if self.MLOps:
            wandb.init(project="unsup-depth-estimation", config=config)

            columns=["id", "image", "gt", "depth_pred", "pose", "pose_pred"]
            self.test_table = wandb.Table(columns=columns)
            self.row_id     = 0

            wandb.watch(self.pose_model,  log_freq=self.log_freq)

    def create_warp_sample(self):

        item = self.dataset.__getitem__(0)

        # load repeatable data
        tgt        = item['tgt'].unsqueeze(0)
        ref_imgs   = [img.unsqueeze(0) for img in item['ref_imgs']]
        intrinsics = torch.from_numpy(item['intrinsics'])

        sample = {'tgt': tgt,
                'ref_imgs': ref_imgs,
                'intrinsics': intrinsics}
        return sample

    def log_predictions(self, samples, outputs):

        # get samples
        image      = np.transpose(self.unnormalize(samples['tgt'][0].squeeze()).cpu().detach().numpy(), (1, 2, 0))
        gt         = samples['groundtruth'][0].squeeze().cpu().detach().numpy()
        pose_oxts  = samples['oxts'][1][0].squeeze().cpu().detach().numpy() # first ref image
        pose_pred  = outputs[1][0][0].squeeze().cpu().detach().numpy()      # first ref image
        depth_pred = disp_to_depth(outputs[0][0][0].squeeze().cpu().detach().numpy())

        self.test_table.add_data(self.row_id, wandb.Image(image), wandb.Image(gt), wandb.Image(depth_pred), str(pose_oxts), str(pose_pred))
        self.row_id += 1

    def create_loaders(self, random_seed, valid_split_ratio):
        dataset_size = len(self.dataset)
        indices      = list(range(dataset_size))
        split    = int(np.floor(valid_split_ratio * dataset_size))

        if self.shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices  = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader      = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                                           sampler=train_sampler, num_workers=16)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                            sampler=valid_sampler, num_workers=8)
        return train_loader, validation_loader
            
    def save_chkpnt(self):
        print("Saving checkpoint..")

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

    def set_train(self):
        self.depth_model.train()
        self.pose_model.train()

    def set_eval(self):
        self.depth_model.eval()
        self.pose_model.eval()

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
            acc = compute_errors(gt, outputs[0])
            
            if self.MLOps:
                wandb.log(acc, step=self.epoch)
            

            # compare againt checkpoint
            # if abs_rel > self.valid_acc:
            #     self.valid_acc = abs_rel
    
    @torch.no_grad()
    def log_warps(self, indx):
        self.set_eval()

        # pass through model
        outputs = self.process_batch(self.warp_sample, warp_test=True)
        depth   = disp_to_depth(outputs[0][0].squeeze().unsqueeze(0))

        poses   = outputs[1]
        poses   = poses[:, 0, :]

        ref_imgs   = self.warp_sample['ref_imgs'][0].to(self.device)
        intrinsics = self.warp_sample['intrinsics'].to(self.device)

        # create warp
        projected_img = inverse_warp(ref_imgs, depth, poses, intrinsics, warp_test=True)
        projected_img = np.transpose((projected_img.squeeze()).cpu().detach().numpy(), (1, 2, 0))
        projected_img = 0.5 + (projected_img * 0.5) # remove normalization

        file_name = './images/warping/' + str(indx) + '.png'
        plt.imsave(file_name, projected_img)

        self.set_train()

    def run_epoch(self):

        # process batch
        for batch_indx, samples in tqdm(enumerate(self.train_loader), unit='images',
                                        total=len(self.train_loader), desc=f"Epoch {self.epoch} BATCH"):

            self.model_optimizer.zero_grad()
            
            outputs, self.loss = self.process_batch(samples)
            #sum(self.loss).backward()
            self.model_optimizer.step() 

            if self.MLOps:
                wandb.log({"loss":sum(self.loss), "mul_app_loss": self.loss[0], \
                        "smoothness_loss":self.loss[1]})

                # if self.epoch < 1:
                    # self.log_warps(batch_indx)

                
                if (batch_indx + 1) % self.log_freq == 0:
                    self.log_predictions(samples, outputs)
            
            break
                    
            

        self.model_lr_scheduler.step()

        # validate after each epoch
        # self.validate()

        # save checkpoint
        self.save_chkpnt()

    def process_batch(self, samples, warp_test=False):
        tgt        = samples['tgt'].to(self.device) # T(B, 3, H, W)
        ref_imgs   = [img.to(self.device) for img in samples['ref_imgs']] # [T(B, 3, H, W), T(B, 3, H, W)]
        intrinsics = samples['intrinsics'].to(self.device)

        disp = self.depth_model(tgt) # [T(B, 1, H, W), T(B, 1, H_re, W_re), ....rescaled)
        poses = self.pose_model(tgt, ref_imgs) # T(B, 2, 6)

        if warp_test:
            return [disp, poses]
        else:
            # forward + backward 
            loss = self.criterion.forward(tgt, ref_imgs, disp, poses, intrinsics)
            return [disp, poses], loss
