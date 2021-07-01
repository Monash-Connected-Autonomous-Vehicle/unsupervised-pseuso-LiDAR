import os
import glob
import torch
import yaml
from itertools import islice
from collections import deque

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt

from PIL import Image

from .calibration import Calibration
from .oxts_parser import *

class UnSupKittiDataset(Dataset):

    '''
        Assumming a base directory of 'KITTI/date/'
        the stack would be as follows:

        calib_cam_to_cam.txt
        calib_imu_to_velo.txt
        calib_velo_to_cam.txt
        drive_sync_1/image_02/data/sample0001.png
        drive_sync_1/image_02/data/sample0002.png
        ...
        drive_sync_2/image_02/data/sample0001.png
        drive_sync_2/image_02/data/sample0002.png
        ...
    '''

    def __init__(self, config, transforms=None):

        super(UnSupKittiDataset, self).__init__()
        self.count = 0
        self.kitti_filepath  = config['datasets']['path']
        self.img_width       = config['datasets']['augmentation']['image_width']
        self.img_height      = config['datasets']['augmentation']['image_height']
        self.seq_len         = config['datasets']['sequence_length']

        self.transforms = transforms
        self.samples = []

        self._init_samples()

    def sliding_window(self, iterable, size):
        '''
            returns a iterable generator object 
            that is a sliding windowed list of length 
            `size`.
        '''
        iterable = iter(iterable)
        window = deque(islice(iterable, size), maxlen=size)
        for item in iterable:
            yield list(window)
            window.append(item)
        if window:  
            yield list(window)

    def get_dirs(self, path):
        drive_dates = glob.glob(path + '*')
        img_dirs    = []
        oxts_dirs   = []

        for date in drive_dates:
            drives = glob.glob(date + '/*_sync')
            for drive in drives:
                images = glob.glob(drive + '/image_02/data/*.png')
                oxts_pckts = glob.glob(drive + '/oxts/data/*.txt') 
                img_dirs.extend(images)
                oxts_dirs.extend(oxts_pckts)

        return sorted(img_dirs), sorted(oxts_dirs)
    
    def load_img(self, path):
        img = np.asarray(Image.open(path), dtype=np.float32) / 255.0

        if transforms:
            img = self.transforms(img)

        return img

    def _init_samples(self):
        '''
            A sample is of the form:
            sample = {
                'tgt'       : target image
                'ref_imgs'  : refrence images
                'intrinsics': intrinsic matrix
                'extrinsics': transformation matrix
                }
        '''
        
        img_dirs, oxts_dirs   = self.get_dirs(self.kitti_filepath)
        mid        = self.seq_len//2
        ref_imgs = []

        for window in self.sliding_window(img_dirs, self.seq_len):
            sample   = {} # must be defined for each new iteration
            
            tgt_dir      = window.pop(mid)
            ref_img_dirs = window

            
            sample['tgt']      = tgt_dir
            sample['ref_imgs'] = ref_img_dirs
            
            calib_dir = tgt_dir[:20] # if no data is being loaded, check the file stuct
            calib     = Calibration(calib_dir)
            sample['intrinsics'] = calib.P
            sample['extrinsics'] = calib.Tx

            oxts_lst = [oxts_dirs[img_dirs.index(tgt_dir)],  
                       oxts_dirs[img_dirs.index(ref_img_dirs[0])],
                       oxts_dirs[img_dirs.index(ref_img_dirs[1])]]
            sample['oxts'] = load_oxts_packets_and_poses(oxts_lst)

            self.samples.append(sample)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        # only contains dir links to save cpu memory
        sample = self.samples[idx]

        # init return sample
        # can't change sample as we load
        # image each epoch
        ret_sample = {}

        # prep target
        ret_sample['tgt'] = self.load_img(sample['tgt'])

        # prep sources
        imgs = []
        for img in sample['ref_imgs']:
            imgs.append(self.load_img(img))
        ret_sample['ref_imgs'] = imgs

        ret_sample['intrinsics'] = sample['intrinsics']
        ret_sample['extrinsics'] = sample['extrinsics']
        ret_sample['oxts']       = sample['oxts']

        return ret_sample

    def get_mul_items(self, indx_list):
        items = []
        for x in indx_list:
            items.append(self.__getitem__(x))
        return items
