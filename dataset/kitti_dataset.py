import os
import glob
import torch
import yaml
from itertools import islice
from collections import deque

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

from transform import Transform
from calibration import Calibration

def sliding_window(iterable, size):
    '''
        returns a generator object 
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield list(window)
        window.append(item)
    if window:  
        yield list(window)

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

        self.kitti_filepath  = config['datasets']['path']
        self.img_width       = config['datasets']['augmentation']['image_width']
        self.img_height      = config['datasets']['augmentation']['image_height']
        self.seq_len         = config['datasets']['sequence_length']
        self.train           = config['train']
        
        self.transforms = transforms
        self.samples = self._init_samples()

    def get_img_dirs(self, path):
        drive_dates = glob.glob(path + '*')
        img_dirs    = []

        for date in drive_dates:
            drives = glob.glob(date + '/*_sync')
            for drive in drives:
                images = glob.glob(drive + '/image_02/data/*.png')
                img_dirs.extend(images)

        return sorted(img_dirs)
    
    def load_img(self, path):
        img = Image.open(path)
        img = self.transforms(img)
        return img.astype(np.float32)

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

        img_dirs   = self.get_img_dirs(self.kitti_filepath)
        mid        = self.seq_len//2

        sample   = {}
        samples  = []
        ref_imgs = []
        for window in sliding_window(img_dirs, self.seq_len):
            tgt_dir      = window.pop(mid)
            ref_img_dirs = window

            sample['tgt']     = tgt_dir
            sample['ref_imgs'] = ref_img_dirs

            calib_dir = tgt_dir[:23]
            calib     = Calibration(calib_dir)
            sample['intrinsics'] = calib.P
            sample['extrinsics'] = calib.Tx

            samples.append(sample)

        return samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # prep target
        sample['tgt'] = self.load_img(sample['tgt'])

        # prep sources
        imgs = []
        for img in sample['ref_imgs']:
            imgs.append(self.load_img(img))
        sample['ref_imgs'] = imgs

        return sample


with open('../configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

transforms = transforms.ToTensor()


kitti = UnSupKittiDataset(config, transforms)
print(kitti.__len__())






