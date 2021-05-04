import os
import glob
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils

from transform import Transform
from calibration import Calibration

class UnSupKittiDataset(Dataset, Transform):

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

        A sample is of the form:
        sample = {'target'    : tgt
                  'refrences' : refs
                  'intrinsics': T}
    '''

    def __init__(self, config):

        kitti_filepath = config['datasets']['path']
        img_width      = config['datasets']['augmentation']['image_width']
        img_height     = config['datasets']['augmentation']['image_height']

        super(UnSupKittiDataset, self).__init__(kitti_filepath, img_width, img_height)

        self.samples = self._init_samples()
        pass

    def _init_samples(self):
        path       = self.kitti_filepath
        drive_dirs = glob.glob(path + "*_sync")
        img_dirs   = []

        for _dir in drive_dirs:
            img  = glob.glob(_dir + "/image_02/data/*png")
            img_dirs.extend(img)
        
        img_dirs = sorted(img_dirs)
        print(img_dirs[:20])
        return None
        
    
    def __len__(self):
        return len(self.samples)

    # def __getitem__(self, idx):



with open('../configs/basic_config.yaml') as file:
    config = yaml.full_load(file)

kitti = UnSupKittiDataset(config)







