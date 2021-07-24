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

from utils.calibration import Calibration
from utils.oxts_parser import *

class KittiDataset(Dataset):
    def __init__(self, config, transforms=None):

        super(KittiDataset, self).__init__()
        self.split           = config['datasets']['split']
        self.kitti_filepath  = config['datasets']['path']
        self.img_width       = config['datasets']['augmentation']['image_width']
        self.img_height      = config['datasets']['augmentation']['image_height']
        self.seq_len         = config['datasets']['sequence_length']

        self.transforms = transforms
        self.samples = []

    def load_img(self, path):
        img = np.asarray(Image.open(path), dtype=np.float32) / 255.0

        if transforms:
            img = self.transforms(img)
        return img
    
    def load_depth_img(self, filename):

        if not filename:
            return None
        
        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        return depth
        
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
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        # only contains dir links to save cpu memory
        sample = self.samples[index]

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

        ret_sample['groundtruth'] = self.load_depth_img(sample['groundtruth'])

        return ret_sample

    def get_mul_items(self, indx_list):
        items = []
        for x in indx_list:
            items.append(self.__getitem__(x))
        return items


class UnSupKittiDataset(KittiDataset):
    '''
        Uses one of the split test files to 
        create a dataset.
    '''
    def __init__(self, *args, **kwargs):
        super(UnSupKittiDataset, self).__init__(*args, **kwargs)
        self._init_samples()

    def _init_samples(self):

        print('Initializing samples..')
        
        file   = open(self.split, 'r')
        lines  = [line.strip('\n') for line in file]
        
        sample = {}
        for line in lines:
            sample_dirs   = line.split(' ')

            sample['tgt']      = sample_dirs[0]
            sample['ref_imgs'] = sample_dirs[1:3]

            calib_dir = sample_dirs[0][:20] 
            calib     = Calibration(calib_dir)
            sample['intrinsics'] = calib.P
            sample['extrinsics'] = calib.Tx

            oxts_lst = []
            for i in range(3):
                oxts_dir = sample_dirs[i]

                img_indx = oxts_dir[-14:-4]
                oxts_dir = oxts_dir[0:46]
                oxts_dir = oxts_dir + '/oxts/data/' + img_indx + '.txt'

                oxts_lst.append(oxts_dir)
            
            sample['oxts'] = load_oxts_packets_and_poses(oxts_lst)

            sample['groundtruth'] = sample_dirs[3]
            
            self.samples.append(sample)


# TODO: use image transforms on velodyne points
# to create GT.
class UnSupFullKittiDataset(KittiDataset):
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

    def __init__(self, *args, **kwargs):
        super(UnSupFullKittiDataset, self).__init__(*args, **kwargs)
        self._init_samples()


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

        print('Initializing samples..')

        img_dirs, oxts_dirs   = self.get_dirs(self.kitti_filepath)
        mid      = self.seq_len//2
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

            sample['groundtruth'] = None

            self.samples.append(sample)
    