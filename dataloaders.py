import os
import glob
import torch
import yaml
from copy import deepcopy
from itertools import islice
from collections import deque

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from PIL import Image

from geometry.calibration import Calibration
from geometry.oxts_parser import *
from geometry.pose_geometry import *

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

    def load_img(self, path, gt=False):
        img = np.asarray(Image.open(path), dtype=np.float32) / 255.0

        h = None
        w = None
        if not gt:
            h = img.shape[0]
            w = img.shape[1]

        for t in self.transforms[:-1]:
            img = t(img)

        # do not normalize gt depth
        if not gt:
            img = self.transforms[-1](img)

        return img.squeeze(), h, w
        
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
        ret_sample['tgt'], og_h, og_w = self.load_img(sample['tgt'])

        # prep sources
        imgs = []
        for img in sample['ref_imgs']:
            imgs.append(self.load_img(img)[0])
        ret_sample['ref_imgs'] = imgs

        # load and scale intrinsics based
        # on size of Resize
        intrinsics = sample['intrinsics']
        intrinsics[0] *= self.img_width  / og_w
        intrinsics[1] *= self.img_height / og_h
        ret_sample['intrinsics'] = torch.from_numpy(intrinsics) 
        
        # oxts packets to poses
        oxts   = load_oxts_packets_and_poses(sample['oxts'])

        # TODO: transform oxts pose from imu to cam coords
        oxts[1] = sample['velo_to_cam'] @ sample['imu_to_velo'] @ oxts[1]
        oxts[2] = sample['velo_to_cam'] @ sample['imu_to_velo'] @ oxts[2]

        # convert poses from mat to euler 
        poses  = [oxts[1], oxts[2]]
        angles = [mat2euler(pose[:3,:3]) for pose in poses]
        ts     = [pose[:3, 3] for pose in poses]

        ret_sample['oxts'] = [torch.from_numpy(np.concatenate((np.array([0, 0, 0]), t))) for ang, t in zip(angles, ts)]


        ret_sample['groundtruth'], _, _ = self.load_img(sample['groundtruth'], gt=True)

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

            calib_dir = sample_dirs[0][:20] # mac - 20 , beauty - 29
            calib     = Calibration(calib_dir)
            sample['intrinsics']  = calib.P[:, :3]
            sample['imu_to_velo'] = calib.T_imu_velo
            sample['velo_to_cam'] = calib.T_velo_cam

            oxts_lst = []
            for i in range(3):
                oxts_dir = sample_dirs[i]

                img_indx = oxts_dir[-14:-4]
                oxts_dir = oxts_dir[0:46] # mac - 46, beauty - 55
                oxts_dir = oxts_dir + '/oxts/data/' + img_indx + '.txt'

                oxts_lst.append(oxts_dir)

            sample['oxts'] = oxts_lst

            sample['groundtruth'] = sample_dirs[3]
            
            self.samples.append(deepcopy(sample))


# TODO: use image transforms on velodyne points
# to create GT.
class UnSupStackedDataset(KittiDataset):
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
        super(UnSupStackedDataset, self).__init__(*args, **kwargs)
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
            sample['intrinsics'] = calib.K

            oxts_lst = [oxts_dirs[img_dirs.index(tgt_dir)],  
                       oxts_dirs[img_dirs.index(ref_img_dirs[0])],
                       oxts_dirs[img_dirs.index(ref_img_dirs[1])]]
            sample['oxts'] = load_oxts_packets_and_poses(oxts_lst)

            sample['groundtruth'] = None

            self.samples.append(deepcopy(sample))
    
