import yaml
import importlib
import sys
from inspect import getmembers, isclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from dataset.kitti_dataset import UnSupKittiDataset

def str_to_class(str):
    print(sys.modules[__name__])
    # return getattr(sys.modules[__name__], str)

class Trainer:
    def __init__(self, config):
        # init models based on config
        self.depth_model = self.load_from_config(config, model_type='depth')
        self.pose_model  = self.load_from_config(config, model_type='pose')

        if self.depth_model == None or self.pose_model == None:
            assert("Config file format is incorrect: Take a look at example \
                    config files")
        
        # if checkpoint exists, load weights
        # else init weights (xavier: uniformly distributed)

        # init dataset (train, validation, test)

        # init optimiser and LR shcheduler

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