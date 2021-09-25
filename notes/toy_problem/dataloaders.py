import cv2
import glob
from copy import deepcopy
from itertools import islice
from collections import deque
from mat4py import loadmat
from scipy import linalg

from trainer import *

class dino_dataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.samples = []
        
        self.init_samples(2)
        
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
            
    def init_pose(self):
        data = loadmat('./utils/dino_Ps.mat')['P']
        data = np.array(data)
        return data
    
    def factor_P(self, P):
        """  
            Factorize the camera matrix into K,R,t as P = K[R|t]. 
        """

        # factor first 3*3 part
        K,R = linalg.rq(P[:,:3])

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1,1] *= -1

        self.K = torch.fron_numpy(np.dot(K,T))
        self.R = torch.from_numpy(np.dot(T,R)) # T is its own inverse
        self.t = torch.from_numpy(np.dot(linalg.inv(self.K),self.P[:,3]))

        return self.K, self.R, self.t
            
    def init_samples(self, seq_size):
        img_dir = './images/*.ppm'
        
        poses = self.init_pose()
        
        x = sorted(glob.glob(img_dir))
        
        sample = {}
        for ind, window in enumerate(self.sliding_window(x, seq_size)):
            sample['tgt']  = window[0]
            sample['ref']  = window[1]
            sample['oxts'] = poses[ind]
            
            self.samples.append(deepcopy(sample))
            
    def load_img(self, img_dir):
        img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)[:, :650]
        img = np.transpose(img, (2, 0, 1))

        # apply transforms
        for t in self.transforms:
            img = t(img)

        return img
        
        
    def __getitem__(self, indx):
        ret_sample = {}
        sample = self.samples[indx]
        
        ret_sample['tgt'] = self.load_img(sample['tgt'])
        ret_sample['ref'] = self.load_img(sample['ref'])

        # decompose projection matrix 
        ret_sample['K'], ret_sample['R'], ret_sample['t'] = self.factor_P(sample['oxts'])
        
        return ret_sample
        
    def __len__(self):
        return len(self.samples)