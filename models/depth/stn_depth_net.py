import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, zeros_, constant_
import sys


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True)
    )

def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.GroupNorm(16, out_planes),
        nn.ReLU(inplace=True)
    )

'''
TODO: 
    1. Skip connections
    2. 
'''
class StnDispNet(nn.Module):
    '''
        A spatial transformer disparity estimation
        network. 
    '''
    def __init__(self):
        super(StnDispNet, self).__init__()

        conv_planes = [32, 64, 128, 256]
        self.conv1  = downsample_conv(3             , conv_planes[0])
        self.conv2  = downsample_conv(conv_planes[0], conv_planes[1])
        self.conv3  = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4  = downsample_conv(conv_planes[2], conv_planes[3])

        upconv_planes = [256, 128, 64, 32, 16]
        self.upconv_1 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv_2 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv_3 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv_4 = upconv(upconv_planes[3], upconv_planes[4])

        self.predict = predict_disp(upconv_planes[4])

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            downsample_conv(3, 16),
            downsample_conv(16, 32),
            downsample_conv(32, 32),
            downsample_conv(32, 32), 
            downsample_conv(32, 32)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 12 * 40, 1280),
            nn.ReLU(True),
            nn.Linear(1280, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 12 * 40)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                constant_(m.bias, 0)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[6].weight.data.zero_()
        self.fc_loc[6].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # transform the input
        #x = self.stn(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        out = self.upconv_1(out)
        out = self.upconv_2(out)
        out = self.upconv_3(out)
        out = self.upconv_4(out)
        
        out = self.predict(out)

        return [out]