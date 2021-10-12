import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_,constant_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseFc(nn.Module):

    def __init__(self, nb_ref_imgs=2):
        super(PoseFc, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        
        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
        
        # Regressor for the 2 x 6 pose matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(12 * 3 * 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 12)
        )
        
        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                constant_(m.bias, 0)
        
        # set identity for pose
        # Initialize the weights/bias
        self.fc_loc[-1].weight.data.zero_()

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        
        pose = self.pose_pred(out_conv7)
        
        # replacing mean with learnable 
        # parameters
        pose = pose.view(-1, 12 * 3 * 10) #12 * 3 * 10
        pose = self.fc_loc(pose)
        pose = pose.view(pose.size(0), self.nb_ref_imgs, 6)
        pose[:, :, :3] = 0

        return pose