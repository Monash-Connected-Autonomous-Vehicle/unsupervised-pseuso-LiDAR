#!/usr/bin/env python

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class Transform():

    def __init__(self, P, Tx, img_width, img_height):
        self.P      = P
        self.Tx     = Tx
        self.width  = img_width
        self.height = img_height
        self.pixel_coords = None

    def set_id_grid(self, depth):
        
        b, _,h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

        return pixel_coords

    def project_cam_to_img(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]

        return pixel_coords.reshape(b, h, w, 2)
    
    def project_img_to_cam(self, depth, K):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, _,  h, w = depth.size()

        if (self.pixel_coords is None) or self.pixel_coords.size(2) < h:
            self.pixel_coords = self.set_id_grid(depth)

        current_pixel_coords = self.pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        current_pixel_coords = current_pixel_coords.type(torch.cuda.DoubleTensor)

        K_inv = K.inverse()

        cam_coords = (K_inv @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth
    
