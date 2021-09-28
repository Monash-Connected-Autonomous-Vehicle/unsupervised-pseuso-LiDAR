#!/usr/bin/env python

from numpy.core.numeric import identity
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class Transform():

    def meshgrid(self, B, H, W, dtype, device, normalized=False):
        '''
        Create meshgrid with a specific resolution
        Parameters
        ----------
        B : int
            Batch size
        H : int
            Height size
        W : int
            Width size
        dtype : torch.dtype
            Meshgrid type
        device : torch.device
            Meshgrid device
        normalized : bool
            True if grid is normalized between -1 and 1
        Returns
        -------
        xs : torch.Tensor [B,1,W]
            Meshgrid in dimension x
        ys : torch.Tensor [B,H,1]
            Meshgrid in dimension y
        '''
        if normalized:
            xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        else:
            xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
            ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
        ys, xs = torch.meshgrid([ys, xs])
        return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])

    def image_grid(self, B, H, W, dtype, device, normalized=False):
        """
        Create an image grid with a specific resolution
        Parameters
        ----------
        B : int
            Batch size
        H : int
            Height size
        W : int
            Width size
        dtype : torch.dtype
            Meshgrid type
        device : torch.device
            Meshgrid device
        normalized : bool
            True if grid is normalized between -1 and 1
        Returns
        -------
        grid : torch.Tensor [B,3,H,W]
            Image grid containing a meshgrid in x, y and 1
        """
        xs, ys = self.meshgrid(B, H, W, dtype, device, normalized=normalized)
        ones = torch.ones_like(xs)
        grid = torch.stack([xs, ys, ones], dim=1)
        return grid

    def reconstruct(self, depth, K):
        """
        Reconstructs pixel-wise 3D points from a depth map.
        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world
        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        depth = depth.unsqueeze(1)
        B, C, H, W = depth.shape
        assert C == 1
        
        Kinv = K.inverse().float()
        
        # Create flat index grid
        grid = self.image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        # Estimate the outward rays in the camera frame
        xnorm = (Kinv.bmm(flat_grid))
        xnorm = xnorm.view(B, 3, H, W)

        # Scale rays to metric depth
        Xc = xnorm * depth

        return Xc

    def k_hom(self, K):
        K_hom = torch.eye(4)
        K_hom = K_hom.reshape((1, 4, 4))
        K_hom = K_hom.repeat(4, 1, 1).to(device=K.device)
        K_hom[:, :3, :3] = K.clone()
        return K_hom

    def project(self, X, K, Tcw):
        """
        Projects 3D points onto the image plane
        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world
        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        
        # flatten camera coords
        Xc   = X.view(B, 3, -1)

        #test
        ones   = torch.ones(1, Xc.shape[-1]).repeat(B, 1, 1).cuda()
        Xc_hom = torch.cat([Xc, ones], 1)

        K_hom = self.k_hom(K)

        Tx = (K_hom @ Tcw)[:, :3, :]

        cam_points = Tx @ Xc_hom
    
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-5)
        pix_coords = pix_coords.view(B, 2, H, W)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= W - 1
        pix_coords[..., 1] /= H - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords

        
        