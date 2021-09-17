#!/usr/bin/env python

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
        Xc = X.view(B, 3, -1)
        
        #collect rot and trans matrix
        rot   = Tcw[:, :, :3]
        trans = Tcw[:, :, -1:]
        
        # transform each poin
        Xc = rot.type(torch.cuda.DoubleTensor) @ Xc.type(torch.cuda.DoubleTensor)
        Xc = K.type(torch.cuda.DoubleTensor) @ Xc
        Xc = Xc + trans.type(torch.cuda.DoubleTensor)
        
        
        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        Xnorm = 2 * (X / Z) / (W - 1) - 1.
        Ynorm = 2 * (Y / Z) / (H - 1) - 1.

        # Clamp out-of-bounds pixels
        # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
        # Xnorm[Xmask] = 2.
        # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
        # Ynorm[Ymask] = 2.

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)

        
        