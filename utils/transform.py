#!/usr/bin/env python

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

pixel_coords = None

class Transform():

    def __init__(self, P, Tx, img_width, img_height):
        self.P      = P
        self.Tx     = Tx
        self.width  = img_width
        self.height = img_height

    def set_id_grid(self, depth):
        global pixel_coords
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)

        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    def inverse_rigid_trans_np(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

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
    
    def project_img_to_cam(self, depth, intrinsics_inv):
        global pixel_coords
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """

        b, h, w = depth.size()
        if (pixel_coords is None) or pixel_coords.size(2) < h:
            self.set_id_grid(depth)
        current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        current_pixel_coords = current_pixel_coords.type(torch.cuda.DoubleTensor)

        cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)
       
    def project_velo_to_img_np(self, point_cloud):
        ''' projects point cloud to image
        '''
        
        assert Tx == None, 'Velodyne transformation matrix cannot be None'

        # scan velodyne points and remove reflectance 
        point_cloud = point_cloud[:, :3]

        depth_array = np.zeros((self.width, self.height))

        # transform points

        # TODO: Test the below
        # depth_array = np.matmul(pnt_cloud, T.t()).clamp_(min=0)

        for pnt in point_cloud:

            # calculate velo distance
            x = pnt[0]
            y = pnt[1]
            z = pnt[2]
            dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            
            # make point homogeneous
            pnt = pnt.reshape(3, 1)
            pnt = np.vstack((pnt, [1]))

            # velo-to-cam
            xyz_pnt = np.matmul(self.Tx, pnt)

            # cam-to-img
            uv_pnt  = np.matmul(self.P, xyz_pnt)
            uv_pnt = (uv_pnt/uv_pnt[2]).reshape(-1) # scale adjustment

            if uv_pnt[0] >= 0 and uv_pnt[0] < self.width and \
                        uv_pnt[1] >= 0 and uv_pnt[1] < self.height and dist <= 120 and x > 0:

                depth_array[int(uv_pnt[0])][int(uv_pnt[1])] = xyz_pnt[2]

        # depth
        depth_img = np.transpose(depth_array)

        return depth_img

    def project_img_to_cam_np(self, depth_img):
        rows, cols = depth_img.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth_img])
        points = points.reshape((3, -1))
        uv_depth = points.T

        # Camera intrinsics and extrinsics
        c_u = self.P[0, 2]
        c_v = self.P[1, 2]
        f_u = self.P[0, 0]
        f_v = self.P[1, 1]
        b_x = self.P[0, 3] / (-f_u)  # relative
        b_y = self.P[1, 3] / (-f_v)

        # image to cam transform
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
        y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
        pts_3d_cam = np.zeros((n, 3))
        pts_3d_cam[:, 0] = x
        pts_3d_cam[:, 1] = y
        pts_3d_cam[:, 2] = uv_depth[:, 2]

        return pts_3d_cam

    def project_img_to_velo_np(self, depth_img):

        assert Tx == None, 'Velodyne transformation matrix cannot be None'
        
        points_3d_cam = self.project_img_to_cam_np(depth_img)

        T_inv =self.inverse_rigid_trans(self.Tx)

        n = pts_3d_cam.shape[0]
        pts_3d_hom = np.hstack((pts_3d_cam, np.ones((n, 1))))

        # cam to velodyne transform
        cloud = np.matmul(pts_3d_cam, np.transpose(T_inv))

        max_high = 1 # meters

        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        cloud  = cloud[valid]

        return cloud[valid]
