#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from .calibration import Calibration

class Transform(Calibration):

    def __init__(self, kitti_filepath, img_width, img_height):
        super(Transform, self).__init__(kitti_filepath)
        self.width  = img_width
        self.height = img_height

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def project_velo_to_img(self, point_cloud):
        ''' projects point cloud to image
        '''
        
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
            xyz_pnt = np.matmul(self.T, pnt)

            # cam-to-img
            uv_pnt  = np.matmul(self.P, xyz_pnt)
            uv_pnt = (uv_pnt/uv_pnt[2]).reshape(-1) # scale adjustment

            if uv_pnt[0] >= 0 and uv_pnt[0] < self.width and \
                        uv_pnt[1] >= 0 and uv_pnt[1] < self.height and dist <= 120 and x > 0:

                depth_array[int(uv_pnt[0])][int(uv_pnt[1])] = xyz_pnt[2]

        # depth
        depth_img = np.transpose(depth_array)

        return depth_img
    
    def project_img_to_velo(self, depth_img):
        
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

        T_inv =self.inverse_rigid_trans(self.Tx)

        n = pts_3d_cam.shape[0]
        pts_3d_hom = np.hstack((pts_3d_cam, np.ones((n, 1))))

        # cam to velodyne transform
        cloud = np.matmul(pts_3d_cam, np.transpose(T_inv))

        max_high = 1 # meters

        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        cloud  = cloud[valid]

        return cloud[valid]
