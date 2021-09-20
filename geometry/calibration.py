#!/usr/bin/env python

""" Helper methods for loading and parsing KITTI data.
"""

import numpy as np
import os
import math

class Calibration():
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
       
    """

    def __init__(self, kitti_filepath):

        self.kitti_filepath = kitti_filepath
        calib_velo_to_cam = self.read_calib_file(kitti_filepath + "calib_velo_to_cam.txt")
        calib_cam_to_cam  = self.read_calib_file(kitti_filepath + "calib_cam_to_cam.txt")
        calib_imu_to_velo = self.read_calib_file(kitti_filepath + "calib_imu_to_velo")

        # camera intrinsics
        self.K = calib_cam_to_cam["K_02"]
        
        # Projection matrix from rect camera coord to image2 coord
        self.P = calib_cam_to_cam["P_rect_02"].reshape(3, 4)

        # Rotation and Translation matrix (velodyne)
        self.R = calib_velo_to_cam["R"].reshape(3, 3)
        self.T = calib_velo_to_cam["T"].reshape(3, 1)

        # Rigid transform from Velodyne coord to reference camera coord
        self.Tx = np.concatenate((self.R, self.T), axis = 1)
        self.Tx = np.vstack([self.Tx, [0, 0, 0, 1]])

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """

        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data