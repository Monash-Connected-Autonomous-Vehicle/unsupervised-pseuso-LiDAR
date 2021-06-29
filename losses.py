import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.pose_geometry import inverse_warp

# TODO:
#  1. Find losses to be made
#  2. Build multiview photomentric loss (appearance matching loss)
#  3. Add minimum per pixel photometric loss
#  4. Build depth smoothness loss
#  5. Add velocity supervision loss


class Losses:

    def SSIM(self, x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
        """
        Structural SIMilarity (SSIM) distance between two images.
        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        C1,C2 : float
            SSIM parameters
        kernel_size,stride : int
            Convolutional parameters
        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM distance
        """
        pool2d = nn.AvgPool2d(kernel_size, stride=stride)
        refl = nn.ReflectionPad2d(1)

        x, y = refl(x), refl(y)
        mu_x = pool2d(x)
        mu_y = pool2d(y)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = pool2d(x.pow(2)) - mu_x_sq
        sigma_y = pool2d(y.pow(2)) - mu_y_sq
        sigma_xy = pool2d(x * y) - mu_x_mu_y
        v1 = 2 * sigma_xy + C2
        v2 = sigma_x + sigma_y + C2

        ssim_n = (2 * mu_x_mu_y + C1) * v1
        ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
        ssim = ssim_n / ssim_d

        return ssim

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=120.0):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth

        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def compute_reprojection_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.SSIM(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def multiview_appearence_matching(self, tgt_img, ref_imgs, disparity, poses, intrinsics, mode='bilinear'):
        '''
        This is the multiview photometric loss that
        uses SSIM appearance matching as atated in
        '''
        # project dept to 3D
        # depth content size are as follows
        # torch.Size([4, 1, 375, 1242])
        # torch.Size([4, 1, 188, 621])
        # torch.Size([4, 1, 94, 311])
        # torch.Size([4, 1, 47, 156])
        depth  = self.disp_to_depth(disparity[0])

        # split poses
        poses_t_0 = poses[:, 0, :]
        poses_t_2 = poses[:, 1, :]
        intrinsics = intrinsics[:, :3, :3]

        # do an inverse warp
        projected_img_t_0, valid_points_t_0 = inverse_warp(ref_imgs[0], depth, poses_t_0, intrinsics)
        projected_img_t_2, valid_points_t_2 = inverse_warp(ref_imgs[1], depth, poses_t_2, intrinsics)



        
        
    def smooth_loss(pred_map):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1.

        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
            weight /= 2.3  # don't ask me why it works better
        return loss

    

        









