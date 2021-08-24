import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.pose_geometry import inverse_warp, disp_to_depth

# TODO:
#  1. Add velocity supervision loss

class SSIM:

    def standard(self, x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
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

        # SSIM is actually in the [-1, 1] range.
        # The clamping is here to avoid possible 
        # overflow in case SSIM_d becomes too small 
        # resulting in precision issues.
        return torch.clamp((1. - ssim) / 2., 0., 1.)

class Losses:

    def __init__(self):
        self.SSIM = SSIM()

    def compute_photometric_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        l1_loss = torch.abs(target - pred)
        # l1_loss  = abs_diff.mean(1, True).squeeze().mean(-1).mean(-1) # pytorch multidim reduce issue

        if no_ssim:
            photometric_loss = l1_loss
        else:
            ssim_loss = self.SSIM.standard(pred, target)
            photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        # Clip loss
        # if 1 > 0.0:
        #     for i in range(reprojection_loss.shape):
        #         print(i)
                # mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                # photometric_loss[i] = torch.clamp(
                #     photometric_loss[i], max=float(mean + self.clip_loss * std))

        return photometric_loss

    def multiview_reprojection_loss(self, tgt_img, ref_imgs, depth, poses, intrinsics, mode='min'):
        '''
        This is the multiview photometric loss that
        uses SSIM appearance matching.

        mode: mean / min
        '''

        def reduce_loss(tensor):
            return tensor.mean(1, True).squeeze().mean(-1).mean(-1)


        # split poses
        poses_t_0 = poses[:, 0, :]
        poses_t_2 = poses[:, 1, :]
        poses     = [poses_t_0, poses_t_2] 
        
        # inverse warp from 
        projected_imgs = [inverse_warp(ref_img, depth, pose, intrinsics) for ref_img, pose in zip(ref_imgs, poses)]

        # reprojection between projected and target
        reprojection_losses = [self.compute_photometric_loss(proj_img, tgt_img) for proj_img in projected_imgs]

        # reprojection between reference and target
        automasking_loss = [self.compute_photometric_loss(ref_img, tgt_img) for ref_img in ref_imgs]

        if mode == 'min':
            # element-wise minimum
            min_rpl      = torch.minimum(reprojection_losses[0], reprojection_losses[1])
            min_automask_loss = torch.minimum(automasking_loss[0], automasking_loss[1])
            
            # binary automask
            mu = torch.where(min_rpl < min_automask_loss, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
            
            batch_multiview_loss = reduce_loss(mu * min_rpl)
            
            return batch_multiview_loss.mean()
        else:
            assert("different losses not implmented")

    def smooth_loss(self, pred_map):
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

    def forward(self, tgt_img, ref_imgs, disparity, poses, intrinsics):
        
        # project dept to 3D
        # depth content size are as follows
        # torch.Size([4, 1, 375, 1242])
        # torch.Size([4, 1, 188, 621])
        # torch.Size([4, 1, 94, 311])
        # torch.Size([4, 1, 47, 156])
        depth = disp_to_depth(disparity[0])

        loss_mam    = self.multiview_reprojection_loss(tgt_img, ref_imgs, depth, poses, intrinsics, mode='min')
        loss_smooth = self.smooth_loss(depth)

        return [loss_mam, loss_smooth]

    

        









