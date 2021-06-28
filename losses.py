import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.transform import Transform
from utils.pose_geometry import pose_vec2mat

# TODO:
#  1. Find losses to be made
#  2. Build multiview photomentric loss (appearance matching loss)
#  3. Add minimum per pixel photometric loss
#  4. Build depth smoothness loss
#  5. Add velocity supervision loss


class Losses:
    def __init__(self, config):
        self.loss_use = config['action']['loss']
    
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

    def disp_to_depth(self, disp, min_depth=0, max_depth=120):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def multiview_appearence_matching(self, tgt_img, ref_imgs, disp, poses, P, mode='bilinear'):
        '''
        This is the multiview photometric loss that
        uses SSIM appearance matching as atated in
        PackNet SfM : https://arxiv.org/pdf/1905.02693.pdf
        '''

        warper = Transform(P, None, tgt_img.shape[0], tgt_img.shape[1])

        # convert poses to mat (euler form)
        pose_mtxs = [pose_vec2mat(pose) for pose in poses]

        # project dept to 3D
        depth  = self.disp_to_depth(disp)
        img_to_cam = warper.project_img_to_cam(depth)

        # transform to target image and sample
        projected_imgs = []
        valid_pnts     = []
        for ind in range(len(poses)):
            proj_cam_to_src_pixel = P @ pose_mtxs[ind]  # [B, 3, 4]
            rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
            src_pixel_coords = project_cam_to_img(img_to_cam, rot, tr)
            projected_img = F.grid_sample(tgt_img, src_pixel_coords, padding_mode='zeros', align_corners=True)
            valid_p = src_pixel_coords.abs().max(dim=-1)[0] <= 1

            projected_imgs.append(projected_img)
            valid_pnts.append(valid_p)
        
        # this should be inverse warp
        return projected_imgs, valid_pnts
        
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

    

        









