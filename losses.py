import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from geometry.pose_geometry import inverse_warp, disp_to_depth

from utils.transforms import UnNormalize

class SSIM:
    def standard_loss(self, x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
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
        #self.L2   = nn.MSELoss()

        self.unnormalize  =  UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.clip_loss = 0.5

    def compute_photometric_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """

        l1_loss = torch.abs(target - pred)
        # l1_loss  = abs_diff.mean(1, True).squeeze().mean(-1).mean(-1) # pytorch multidim reduce issue

        if no_ssim:
            photometric_loss = l1_loss
        else:
            ssim_loss = self.SSIM.standard_loss(pred, target)
            photometric_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        # Clip loss
        mean, std = photometric_loss.mean(), photometric_loss.std()
        photometric_loss = torch.clamp(
            photometric_loss, max=float(mean + self.clip_loss * std))

        return photometric_loss

    def multiview_reprojection_loss(self, tgt_img, ref_imgs, depth, poses, intrinsics, mode='min'):
        '''
        This is the multiview photometric loss that
        uses SSIM appearance matching.

        mode: mean / min
        '''

        def reduce_loss(tensor):
            tensor, _ = torch.max(tensor, dim=1)
            return tensor.mean(1).mean(-1)

        def plot_mask(mu, min_rpl):
            img  = np.transpose(self.unnormalize(tgt_img[0].squeeze()).cpu().detach().numpy(), (1, 2, 0))
            refs = [np.transpose(self.unnormalize(img[0].squeeze()).cpu().detach().numpy(), (1, 2, 0)) for img in ref_imgs]
            mu   = np.transpose(mu[0].squeeze().cpu().detach().numpy(), (1, 2, 0))
            min_rpl = np.transpose(min_rpl[0].squeeze().cpu().detach().numpy(), (1, 2, 0)) 
            
            arr = np.max(mu * min_rpl, axis=2)

            plt.imshow(arr)
            plt.show()
        
        def save_img(proj_img):
            img = proj_img.clone()
            img = img.cpu().detach().numpy()
            #img = np.transpose(img, (1, 2, 0))
            img = 0.44 + (0.2 * img)
            plt.imsave('./images/warping/0.png', img)

        # split poses
        poses_t_0 = poses[:, 0, :]
        poses_t_2 = poses[:, 1, :]
        poses     = [poses_t_0, poses_t_2] 
        
        print(poses)
        '''
        depth = depth[0]
        depth = depth.squeeze()
        '''
        reprojection_losses = []
        automasking_loss    = []
        projected_lst      = []
        for D in depth:
            _, _, H, W = depth[0].shape

            # Resize all D to 
            # disp[0].shape
            if D.shape[-1] != W:
                D = F.interpolate(D, [H, W], mode='bilinear', align_corners=False)

            D = D.squeeze()
            # save_img(D[0])


            # inverse warp from 
            projected_imgs = [inverse_warp(ref_img, D, pose, intrinsics) for ref_img, pose in zip(ref_imgs, poses)]
            projected_lst.append(projected_imgs)
            #save_img(projected_imgs[0][0])

            # reprojection between projected and target
            reprojection_losses.append([self.compute_photometric_loss(proj_img, tgt_img, no_ssim=True) for proj_img in projected_imgs])

            # reprojection between reference and target
            #automasking_loss.append([self.compute_photometric_loss(ref_img, tgt_img) for ref_img in ref_imgs])
        
        loss = []
        for rp_loss, proj_imgs in zip(reprojection_losses, projected_lst) :
            if mode == 'min':
                # element-wise minimum
                #min_rpl           = torch.minimum(rp_loss[0], rp_loss[1])
                #min_automask_loss = torch.minimum(auto_loss[0], auto_loss[1])
                mean_rpl          = torch.mean(torch.stack(rp_loss))
                print('min_rpl', mean_rpl)

                # binary automask
                # mu = torch.where(min_rpl < min_automask_loss, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())

                # visualise mask
                # plot_mask(mu, min_rpl)
        
                #batch_multiview_loss = reduce_loss(mean_rpl) # mu * min_rpl for mask
        
                #loss.append(batch_multiview_loss.mean())
                loss.append(mean_rpl)
            elif mode == 'mse':
                mse_loss  = self.L2(proj_imgs[0].type(torch.cuda.DoubleTensor), tgt_img.type(torch.cuda.DoubleTensor))
                mse_loss += self.L2(proj_imgs[1].type(torch.cuda.DoubleTensor), tgt_img.type(torch.cuda.DoubleTensor))
                loss.append(mse_loss / 2.0)
            elif mode == 'l1':
                l1_loss  = self.L1(proj_imgs[0].type(torch.cuda.DoubleTensor), tgt_img.type(torch.cuda.DoubleTensor))
                l1_loss += self.L1(proj_imgs[1].type(torch.cuda.DoubleTensor), tgt_img.type(torch.cuda.DoubleTensor))
                loss.append(l1_loss / 2.0)
            else:
                assert("different losses not implmented")
        return sum(loss) / len(depth)

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

    def create_mul_mask(self, tensor):
    
        mul_mask = torch.ones_like(tensor)
        
        C, H, W     = mul_mask.shape
        c_h, c_w = H//2, W//2
        
        mul_mask[:, c_h-150:c_h+150, c_w-500:c_w+500] = 2
        mul_mask[:, c_h-100:c_h+100, c_w-300:c_w+300] = 5
        mul_mask[:, c_h-50:c_h+50, c_w-100:c_w+100]   = 10
        
        return mul_mask

    def forward(self, tgt_img, ref_imgs, disparity, poses, intrinsics, gt):
        
        # create depth from disparity
        depth = disp_to_depth(disparity)

        loss_mam    = self.multiview_reprojection_loss(tgt_img, ref_imgs, depth, poses, intrinsics, mode='min')

        loss_smooth = self.smooth_loss(depth)

        return [loss_mam, loss_smooth]

    

        









