"""
Tesla Chrono-Voxel Neural Fields (CV-NF) Self-Supervised Photometric Engine
==========================================================================
Differentiable Volume Raymarching, Structural Similarity (SSIM) Loss,
Inverse Warping, and Multi-View Photometric Consistency without Human Labels.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index Measure (SSIM) Loss.
    Robust to extreme dynamic range transitions and HDR lighting changes.
    """
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.tensor([
            -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)
            for x in range(window_size)
        ]).exp()
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return torch.clamp((1.0 - ssim_map) / 2.0, 0.0, 1.0).mean()


class SelfSupervisedPhotometricEngine(nn.Module):
    """
    Self-Supervised Neural Rendering & Differentiable Reprojection Engine.
    Trains continuous 4D Occupancy from unlabeled temporal video frames.
    """
    def __init__(self, ssim_weight: float = 0.85, smooth_weight: float = 0.001):
        super().__init__()
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight
        self.l1_weight = 1.0 - ssim_weight
        self.smooth_weight = smooth_weight

    def differentiable_raymarch(
        self,
        sigmas: torch.Tensor,
        colors: torch.Tensor,
        z_vals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable Volume Integration (Native PyTorch Autograd implementation).
        sigmas: [B, H, W, N_samples]
        colors: [B, H, W, N_samples, 3]
        z_vals: [B, H, W, N_samples]
        returns: (rendered_rgb [B, H, W, 3], rendered_depth [B, H, W])
        """
        # Step intervals: delta_i = z_{i+1} - z_i
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        delta_inf = torch.full_like(deltas[..., :1], 1e5)
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        alphas = 1.0 - torch.exp(-sigmas * deltas)
        
        # Cumulative Transmittance: T_i = \prod_{j=1}^{i-1} (1 - alpha_j)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alphas[..., :1]), 1.0 - alphas + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        weights = transmittance * alphas

        rendered_rgb = torch.sum(weights.unsqueeze(-1) * colors, dim=-2)
        rendered_depth = torch.sum(weights * z_vals, dim=-1)

        return rendered_rgb, rendered_depth

    def inverse_warp(
        self,
        target_img: torch.Tensor,
        current_depth: torch.Tensor,
        pose_t_to_src: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Backproject current 2D pixels to 3D with estimated depth and project into target camera frame via SE(3).
        target_img: [B, 3, H, W]
        current_depth: [B, 1, H, W]
        pose_t_to_src: [B, 4, 4]
        intrinsics: [B, 3, 3]
        """
        B, _, H, W = target_img.shape
        device = target_img.device

        y, x = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=device),
            torch.arange(0, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        x = x.unsqueeze(0).repeat(B, 1, 1)
        y = y.unsqueeze(0).repeat(B, 1, 1)

        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)

        depth_squeezed = current_depth.squeeze(1)
        X = (x - cx) * depth_squeezed / (fx + 1e-7)
        Y = (y - cy) * depth_squeezed / (fy + 1e-7)
        Z = depth_squeezed

        pts_3d = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=1).view(B, 4, -1) # [B, 4, H*W]
        pts_src = torch.bmm(pose_t_to_src, pts_3d)

        # 3D -> 2D Reprojection
        z_src = pts_src[:, 2, :] + 1e-7
        P_x = pts_src[:, 0, :] * fx.view(B, 1) / z_src + cx.view(B, 1)
        P_y = pts_src[:, 1, :] * fy.view(B, 1) / z_src + cy.view(B, 1)

        norm_x = (2.0 * P_x / (W - 1.0)) - 1.0
        norm_y = (2.0 * P_y / (H - 1.0)) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).view(B, H, W, 2)

        return F.grid_sample(target_img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def edge_aware_smoothness_loss(self, depth: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Edge-aware depth gradient smoothness penalty."""
        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()

    def compute_self_supervised_loss(
        self,
        pred_sigmas: torch.Tensor,
        pred_colors: torch.Tensor,
        z_vals: torch.Tensor,
        gt_frame_curr: torch.Tensor,
        gt_frame_next: torch.Tensor,
        rel_pose_next: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates end-to-end self-supervised loss combining Neural Reconstruction and Photometric Reprojection.
        """
        # 1. Differentiable Raymarching
        rendered_rgb, rendered_depth = self.differentiable_raymarch(pred_sigmas, pred_colors, z_vals)
        rendered_rgb_nchw = rendered_rgb.permute(0, 3, 1, 2)
        depth_nchw = rendered_depth.unsqueeze(1)

        # 2. Render vs Current Frame
        l1_rec = F.l1_loss(rendered_rgb_nchw, gt_frame_curr)
        ssim_rec = self.ssim(rendered_rgb_nchw, gt_frame_curr)
        rec_loss = self.l1_weight * l1_rec + self.ssim_weight * ssim_rec

        # 3. Next Frame Inverse Warping
        warped_next = self.inverse_warp(gt_frame_next, depth_nchw, rel_pose_next, intrinsics)
        l1_warp = F.l1_loss(warped_next, gt_frame_curr)
        ssim_warp = self.ssim(warped_next, gt_frame_curr)
        photometric_loss = self.l1_weight * l1_warp + self.ssim_weight * ssim_warp

        # 4. Smoothness
        smooth_loss = self.edge_aware_smoothness_loss(depth_nchw, gt_frame_curr)

        total_loss = rec_loss + photometric_loss + self.smooth_weight * smooth_loss

        return {
            "loss/total": total_loss,
            "loss/reconstruction": rec_loss,
            "loss/photometric": photometric_loss,
            "loss/smoothness": smooth_loss
        }
