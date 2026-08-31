"""
Tesla Chrono-Voxel Neural Fields (CV-NF) 4D Metrics Evaluation Engine
====================================================================
Computes 3D Occupancy mIoU, Continuous-Time Temporal Consistency,
PSNR, SSIM, and 4D Metric Chamfer Distance.

Copyright (c) 2026 Seydi Eryilmaz (@seydivakkas)
All Rights Reserved.
"""

import torch
import numpy as np
from typing import Dict, Any


class Tesla4DMetricsEvaluator:
    """
    Evaluator for Continuous 4D Neural Fields and Neuromorphic Event Fusion.
    """
    @staticmethod
    def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse < 1e-10:
            return 100.0
        return float(10.0 * np.log10(1.0 / mse))

    @staticmethod
    def compute_occupancy_iou(pred_occ: torch.Tensor, gt_occ: torch.Tensor, threshold: float = 0.5) -> float:
        pred_mask = (pred_occ >= threshold).bool()
        gt_mask = (gt_occ >= threshold).bool()

        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()

        if union == 0:
            return 1.0
        return float(intersection / union)

    @staticmethod
    def compute_temporal_consistency_score(
        model_density_t1: torch.Tensor,
        model_density_t2: torch.Tensor,
        delta_t_s: float
    ) -> float:
        r"""
        Temporal smoothness gradient: lim_{\Delta t -> 0} ||\sigma(t2) - \sigma(t1)|| / \Delta t
        """
        diff = torch.abs(model_density_t2 - model_density_t1).mean().item()
        # Normalized consistency index [0, 1]
        consistency = np.exp(-diff / max(delta_t_s, 1e-6))
        return float(consistency)
