# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Computes standard depth metrics: delta1, abs_rel, rmse, silog.
    Uses validation loss as the primary training signal.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
        self.depth_metrics = []  # accumulated per-image metrics

    def init_metrics(self, model):
        """Initialize depth metrics."""
        self.depth_metrics = []

    def preprocess(self, batch):
        """Preprocess batch — move to device and handle precision."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        # Images are already normalized [0,1] by DepthFormat; just handle half precision
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        if "depth" in batch:
            batch["depth"] = batch["depth"].float()  # depth always float32
        return batch

    def postprocess(self, preds):
        """No NMS needed for depth — return predictions as-is."""
        return preds

    def update_metrics(self, preds, batch):
        """Compute per-batch depth metrics."""
        if "depth" not in batch:
            return

        # Get predicted depth
        if isinstance(preds, dict):
            pred_depth = preds.get("depth", preds.get("proto"))
        elif isinstance(preds, torch.Tensor):
            pred_depth = preds
        elif isinstance(preds, (tuple, list)):
            pred_depth = preds[0] if isinstance(preds[0], torch.Tensor) else preds
        else:
            return

        gt_depth = batch["depth"]  # (B, 1, H, W) or (B, H, W)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)

        # Resize pred to match GT if needed
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)

        # Compute per-image metrics
        for i in range(pred_depth.shape[0]):
            p = pred_depth[i].squeeze().cpu().numpy()
            g = gt_depth[i].squeeze().cpu().numpy()

            # Valid mask
            mask = g > 0.001
            if mask.sum() < 100:
                continue

            p_valid = np.clip(p[mask], 0.001, None)
            g_valid = g[mask]

            # Threshold accuracy
            thresh = np.maximum(p_valid / g_valid, g_valid / p_valid)
            delta1 = float((thresh < 1.25).mean())
            delta2 = float((thresh < 1.25 ** 2).mean())
            delta3 = float((thresh < 1.25 ** 3).mean())

            # Error metrics
            abs_rel = float(np.mean(np.abs(p_valid - g_valid) / g_valid))
            rmse = float(np.sqrt(np.mean((p_valid - g_valid) ** 2)))

            # SILog
            log_diff = np.log(p_valid) - np.log(g_valid)
            silog = float(np.sqrt(np.mean(log_diff ** 2) - 0.5 * np.mean(log_diff) ** 2) * 100)

            self.depth_metrics.append({
                "delta1": delta1,
                "delta2": delta2,
                "delta3": delta3,
                "abs_rel": abs_rel,
                "rmse": rmse,
                "silog": silog,
            })

    def get_stats(self):
        """Aggregate depth metrics across all batches."""
        if not self.depth_metrics:
            return {}

        stats = {}
        for key in self.depth_metrics[0]:
            vals = [m[key] for m in self.depth_metrics]
            stats[f"metrics/{key}"] = float(np.mean(vals))

        # Use delta1 as fitness (higher is better)
        stats["fitness"] = stats.get("metrics/delta1", 0.0)
        return stats

    def print_results(self):
        """Print depth validation results."""
        if not self.depth_metrics:
            return
        n = len(self.depth_metrics)
        d1 = np.mean([m["delta1"] for m in self.depth_metrics])
        ar = np.mean([m["abs_rel"] for m in self.depth_metrics])
        rmse = np.mean([m["rmse"] for m in self.depth_metrics])
        silog = np.mean([m["silog"] for m in self.depth_metrics])
        LOGGER.info(f"Depth val ({n} images): delta1={d1:.4f}  abs_rel={ar:.4f}  rmse={rmse:.4f}  silog={silog:.2f}")

    def finalize_metrics(self):
        """No-op for depth (metrics already finalized in get_stats)."""
        pass

    def get_desc(self):
        """Return description for progress bar."""
        return f"{'Class':>22}{'Images':>11}{'delta1':>11}{'abs_rel':>11}{'rmse':>11}{'silog':>11}"

    def plot_predictions(self, batch, preds, ni):
        """Skip detection-style prediction plotting for depth."""
        pass

    def plot_val_samples(self, batch, ni):
        """Skip detection-style sample plotting for depth."""
        pass
