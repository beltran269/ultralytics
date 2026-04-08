# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation trainer for YOLO models."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


class DepthTrainer(yolo.detect.DetectionTrainer):
    """Trainer for YOLO depth estimation models.

    Extends DetectionTrainer with depth-specific model initialization,
    dataset loading, and validation.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26n-depth.yaml", data="nyu-depth.yaml", epochs=100)
        >>> trainer = DepthTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DepthTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a DepthModel initialized with the given config and weights."""
        model = DepthModel(cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a DepthDataset for training or validation."""
        from ultralytics.data.dataset import DepthDataset

        gs = max(int(unwrap_model(self.model).stride.max() if hasattr(unwrap_model(self.model), "stride") else 32), 32)
        return DepthDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task="depth",
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return a DepthValidator for model validation."""
        self.loss_names = "silog", "l1"
        return yolo.depth.DepthValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
