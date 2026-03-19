"""
Faster R-CNN Detection Architecture.

Wraps ``torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`` with
automatic head replacement for custom class counts. The ``+1`` for the
background class is handled internally — callers pass only the number of
object categories.
"""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor


def build_fasterrcnn(
    num_classes: int,
    in_channels: int = 3,  # noqa: ARG001  # pragma: no mutate
    pretrained: bool = True,  # pragma: no mutate
) -> nn.Module:
    """
    Build a Faster R-CNN with ResNet-50-FPN v2 backbone.

    Loads the torchvision pre-built model and replaces the box predictor
    head to match the target number of classes. Background is added
    automatically (``num_classes + 1``).

    Args:
        num_classes: Number of object categories (excluding background).
        in_channels: Input channels (accepted for API compatibility,
            FasterRCNN uses RGB internally).
        pretrained: If True, load COCO-pretrained weights for the backbone.

    Returns:
        Faster R-CNN model with custom class head, on CPU.
    """
    weights = "DEFAULT" if pretrained else None  # pragma: no mutate
    model = cast(FasterRCNN, fasterrcnn_resnet50_fpn_v2(weights=weights))  # pragma: no mutate

    # Replace the classification head for the target number of classes
    # (+1 for background, which torchvision expects)
    in_features: int = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return cast(nn.Module, model)  # pragma: no mutate
