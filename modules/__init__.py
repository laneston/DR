# Ultralytics YOLO 🚀, AGPL-3.0 license

from .cnn import CNNConv2dTrainer, cnn_conv2d_pretrained
from .efficientnet import EfficientnetB0Trainer

__all__ = (
    "CNNConv2dTrainer",
    "process_image_to_mnist",
    "cnn_conv2d_pretrained",
    "EfficientnetB0Trainer",
)
