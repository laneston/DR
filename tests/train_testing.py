import torch
import sys
from pathlib import Path


current_path = Path(__file__).resolve()
project_parent = (
    current_path.parent.parent
)  # Adjust the times of .parent according to the actual structure


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"destination directory does not exist: {project_parent}")

try:
    # from modules import CNNConv2dTrainer
    from modules import EfficientnetB0Trainer

except ImportError as e:
    print(f"Import failed: {e}")

"""
chose the model according to your need.

CNNConv2dTrainer: convolutional neural network.

EfficientnetB0Trainer: EfficientNet achieves a breakthrough balance between model efficiency 
and performance through the collaborative design of composite scaling and MBConv modules. 
Its serialized architecture adapts to the full scenario requirements from edge devices to cloud servers, 
and has become one of the preferred backbone networks for computer vision tasks in the industrial sector.
"""

# handle_trainer = CNNConv2dTrainer(64, "./data")
handle_trainer = EfficientnetB0Trainer(16, "./data")
handle_trainer.graph_build()
__t_model, __t_optimizer = handle_trainer.train_beta(5)


torch.save(
    {
        "model_state_dict": __t_model,
        "optimizer_state_dict": __t_optimizer,
    },
    "modelspath/mnist_efficientnet.pth",
)
# Predict the number of the fifth image
handle_trainer.prediction(5)
