import torch
import sys
from pathlib import Path


current_path = Path(__file__).resolve()
project_parent = current_path.parent.parent  # 根据实际结构调整.parent次数


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"目标目录不存在: {project_parent}")

try:
    from modules import CNNConv2dTrainer

except ImportError as e:
    print(f"导入失败: {e}")


handle_trainer = CNNConv2dTrainer(64, "./data")
handle_trainer.graph_build()
trainer_model = handle_trainer.train_beta(15)

torch.save(trainer_model, "modelspath/mnist_cnn.pth")
# 预测图片5数字
handle_trainer.prediction(5)
