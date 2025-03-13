import torch
from cnn import DigitTrainer


handle_trainer = DigitTrainer(64, "./data")
handle_trainer.graph_build()
trainer_model = handle_trainer.train_beta(15)

torch.save(trainer_model, "mnist_cnn.pth")
# 预测图片5数字
handle_trainer.prediction(5)
