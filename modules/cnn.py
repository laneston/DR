import torch
import torch.nn as nn


# 定义神经网络模型
class CNNConv2d(nn.Module):
    def __init__(self):
        super(CNNConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, 3, 1
        )  # 卷积层1：输入通道1（灰度图），输出通道32，3x3卷积核，步长1
        self.conv2 = nn.Conv2d(
            32, 64, 3, 1
        )  # 卷积层2：输入通道32，输出通道64，3x3卷积核，步长1
        # self.dropout1 = nn.Dropout2d(0.25)               # Dropout层：防止过拟合，丢弃率25%
        # self.dropout2 = nn.Dropout2d(0.5)
        self.dropout1 = nn.Dropout(0.25)  # Dropout层：防止过拟合，丢弃率25%
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            9216, 128
        )  # 全连接层1：输入9216维，输出128维（需根据特征图尺寸计算）
        self.fc2 = nn.Linear(128, 10)  # 全连接层2：输出10维（对应0-9数字分类）

    def forward(self, x):
        x = self.conv1(x)  # 第一次卷积 + ReLU激活 (batch_size, 32, 26, 26)
        x = nn.functional.relu(x)
        x = self.conv2(x)  # 第二次卷积 + ReLU激活 (batch_size, 64, 24, 24)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(
            x, 2
        )  # 最大池化（下采样）     (batch_size, 64, 12, 12)
        x = self.dropout1(x)  # Dropout正则化
        x = torch.flatten(x, 1)  # 展平特征图 (batch_size, 64x12x12=9216)
        x = self.fc1(x)  # 全连接层1 + ReLU激活
        x = nn.functional.relu(x)
        x = self.dropout2(x)  # 第二次Dropout（丢弃率50%）
        x = self.fc2(x)  # 输出层 + log_softmax（配合NLLLoss损失函数）
        return nn.functional.log_softmax(x, dim=1)
