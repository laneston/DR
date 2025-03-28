import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from torchviz import make_dot


# 修改EfficientNet模型
def efficientnet_b0_create():
    model = efficientnet_b0(weights=None)

    # 修改第一个卷积层适应单通道输入
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,  # 改为单通道
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    # 修改分类层
    model.classifier[1] = nn.Linear(
        in_features=model.classifier[1].in_features, out_features=10
    )
    return model


class EfficientnetB0Trainer:
    def __init__(self, batch_size, mnist_path):
        if (batch_size > 0) & (batch_size < 65535) & (batch_size % 8 == 0):
            self.__batch_size = batch_size
        else:
            raise TypeError(
                f"{batch_size} is not in compliance with regulations, should be rang 0 to 65535."
            )

        if type(mnist_path) == str:
            self.__mnist_path = mnist_path
        else:
            raise TypeError(
                f"{mnist_path} is not in compliance with regulations, should be string format."
            )

        # 设置随机种子保证可重复性
        torch.manual_seed(42)

        # 数据预处理
        self.__transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # EfficientNet需要较大输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081]),  # MNIST统计值
            ]
        )

        # 加载训练数据集
        self.train_dataset = torchvision.datasets.MNIST(
            root=mnist_path, train=True, download=True, transform=self.__transform
        )

        self.__train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        # 加载测试数据集
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.__mnist_path,
            train=False,
            download=True,
            transform=self.__transform,
        )
        self.__test_loader = DataLoader(
            self.test_dataset, batch_size=self.__batch_size, shuffle=True
        )

        # 检查可用设备
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.__model = efficientnet_b0_create().to(self.__device)

        # 定义损失函数和优化器
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.__optimizer, "max", patience=2
        )

    def graph_build(self):
        # 虚拟传入数值，生成计算图
        x = torch.randn(1, 1, 28, 28, requires_grad=True)  # 关键：requires_grad=True
        x = x.to(self.__device)

        # 生成计算图
        output = self.__model(x)
        graph = make_dot(
            output,
            params=dict(list(self.__model.named_parameters()) + [("input", x)]),
            show_attrs=True,
            show_saved=True,
        )

        # 保存为图片
        graph.render("efficientnet_b0_create_graph", format="svg", cleanup=True)
