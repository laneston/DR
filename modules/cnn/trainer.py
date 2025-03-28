import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchviz import make_dot

from .cnn import CNNConv2d


class CNNConv2dTrainer:
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

        # 定义数据预处理
        self.__transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST数据的均值和标准差
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
        self.__model = CNNConv2d().to(self.__device)

        # 定义损失函数和优化器
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)

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
        graph.render("CNNConv2d_graph", format="svg", cleanup=True)

    # 训练函数
    def train(self, cycles):
        self.__model.train()
        for cycle in range(1, cycles + 1):
            for batch_idx, (data, target) in enumerate(self.__train_loader):
                data, target = data.to(self.__device), target.to(self.__device)
                """
                Clear gradient cache: PyTorch accumulates gradients by default. 
                If not manually reset, multiple backpropagation will cause gradient accumulation.
                Associated parameters: If using optimizers such as Adam or SGD, 
                this operation must be performed before each iteration
                """
                self.__optimizer.zero_grad()
                output = self.__model(data)
                # 计算损失函数值，衡量预测值与真实值的差异，驱动模型参数更新方向
                loss = self.__criterion(output, target)
                """
                Backpropagation: Automatically calculate the gradient of loss on model parameters.
                Bottom level mechanism: PyTorch's Autograd system constructs 
                computational graphs and calculates gradients
                """
                loss.backward()
                # 更新模型参数：根据梯度方向和优化器规则（如Adam的动量、自适应学习率）调整参数
                self.__optimizer.step()
                # 每100个批次输出一次进度和损失值，便于监控训练过程
                if batch_idx % 100 == 0:
                    print(
                        f"Train Epoch: {cycle} [{batch_idx * len(data)}/{len(self.__train_loader.dataset)}"
                        f" ({100. * batch_idx / len(self.__train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

        return self.__model.state_dict()

    def train_beta(self, cycles):
        for cycle in range(1, cycles + 1):
            self.__model.train()
            for batch_idx, (data, target) in enumerate(self.__train_loader):
                data, target = data.to(self.__device), target.to(self.__device)
                self.__optimizer.zero_grad()
                output = self.__model(data)
                # 计算损失函数值，衡量预测值与真实值的差异，驱动模型参数更新方向
                loss = self.__criterion(output, target)
                loss.backward()
                # 更新模型参数：根据梯度方向和优化器规则（如Adam的动量、自适应学习率）调整参数
                self.__optimizer.step()
                # 每100个批次输出一次进度和损失值，便于监控训练过程
                if batch_idx % 100 == 0:
                    print(
                        f"Train Epoch: {cycle} [{batch_idx * len(data)}/{len(self.__train_loader.dataset)}"
                        f" ({100. * batch_idx / len(self.__train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

            self.__model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.__test_loader:
                    data, target = data.to(self.__device), target.to(self.__device)

                    output = self.__model(data)
                    # test_loss += self.__criterion(output, target).item()
                    test_loss += self.__criterion(output, target).item() * data.size(
                        0
                    )  # 累加批次总损失
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.__test_loader.dataset)
            accuracy = 100.0 * correct / len(self.__test_loader.dataset)
            print(
                f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.__test_loader.dataset)}"
                f" ({accuracy:.2f}%)\n"
            )

        return self.__model.state_dict()

    def prediction(self, nums):
        if (nums >= 0) & (nums < 100):
            # 示例预测（使用测试集中的第一个样本）
            sample_data, sample_label = self.test_dataset[nums]
            sample_data = sample_data.unsqueeze(0).to(self.__device)
            self.__model.eval()
            with torch.no_grad():
                prediction = self.__model(sample_data).argmax(dim=1)
            print(
                f"\nExample prediction: True label: {sample_label}, Predicted: {prediction.item()}"
            )
        else:
            raise TypeError(
                f"{nums} is not in compliance with regulations, should be rang 0 to 99."
            )


def cnn_conv2d_pretrained(model_path, mnist_format):
    __device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    __model = CNNConv2d().to(__device)
    __model.load_state_dict(torch.load(model_path, map_location=__device))
    # 切换为评估模式
    __model.eval()

    mnist_format = mnist_format.to(__device)
    # 执行推理
    with torch.no_grad():
        output = __model(mnist_format)
        prob = nn.functional.log_softmax(output, dim=1)
        pred = prob.argmax(dim=1, keepdim=True)

        return pred.item(), prob.squeeze().cpu().numpy()
