import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from .model import DigitRecognizer
from .slicer import process_image_to_mnist


class HandwriteParser:
    def __init__(self, model_path):
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = DigitRecognizer().to(self.__device)
        self.__model.load_state_dict(torch.load(model_path, map_location = self.__device))
        # 切换为评估模式
        self.__model.eval()

    def handwrite_pretrained(self, mnist_format):
        mnist_format = mnist_format.to(self.__device)
        # 执行推理
        with torch.no_grad():
            output = self.__model(mnist_format)
            prob = nn.functional.log_softmax(output, dim=1)
            pred = prob.argmax(dim=1, keepdim=True)
        
        return pred.item(), prob.squeeze().cpu().numpy()
