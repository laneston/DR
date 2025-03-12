import cv2
import numpy as np
import torch
from torchvision import transforms


# def process_image_to_mnist(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or invalid image path")
    
#     # 图像预处理流程
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY_INV, 11, 2)
    
#     # 形态学操作优化
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
#     # 轮廓检测与处理
#     contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     valid_contours = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = w / float(h)
#         area = cv2.contourArea(cnt)
        
#         # 高级过滤条件（可根据实际调整）
#         if area > 80 and 0.25 < aspect_ratio < 4:
#             valid_contours.append(cnt)
    
#     # 按从左到右排序数字
#     valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
    
#     digits = []
#     for cnt in valid_contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         roi = processed[y:y+h, x:x+w]
        
#         # 专业MNIST格式化处理
#         # 步骤1：创建正方形画布并保持宽高比
#         (h, w) = roi.shape
#         padding = 4
#         if w > h:
#             pad_total = w - h
#             pad_top = pad_total // 2
#             pad_bottom = pad_total - pad_top
#             roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
#         else:
#             pad_total = h - w
#             pad_left = pad_total // 2
#             pad_right = pad_total - pad_left
#             roi = cv2.copyMakeBorder(roi, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
#         # 步骤2：添加额外padding并缩放
#         roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
#         roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
        
#         # 步骤3：居中放置到28x28画布
#         mnist_format = np.zeros((28, 28), dtype=np.uint8)
#         mnist_format[4:24, 4:24] = roi  # 居中放置
        
#         digits.append(mnist_format)
    
#     # PyTorch标准化处理
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),  # MNIST官方参数
#         # 添加批次维度：将 [C, H, W] -> [B, C, H, W]
#         transforms.Lambda(lambda x: x.unsqueeze(0))
#     ])
    
#     return [transform(digit) for digit in digits]



def process_image_to_mnist(image_path):
    # 图像读取与预处理
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像读取失败，请检查路径有效性")
    
    # 灰度化与降噪处理 [3,5](@ref)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 自适应阈值二值化（MNIST风格：黑底白字）[5,7](@ref)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 形态学优化（连接断裂笔画）[2,5](@ref)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 多数字轮廓检测 [3,5](@ref)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        # 轮廓过滤条件（排除噪点）[5,7](@ref)
        if area > 80 and 0.25 < aspect_ratio < 4:
            valid_contours.append((x, y, w, h))
    
    # 按水平位置排序（保持数字顺序）[5](@ref)
    valid_contours = sorted(valid_contours, key=lambda c: c[0])
    
    digits = []
    for (x, y, w, h) in valid_contours:
        # 提取单个数字ROI
        roi = processed[y:y+h, x:x+w]
        
        # MNIST标准化处理流程 [5,6](@ref)
        # 步骤1：创建正方形画布
        (h, w) = roi.shape
        padding = 4
        if w > h:
            pad_total = w - h
            roi = cv2.copyMakeBorder(roi, pad_total//2, pad_total - pad_total//2, 
                                   0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_total = h - w
            roi = cv2.copyMakeBorder(roi, 0, 0, 
                                   pad_total//2, pad_total - pad_total//2, 
                                   cv2.BORDER_CONSTANT, value=0)
        
        # 步骤2：缩放并居中放置
        roi = cv2.resize(roi, (20,20), interpolation=cv2.INTER_AREA)
        mnist_canvas = np.zeros((28,28), dtype=np.uint8)
        mnist_canvas[4:24, 4:24] = roi  # 四周各保留4像素边距
        
        # PyTorch标准化处理 [6,8](@ref)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST官方参数
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])
        digit_tensor = transform(mnist_canvas)
        
        digits.append(digit_tensor)
    
    return digits


