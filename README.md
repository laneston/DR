# 操作方式

## 训练模型

```
python train_testing.py
```

## 测试图像分割功能

```
python train_testing.py
```

## 验证手写数字

```
python handwrite_testing.py
```

# 案例结果

## 输入

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/nums.jpg"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/numa.jpg"></div>

## 训练5轮模型识别效果

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_nums.jpg"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_numa.jpg"></div>

## 训练15轮模型识别效果

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_nums15.jpg"></div>

<div align="center"><img src="https://github.com/laneston/note/blob/main/00-img/Post-tensor/digital_numa15.jpg"></div>

## 结论

- 在数字图像为正常角度的情况下，随着训练次数增加可提高识别的准确性；
- 在数字图像为非正常角度的情况下，随着训练次数增加无法明显提高识别的准确性，需要进行多次形变（矩阵旋转、转置等）后再进行识别，且需要对此图像是否为数字进行判断，以保证识别的有效性；
