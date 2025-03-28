from pathlib import Path
import sys

current_path = Path(__file__).resolve()
project_parent = current_path.parent.parent  # 根据实际结构调整.parent次数


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"目标目录不存在: {project_parent}")

try:
    from modules import process_image_to_mnist

except ImportError as e:
    print(f"导入失败: {e}")


current_dir = Path.cwd()
target_dir = current_dir / "uploads/nums.jpg"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(f"target_dir is {target_dir}")
    # 处理图像
    digit_tensors = process_image_to_mnist(target_dir)

    # 计算切割后的图像数目
    digital_nums = len(digit_tensors)
    print(">>>>>digital_nums:", digital_nums)
    show_column = digital_nums // 10

    if (digital_nums / 10) > show_column:
        show_column += 1

    for tensor_object in digit_tensors:
        # 打印输出信息
        print(
            f">>>单个样本数据类型: {type(tensor_object)}; \
            张量形状: {tensor_object.shape}; \
            像素值范围:, {tensor_object.min().item()}, {tensor_object.max().item()}"
        )

    # 可视化显示
    plt.figure(
        figsize=(20, (4 * show_column))
    )  # 创建一个图形窗口，设置整体尺寸为宽x英寸、高y英寸
    for i, tensor in enumerate(digit_tensors[:digital_nums]):
        plt.subplot(show_column, 10, i + 1)
        plt.imshow(tensor.numpy().squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()
