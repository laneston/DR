from pathlib import Path
import sys
import matplotlib.pyplot as plt


current_path = Path(__file__).resolve()
project_parent = current_path.parent.parent  # 根据实际结构调整.parent次数


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"目标目录不存在: {project_parent}")

try:
    from modules import process_image_to_mnist, cnn_conv2d_pretrained

except ImportError as e:
    print(f"导入失败: {e}")


current_dir = Path.cwd()
target_dir = current_dir / "uploads/numa.jpg"
model_dir = current_dir / "modelspath/mnist_cnn.pth"


if __name__ == "__main__":
    print(f">>>>target_dir is {target_dir}; model_dir is {model_dir}")

    digit_tensors = process_image_to_mnist(target_dir)

    # 计算切割后的图像数目
    digital_nums = len(digit_tensors)
    print(">>>>>digital_nums:", digital_nums)
    show_column = digital_nums // 5

    if (digital_nums / 5) > show_column:
        show_column += 1

    plt.figure(
        figsize=(10, (2 * show_column))
    )  # 创建一个图形窗口，设置整体尺寸为宽x英寸、高y英寸

    for i, tensor_object in enumerate(digit_tensors[:digital_nums]):
        # for tensor_object in digit_tensors[:digital_nums]:
        # 打印输出信息
        print(
            f">>>单个样本数据类型: {type(tensor_object)}; \
            张量形状: {tensor_object.shape}; \
            像素值范围:, {tensor_object.min().item()}, {tensor_object.max().item()}"
        )
        prediction, probabilities = cnn_conv2d_pretrained(model_dir, tensor_object)

        print(f"------>预测结果: {prediction}; 概率分布: {probabilities}")

        plt.subplot(show_column, 5, i + 1)
        plt.imshow(tensor_object.numpy().squeeze(), cmap="gray")
        plt.text(
            0.5,
            -0.15,
            f"estimate: {prediction}",
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=7,
        )  # 底部预测结果[7](@ref)
        plt.axis("off")

    plt.show()
