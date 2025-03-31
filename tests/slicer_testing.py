from pathlib import Path
import sys

current_path = Path(__file__).resolve()
project_parent = (
    current_path.parent.parent
)  # Adjust the times of .parent according to the actual structure


if project_parent.exists():
    sys.path.insert(0, str(project_parent))
else:
    raise FileNotFoundError(f"destination directory does not exist: {project_parent}")

try:
    from modules import process_image_to_mnist

except ImportError as e:
    print(f"Import failed: {e}")


current_dir = Path.cwd()
target_dir = current_dir / "uploads/nums.jpg"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(f"target_dir is {target_dir}")
    digit_tensors = process_image_to_mnist(target_dir)

    digital_nums = len(digit_tensors)
    print(">>>>>digital_nums:", digital_nums)
    show_column = digital_nums // 10

    if (digital_nums / 10) > show_column:
        show_column += 1

    for tensor_object in digit_tensors:
        print(
            f">>>Single sample data type: {type(tensor_object)}; \
            Tensor shape: {tensor_object.shape}; \
            Pixel value range:, {tensor_object.min().item()}, {tensor_object.max().item()}"
        )

    # visualization
    plt.figure(figsize=(20, (4 * show_column)))
    # Create a graphic window with overall dimensions of x inches wide and y inches high
    for i, tensor in enumerate(digit_tensors[:digital_nums]):
        plt.subplot(show_column, 10, i + 1)
        plt.imshow(tensor.numpy().squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()
