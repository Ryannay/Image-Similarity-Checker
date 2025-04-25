from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm


cifar_data = CIFAR10(root='./data', train=True, download=True)


def save_cifar_images(dataset, save_dir="./cifar_images"):
    classes = dataset.classes  # ['airplane', 'automobile', ...]
    data, labels = dataset.data, dataset.targets

    for i, (img_array, label) in enumerate(tqdm(zip(data, labels), total=len(data))):
        class_name = classes[label]
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img = Image.fromarray(img_array)
        img.save(os.path.join(class_dir, f"{class_name}_{i:05d}.png"))

save_cifar_images(cifar_data)