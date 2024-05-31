import torch
from torchvision import transforms
from PIL import Image
import os

def calculate_mean_std(image_folder):
    # 定义转换器将图像转换为张量
    transform = transforms.Compose([
        transforms.ToTensor()            # 将图像转换为张量
    ])
    # 初始化存储像素值的列表
    pixel_values = []

    file = os.listdir(image_folder)
    for name in file:
        path = os.path.join(image_folder, name)
        images = Image.open(path)
        images = transform(images)
        pixel_values.append(images.view(images.size(0), -1))

    # 将像素值列表转换为张量
    pixel_values = torch.cat(pixel_values, dim=1)

    # 计算所有图像的均值和标准差
    mean = torch.mean(pixel_values, dim=1)
    std = torch.std(pixel_values, dim=1)

    return mean, std

# 指定包含图像文件的文件夹路径
image_folder_path = "C:/telun/FCN/Dataset_1/train_set/pic"

# 计算均值和标准差
mean, std = calculate_mean_std(image_folder_path)

# 打印结果
print("Mean:", mean)
print("Standard Deviation:", std)
