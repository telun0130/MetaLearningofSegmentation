import numpy as np
from PIL import Image
import os


def unpatchify(root, destn):
    List = os.listdir(root)
    patch_num = len(List)//9
    for p_id in range(patch_num):
        # 假设你有九张图像，分别是 image1.jpg, image2.jpg, ..., image9.jpg
        # 将它们加载到一个列表中
        image_paths = List[9*p_id:9*(p_id+1)]
        images = [Image.open(os.path.join(root,image_path)) for image_path in image_paths]

        # 获取每张图像的大小（假设它们都有相同的大小）
        image_size = images[0].size

        # 计算拼接后的图像的大小
        num_rows = 3
        num_cols = 3
        new_width = image_size[0] * num_cols
        new_height = image_size[1] * num_rows

        # 创建一个新的图像对象来存储拼接后的图像
        full_image = Image.new('RGB', (new_width, new_height))

        # 将每张图像依次拼接到完整图像中
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                full_image.paste(images[index], (j * image_size[0], i * image_size[1]))

        # 保存拼接后的完整图像
        save_name = 'image_%s.png' %str(p_id)
        full_image.save(os.path.join(destn, save_name))

if __name__ == '__main__':
    unpatchify(root='Temp/in', destn='Temp/out')
