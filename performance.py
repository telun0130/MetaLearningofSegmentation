import os
from torchvision import transforms
from PIL import Image

img_trans = transforms.Compose(
    [transforms.ToTensor(),
    #  transforms.Grayscale(3),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.ToPILImage()
    ])

img = Image.open("C:/telun/FCN/Dataset_1/valid_set/pic/81_11.png")
img = img_trans(img)
img.show()

