import torch
import os
import torch
import torch.nn as nn
import numpy
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from MLFunc.Model import FCN

mask_trans  = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
modelPath = 'C:/telun/FCN/Evaluation_exp2/Reptile_FCN/model/Reptile_FCN_600.pth'
dataPath = 'C:/telun/FCN/Test_fold1'
savePath = 'C:/telun/FCN/Evaluation_exp2/Reptile_FCN/result/Reptile_FCN_600'

class dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mask_transform = None):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "pic"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.transform = transform
        self.mask_transform = mask_transform
    def __getitem__(self, idx):
        data_name = self.imgs[idx]
        img_path = os.path.join(self.root, "pic", data_name)
        mask_path = os.path.join(self.root, "mask", data_name)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img= self.transform(img)
        if self.mask_transform is not None:
            mask= self.mask_transform(mask)
        return img, mask, data_name
    def __len__(self):
        return len(self.imgs)
    
validset = dataset(dataPath, transform=img_trans, mask_transform=mask_trans)
evalloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

model = FCN(2)

model.load_state_dict(torch.load(modelPath)['model'])

model.cuda()
model.eval()

for i, data in enumerate(evalloader, 0):
  filename = data[2]
  img, mask = data[0].cuda(), data[1].cuda()
  predict = model(img)

  predict = predict.cpu()

  sigmoid = nn.Sigmoid()
  predict = sigmoid(predict)
  threshold = torch.tensor([0.5])

  # 實測後，[0][1] 最能表達理想結果
  predict = (predict[0][1] > threshold).float() * 255
  predict = predict.cpu().detach().numpy()
  predict = predict.astype(numpy.uint8)
  img = Image.fromarray(predict)
  img.save(os.path.join(savePath, filename[0]))
  