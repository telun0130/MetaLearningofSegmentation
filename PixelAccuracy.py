import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import transforms
import torchvision.datasets
from PIL import Image
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from MLFunc.Model import FCN

# mask_trans  = transforms.Compose(
#     [transforms.ToTensor()]
# )
# img_trans = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataPath = 'C:/telun/FCN/7f/fold_7/test_set'

class CM_eval():
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    def __init__(self, cm, mask, predict) -> None:
        self.mask = mask
        self.pred = predict
        if (cm[0][0]):
            self.TN = cm[0][0]
        if (cm[0][1]):
            self.FP = cm[0][1]
        if (cm[1][0]):
            self.FN = cm[1][0]
        if (cm[1][1]):
            self.TP = cm[1][1]
    def Accuracy(self):
        acc = accuracy_score(self.mask, self.pred)
        return acc
    def IoU(self):
        intersection = numpy.logical_and(self.mask, self.pred)
        if numpy.sum(intersection) == 0:
            return 0.0
        iou = self.TP / (self.TP + self.FP + self.FN)
        return iou
    def ReCall(self):
        recall = recall_score(self.mask, self.pred, zero_division=0.0)
        return recall
    def Precision(self):
        precis = precision_score(self.mask, self.pred, zero_division=0.0)
        return precis
    def F1(self):
        f1 = f1_score(self.mask, self.pred, zero_division=0.0)
        return f1

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
        return img, mask
    def __len__(self):
        return len(self.imgs)
    
# validset = dataset(dataPath, transform=img_trans, mask_transform=mask_trans)
# evalloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

# model = FCN(2)

# Path = ['C:/telun/FCN/7fold_exp1/fd7/model/FCN_400.pth',
#         'C:/telun/FCN/7fold_exp1/fd7/model/Reptile_FCN_400.pth',
#         'C:/telun/FCN/7fold_exp1/fd7/model/FCN_600.pth',
#         'C:/telun/FCN/7fold_exp1/fd7/model/Reptile_FCN_600.pth']

# for modelPath in Path:
#     model.load_state_dict(torch.load(modelPath)['model'])

#     model.cuda()
#     model.eval()

#     Acc_total = 0
#     IoU_total = 0
#     Precis_total = 0
#     Recall_total = 0
#     F1_total = 0
#     for i, data in enumerate(evalloader, 0):
#         img, mask = data[0].cuda(), data[1].cuda()
#         predict = model(img)

#         predict = predict.cpu()

#         sigmoid = nn.Sigmoid()
#         predict = sigmoid(predict)
#         threshold = torch.tensor([0.5])

#         mask = mask[0][0].cpu().detach().numpy()

#         # 實測後，[0][1] 最能表達理想結果
#         predict = (predict[0][1] > threshold).float() * 1
#         predict = predict.cpu().detach().numpy()

#         metric = confusion_matrix(mask.flatten(), predict.flatten(), labels=[0., 1.])

#         Metric = CM_eval(metric, mask=mask.flatten(), predict=predict.flatten())
#         Acc_total += Metric.Accuracy()
#         IoU_total += Metric.IoU()
#         Precis_total += Metric.Precision()
#         Recall_total += Metric.ReCall()
#         F1_total += Metric.F1()

#     avg_acc = Acc_total / validset.__len__()
#     avg_iou = IoU_total / validset.__len__()
#     avg_pre = Precis_total / validset.__len__()
#     avg_rec = Recall_total / validset.__len__()
#     avg_f1 = F1_total / validset.__len__()

#     print("Model: %s" % modelPath)
#     print("Accuracy = %s | IoU = %s | Precision = %s | ReCall = %s | F1-Score = %s" % (str(avg_acc), 
#                                                                                     str(avg_iou),
#                                                                                     str(avg_pre),
#                                                                                     str(avg_rec),
#                                                                                     str(avg_f1)))