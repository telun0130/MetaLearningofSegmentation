import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from MLFunc.dataset import dataset
from MLFunc.dataset import reductionset
import MLFunc.Extra as ex_func
from MLFunc.Model import FCN
import itertools
from torchmetrics.classification import BinaryJaccardIndex
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

target_epoch = 300
batch_size = 1

mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(3)])

Dataset = dataset('C:/telun/FCN/7f/fold_1/train_set', transform=img_trans,
                   mask_transform=mask_trans)

trainloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0)

testset = dataset('C:/telun/FCN/7f/fold_1/test_set', transform=img_trans,
                   mask_transform=mask_trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

# model_path = "C:/telun/FCN/Model/Meta/MAMLFCNv4.pth"
model = FCN(2)
# model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)

breakpoint = {
    "param": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 0
}
best_error = 1.
best_epoch = 0

model.train()

fields = ['Epoch', 'avg_error', 'Lr', 'validIoU', 'testIoU'] 
record = []

start = time.time()

for epoch in range(target_epoch):
    # error = 0.
    for i, data in enumerate(trainloader, 0):
        img, mask = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask[0].long())
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        # error = error + loss.cpu()
    scheduler.step()
    # error = error / Trainset.__len__()
    # if(error < best_error):
    #     best_error = error
    #     best_epoch = epoch
    #     breakpoint['model'] = model.state_dict()
    #     breakpoint["optimizer"] = optimizer.state_dict()
    #     breakpoint["epoch"] = best_epoch
    # lr = ex_func.get_lr(optimizer=optimizer)
    # item = []
    # item.append(str(epoch))
    # item.append(str(error.item()))
    # item.append(str(lr))

    torch.cuda.empty_cache()

    # model.eval()
    # with torch.no_grad():
    #     valid_IoU = valid(Validset=Validset, n_sample=100, model=model)
    #     item.append(str(valid_IoU))
    #     mIoU = IoU(epoch=10, sample_num=500, dataloader=testloader, model=model)
    #     item.append(str(mIoU))
    #     record.append(item)
    #     print('Epoch: %s | avg_Err: %s | Lr: %s | validIoU: %s | testIoU: %s' %(str(epoch), str(error.item()), str(lr), str(valid_IoU), str(mIoU)))
    #     torch.cuda.empty_cache()
    # model.train()

Model = {'param': model.state_dict()}
torch.save(Model, 'FCN675_20240312.pth')

end = time.time()
print("執行時間：%f 秒" % (end - start))

with open('FCN675_20240312', 'w') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(record)