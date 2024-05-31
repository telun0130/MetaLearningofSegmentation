import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from MLFunc.dataset import dataset
from MLFunc.dataset import reductionset
from PixelAccuracy import CM_eval, confusion_matrix
from MLFunc.Model import FCN
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 1
mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
Dataset = dataset('C:/telun/FCN/Dataset_1/train_set', transform=img_trans,
                   mask_transform=mask_trans)
trainloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0)

validset = dataset('C:/telun/FCN/Dataset_1/valid_set', transform=img_trans,
                   mask_transform=mask_trans)
validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=True, num_workers=0)

testset = dataset('C:/telun/FCN/Dataset_1/test_set', transform=img_trans,
                   mask_transform=mask_trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

model_path = 'C:/telun/FCN/modelConverge/model/meta/Reptile_init_v0.pth'
model = FCN(2)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.85)

print(f"trainset: {Dataset.__len__()} samples")
print(f"validset: {validset.__len__()} samples")
print(f"testset: {testset.__len__()} samples")
model.train()
trainfields = ['Epoch', 'TrainLoss'] 
trainrecord = []

validfields = ['Epoch', 'validIoU'] 
validrecord = []

testfields = ['Epoch', 'testIoU'] 
testrecord = []

target_epoch = 700
gap = 50
start = time.time()

try:
    Epoch = 0
    IoU_valid = 0
    IoU_test = 0
    for train_epoch in range(target_epoch):
        Epoch += 1
        error = 0.
        IoU_valid = 0
        IoU_test = 0
        for i, data in enumerate(trainloader, 0):
            img, mask = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask[0].long())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            error = error + loss.cpu()
        error = error / Dataset.__len__()
        item = []
        item.append(str(Epoch))
        item.append(str(error.item()))
        trainrecord.append(item)
        print('Epoch: %s | avg_Err: %s' %(str(Epoch), str(error.item())))
        torch.cuda.empty_cache()

        if Epoch % gap == 0:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(validloader, 0):
                    img, mask = data[0].cuda(), data[1].cuda()
                    predict = model(img)
                    predict = predict.cpu()
                    sigmoid = nn.Sigmoid()
                    predict = sigmoid(predict)
                    threshold = torch.tensor([0.5])
                    mask = mask[0][0].cpu().detach().numpy()

                    # 實測後，[0][1] 最能表達理想結果
                    predict = (predict[0][1] > threshold).float() * 1
                    predict = predict.cpu().detach().numpy()

                    metric = confusion_matrix(mask.flatten(), predict.flatten(), labels=[0., 1.])

                    Metric = CM_eval(metric, mask=mask.flatten(), predict=predict.flatten())
                    IoU_valid += Metric.IoU()
                valid_avg_iou = IoU_valid / validset.__len__()
                item = []
                item.append(str(Epoch))
                item.append(str(valid_avg_iou))
                validrecord.append(item)

                for i, data in enumerate(testloader, 0):
                    img, mask = data[0].cuda(), data[1].cuda()
                    predict = model(img)
                    predict = predict.cpu()
                    sigmoid = nn.Sigmoid()
                    predict = sigmoid(predict)
                    threshold = torch.tensor([0.5])
                    mask = mask[0][0].cpu().detach().numpy()

                    # 實測後，[0][1] 最能表達理想結果
                    predict = (predict[0][1] > threshold).float() * 1
                    predict = predict.cpu().detach().numpy()

                    metric = confusion_matrix(mask.flatten(), predict.flatten(), labels=[0., 1.])

                    Metric = CM_eval(metric, mask=mask.flatten(), predict=predict.flatten())
                    IoU_test += Metric.IoU()
                test_avg_iou = IoU_test / testset.__len__()
                item = []
                item.append(str(Epoch))
                item.append(str(test_avg_iou))
                testrecord.append(item)

                Model = {'model': model.state_dict()}
                torch.save(Model, 'C:/telun/FCN/modelConverge/model/ReptileFCN/ReptileFCN_%s.pth' % str(Epoch))      
        model.train()
        scheduler.step()
    end = time.time()
    Time = end - start
    print("時長：%s" %str(Time))
except Exception as err:
    print(err)

with open('C:/telun/FCN/modelConverge/ReptileFCNtrain', 'w') as f:
    write = csv.writer(f)  
    write.writerow(trainfields)
    write.writerows(trainrecord)
with open('C:/telun/FCN/modelConverge/ReptileFCNvalid', 'w') as f:
    write = csv.writer(f)  
    write.writerow(validfields)
    write.writerows(validrecord)
with open('C:/telun/FCN/modelConverge/ReptileFCNtest', 'w') as f:
    write = csv.writer(f)  
    write.writerow(testfields)
    write.writerows(testrecord)