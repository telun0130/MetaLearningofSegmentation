import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from MLFunc.dataset import dataset
from MLFunc.Model import FCN
import csv

print(torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 1
mask_trans = transforms.Compose(
    [transforms.ToTensor()]
)
img_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
Dataset = dataset('C:/telun/FCN/new7fold/7f6/train_set', transform=img_trans,
                   mask_transform=mask_trans)
trainloader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=0)

model_path = 'C:/telun/FCN/Evaluation_exp2/Reptile_FCN/model/Reptile_init_v1.pth'
model = FCN(2)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.85)
breakpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 0
}

model.train()

fields = ['Epoch', 'TrainLoss'] 
record = []

target_epoch = 400
start = time.time()

try:
    Epoch = 0
    for train_epoch in range(target_epoch):
        Epoch += 1
        error = 0.
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
        record.append(item)
        print('Epoch: %s | avg_Err: %s' %(str(Epoch), str(error.item())))
        torch.cuda.empty_cache()

        model.eval()
        # save model in 400 600 epoch
        if Epoch == 300 or Epoch == 350 or Epoch == 400:
            print("Save the Model")
            Model = {'model': model.state_dict()}
            torch.save(Model, 'C:/telun/FCN/7fold_exp1/fd6/model/Reptile_FCN_%s.pth' % str(Epoch))
            # torch.save(Model, 'C:/telun/FCN/7fold_exp1/fd6/model/FCN_%s.pth' % str(Epoch))
        else:
            pass
        model.train()
        scheduler.step()
except Exception as err:
    print(err)
    Model = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(Model, 'C:/telun/FCN/7fold_exp1/fd6/model/breakpoint_Reptile_FCN_%s.pth' % str(Epoch))
    # torch.save(Model, 'C:/telun/FCN/7fold_exp1/fd6/model/breakpoint_FCN_%s.pth' % str(Epoch))

    with open('C:/telun/FCN/7fold_exp1/fd6/record/Reptile_FCN_record', 'w') as f:
    # with open('C:/telun/FCN/7fold_exp1/fd6/record/FCN_record', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(record)

end = time.time()
Time = end - start
print("執行時間：%f 秒" % (Time))

with open('C:/telun/FCN/7fold_exp1/fd6/record/Reptile_FCN_record', 'w') as f:
# with open('C:/telun/FCN/7fold_exp1/fd6/record/FCN_record', 'w') as f:
    write = csv.writer(f)  
    write.writerow(fields)
    write.writerows(record)