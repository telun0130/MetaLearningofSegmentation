import os
import time
import random
import torch
import torch.nn as nn
from MLFunc.Model import FCN
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# hyper parameter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
inner_lr = 0.01
lr = 0.0085
meta_epoch = 50

start = "C:/telun/FCN/Original_metaDataset"
dir = []
for dirlv1 in (os.listdir(start)):
    node1 = os.path.join(start, dirlv1)
    for dirlv2 in (os.listdir(node1)):
        node2 = os.path.join(node1, dirlv2)
        dir.append(node2)
# order = [8, 0, 1, 2, 4, 5, 6, 3, 7] #learn low scale image first, then learn fragment and vessel in different color, finally learn the dark color feature
indexPool = [0, 1, 2, 3, 4, 5, 6]

img_transform =  transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mask_transform = transforms.Compose([transforms.ToTensor()])

class dataset(Dataset):
    def __init__(self, root, img_trans = None, mask_trans = None):
        self.dir_root = root
        self.img_path = os.path.join(self.dir_root, 'pic')
        self.mask_path = os.path.join(self.dir_root, 'mask')
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        self.img_trans = img_trans
        self.mask_trans = mask_trans
    def __getitem__(self, index):
        img_name = self.img_list[index]
        file_path = os.path.join(self.img_path, img_name)
        img = Image.open(file_path).convert('RGB')
        if self.img_trans is not None:
            img = self.img_trans(img)
        img = torch.unsqueeze(img, dim=0) # neural network`s input is not a image, is a image in the "batch"
        mask_name = self.mask_list[index]
        maskfile_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(maskfile_path).convert('L')
        if self.mask_trans is not None:
            mask = self.mask_trans(mask)
        return img, mask
    def __len__(self):
        return len(self.img_list)

model = FCN(num_classes=2)
model.cuda()
model_copy = FCN(num_classes=2)
model_copy.load_state_dict(model.state_dict())
model_copy.cuda()

criterion = nn.CrossEntropyLoss()
meta_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

model_copy.train()
model.train()
torch.autograd.set_detect_anomaly(True)

start = time.time()


try:
    for i in range(meta_epoch):
        grad = [torch.zeros_like(param) for param in model.parameters()]
        Init_Param = model.state_dict()

        print("Meta Epoch: %d \n" %i)
        tag_epoch = "Epoch_%d" %i
        meta_optimizer.zero_grad()

        site_dir = random.sample(indexPool, 6) # Task_i～P(T)
        meta_optimizer.zero_grad()
        for index in site_dir:
            # every tasks
            model_copy.load_state_dict(Init_Param)
            taskname = dir[index]
            d = dataset(os.path.join(taskname, 'train'), img_trans=img_transform, mask_trans=mask_transform)
            d_test = dataset(os.path.join(taskname, 'test'), img_trans=img_transform, mask_trans=mask_transform)
            # seq_list = random.sample(range(0, d.__len__()), 6)
            seq_list = random.choices(range(0, d.__len__()), k=6)

            # inner Train and test each task
            for idx in seq_list:
                meta_optimizer.zero_grad()
                img, mask = d[idx]
                output = model_copy(img.cuda())
                loss = criterion(output, mask.long().cuda())
                loss.backward()
                meta_optimizer.step()
                torch.cuda.empty_cache()
                
            idx_test = random.randint(0, d_test.__len__() - 1)
            img_test, mask_test = d_test[idx_test]
            output_test = model_copy(img_test.cuda())
            task_loss = criterion(output_test, mask_test.long().cuda())
            task_grad = torch.autograd.grad(task_loss, model_copy.parameters())
            grad = [grad_param + task_param for grad_param, task_param in zip(grad, task_grad)]
            torch.cuda.empty_cache()

        for main_param, gradient in zip(model.parameters(), grad):
            main_param.grad = gradient / 6
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.empty_cache()

    torch.save({
                'model_state_dict':model.state_dict(),
                }, 'C:/telun/FCN/Evaluation_exp2/MAML_FCN/model/MAML_init_v1.pth')
    # v0 --> 
    # {inner_lr = 0.01,
    #  lr = 0.0085,
    #  meta_epoch = 50,
    #  criterion = nn.CrossEntropyLoss(),
    #  meta_optimizer = optim.SGD(model_copy.parameters(), lr=inner_lr, momentum=0.9),
    #  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9),
    #  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)}

    # v1 --> setting unchanged，change data sampling to choice(can be repeated)

    end = time.time()
    Time = end - start
    print("執行時間：%f 秒" % (Time))

    txtpath = 'C:/telun/FCN/Evaluation_exp2/MAML_FCN/record/MetaTime.txt'
    f = open(txtpath, 'w')
    f.write(str(Time))
    f.close()
except Exception as E:
    print(E)