import os
import shutil

imgSource = "newdataset/fold6/pic"
maskSource = "newdataset/fold6/mask"

imgdest = "C:/telun/FCN/new7fold/7f7/train_set/pic"
maskdest = "C:/telun/FCN/new7fold/7f7/train_set/mask"

if __name__=="__main__":
    file_name = [f for f in os.listdir(imgSource) if f.endswith(('.png'))]
    for name in file_name:
        pic = os.path.join(imgSource,name)
        mask = os.path.join(maskSource,name)
        dest_pic = os.path.join(imgdest, name)
        dest_mask = os.path.join(maskdest, name)
        shutil.copy(pic, dest_pic)
        shutil.copy(mask, dest_mask)
