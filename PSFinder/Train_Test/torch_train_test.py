from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
import os, random, shutil
from model_vgg import VGGmodel
from collections.abc import Iterable
from typing import Union, List, Dict, Any, cast
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

def GPU_usage_and_test():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    else:
        print("wrong")

# data_dir = '../input/kermany2018/oct2017/OCT2017 '
# TRAIN = 'train'
# VAL = 'val'
# TEST = 'test'


# 制作训练集和测试集
def moveFile(fileDir,tarDir):
        pathDir = os.listdir(fileDir)    # obtain the image path
        filenumber=len(pathDir)
        rate=0.1    #define the ratio of image selection
        picknumber=int(filenumber*rate) # extract images following the ratio
        sample = random.sample(pathDir, picknumber)  #randomly select image samples 
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)


def mv_frames_to_dir(valid_path = "data_copy/valid_frame_update_data",invalid_path = "data_copy/invalid_frame_data",validdir = "frame_data/valid_pytorch",invaliddir = "frame_data/invalid_pytorch"):
    count = 0
    for video in os.listdir(valid_path):
        for img in os.listdir(os.path.join(valid_path,video)):
            # print(img)
            # print(os.path.join(valid_path,video,img))
            if not os.path.exists("frame_data/pytorch_data/"+img):
                copyfile(os.path.join(valid_path,video,img),"frame_data/pytorch_data/"+"valid"+img)
            count+=1
    print("the number of valid frames are "+str(count))
    count = 0
    for video in os.listdir(invalid_path):
        for img in os.listdir(os.path.join(invalid_path,video)):
            if not os.path.exists("frame_data/pytorch_data/"+img):
                copyfile(os.path.join(invalid_path,video,img),"frame_data/pytorch_data/"+"other"+img)
            count+=1
    print("the number of invalid frames are "+str(count))


data_transform  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])


class myDataSet(Dataset):
    def __init__(self, root, transform):
        self.image_files = np.array([x.path for x in os.scandir(root)])
        self.transform = transform
    def __getitem__(self, index):
        
        x = Image.open(self.image_files[index])
        x = self.transform(x)
        # label valid as 1, invalid as 0
        thisLabel =0
        if "valid" in self.image_files[index]:
            thisLabel=1
        return x,thisLabel
 
        if self.transform is not None:
            img = self.transform(img) 
        print("label is "+str(label))
        return img, label

    def __len__(self):
        return len(self.image_files)


# 冻住feature层，只训练FC；另一个解冻feature训练所有参数
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define some parameters
    EPOCH = 30

    GPU_usage_and_test()
    # mv_frames_to_dir()
    # make test set
    # moveFile("frame_data/pytorch_data/","frame_data/test/")
    train_data = myDataSet("frame_data/pytorch_data",data_transform)
    test_data = myDataSet("frame_data/test",data_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=0)

    # you can also try model with VGG
    # print(VGGmodel())
    # model = VGGmodel().to(device)

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # optimizer for psc2code is rmsprop 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    y_predict = []
    y_true = []

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # 训练
    print("start training")

    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0     
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            # inputs, labels = data
            inputs = feature_extractor(images=data, return_tensors="pt")

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每个batch 打印一次loss和准确率
            sum_loss += loss.item()
        # 使用Top5分类
            maxk = max((1,2))
            label_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            total += labels.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
       
    # 

    torch.save(model.state_dict(), 'VGG_model.pth')


