import json
import subprocess
import sys
import os
import time
from PIL import Image
from numpy import average, dot, linalg
import cv2
import skimage.measure
import os
import subprocess
import torch
from torchvision import datasets, models, transforms

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
from collections.abc import Iterable
from earlystop import EarlyStopping
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests


def callsubprocess(video_path,frame_path):

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", videopath,"-f","image2", "-r","0.02","-vf", "fps=fps=1", "-q:v", "2",frame_path+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)

data_transform  = transforms.Compose([
    transforms.Resize((300,300)),

    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

])


class myDataSet(Dataset):
    def __init__(self, root, transform):
        self.image_files = np.array([x.path for x in os.scandir(root)])
        self.transform = transform
    def __getitem__(self, index):
        
        x = Image.open(self.image_files[index])
        x = self.transform(x)
        # label valid as 1, invalid as 0
        thisLabel =1
        if "invalid" in self.image_files[index]:
            thisLabel=0
        return x,thisLabel
 
        if self.transform is not None:
            img = self.transform(img) 
        print("label is "+str(label))
        return img, label

    def __len__(self):
        return len(self.image_files)


img_path = path/data
def prediect(imgdir):
    net = torch.load("model/freeze_20earlystop.pth")
    torch.no_grad()

    for item in os.listdir(imgdir):
        img = os.path.join(imgdir,item)
        # print(imgdir)
        # print(item)
        img = Image.open(img)
        img_1 = data_transform(img).unsqueeze(0)
        img_1 = Variable(img_1)
        import torch.nn.functional as F
        # predict = F.softmax(net(img_1))
        print(predict)


def predict_gpu_experiment1():
    test_data = myDataSet("/data/test",data_transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0)
    net = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    net.load_state_dict(torch.load("experiment1_update.pth"))
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    net = net.cuda()
    with torch.no_grad(): 
        net.eval()
        correct = torch.zeros(1).squeeze().cuda()
        TOTAL = torch.zeros(1).squeeze().cuda()
        Yes = torch.zeros(1).squeeze().cuda()
        TP = torch.zeros(1).squeeze().cuda()
        TN = torch.zeros(1).squeeze().cuda()
        FN = torch.zeros(1).squeeze().cuda()
        FP = torch.zeros(1).squeeze().cuda()

        for data in testloader:
            target, labels = data
            target = feature_extractor(target)
            target, labels = target.cuda(), labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(target)
            # calculate the loss
            # record validation loss
            _, prediction = torch.max(output, 1)
            # prediction = torch.argmax(output, 1)
            # correct += (prediction == labels).sum().float()
            TP+=((prediction == 1) & (labels == 1)).sum()
            TN+=((prediction == 0) & (labels == 0)).sum()
            FN+=((prediction == 0) & (labels == 1)).sum()
            FP+=((prediction == 1) & (labels == 0)).sum()

            Yes+= (labels == 1).sum()
            TOTAL += len(labels)


        # int(correct.item())
        total = int(TOTAL.item())
        tp = int(TP.item())
        tn = int(TN.item())
        fn = int(FN.item())
        fp = int(FP.item())
        print(total)
        print(Yes)
        
        
        print_acc = ("test_ACC: %.4f "%((tp+tn)/total))
        precison = ("test_precision: %.4f "%(tp/(tp+fp)))
        recall = ("test_rercall: %.4f "%(tp/(tp+fn)))
        f1 = ("test_f1: %.4f "%(2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn)))))

    print(print_acc+"|"+precison+"|"+recall+"|"+f1+"|")

# def predict_gpu_vgg(imgdir):
    test_data = myDataSet("experiment2_data/test",data_transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0)
    # net= torch.load("experiment2.pth")
    from torchvision_vgg import VGG,vgg16_bn1
    net = vgg16_bn1()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load("experiment2_1.pth"))
    net = net.cuda()
    with torch.no_grad(): 
        net.eval()
        correct = torch.zeros(1).squeeze().cuda()
        TOTAL = torch.zeros(1).squeeze().cuda()
        TP = torch.zeros(1).squeeze().cuda()
        TN = torch.zeros(1).squeeze().cuda()
        FN = torch.zeros(1).squeeze().cuda()
        FP = torch.zeros(1).squeeze().cuda()

        for data in testloader:
            target, labels = data
            target, labels = target.cuda(), labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(target)
            # calculate the loss
            # record validation loss

            prediction = torch.argmax(output, 1)
            # correct += (prediction == labels).sum().float()
            TP+=((prediction == 1) & (labels == 1)).sum()
            TN+=((prediction == 0) & (labels == 0)).sum()
            FN+=((prediction == 0) & (labels == 1)).sum()
            FP+=((prediction == 1) & (labels == 0)).sum()


            TOTAL += len(labels)


        # int(correct.item())
        total = int(TOTAL.item())
        tp = int(TP.item())
        tn = int(TN.item())
        fn = int(FN.item())
        fp = int(FP.item())
        
        
        print_acc = ("test_ACC: %.4f "%((tp+tn)/total))
        precison = ("test_precision: %.4f "%(tp/(tp+fp)))
        recall = ("test_rercall: %.4f "%(tp/(tp+fn)))
        f1 = ("test_f1: %.4f "%(2*(tp/(tp+fp))*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn)))))

    print(print_acc+"|"+precison+"|"+recall+"|"+f1+"|")
     
# predict_gpu_experiment2(img_path)
predict_gpu_experiment1()
# prediect(img_path)

