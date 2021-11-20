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
from model_vgg import VGGNet
from collections.abc import Iterable
from earlystop import EarlyStopping
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests



def GPU_usage_and_test():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    else:
        print("wrong")


def moveFile(fileDir,tarDir):
        pathDir = os.listdir(fileDir)    
        filenumber=len(pathDir)
        rate=0.1    
        picknumber=int(filenumber*rate) 
        sample = random.sample(pathDir, picknumber)  
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

def train_model1(net,feature_extractor,criterion, optimizer,num_epochs):

    avg_valid_losses = [] 
    avg_train_losses = []
    valid_losses = []
    train_losses = []
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(num_epochs+1):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0     
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs = feature_extractor(inputs)
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



        # validate the data
        with torch.no_grad(): 
            net.eval()
            correct = torch.zeros(1).squeeze().cuda()
            total = torch.zeros(1).squeeze().cuda()
            for data in testloader:
                target, labels = data
                target = feature_extractor(target)
                target, labels = target.cuda(), labels.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(target)
                # calculate the loss
                loss = criterion(output, labels)
                # record validation loss
                valid_losses.append(loss.item())

                prediction = torch.argmax(output, 1)
                correct += (prediction == labels).sum().float()
                total += len(labels)
            

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            valid_ACC = (correct/total).cpu().detach().data.numpy()
       
            epoch_len = len(str(num_epochs+1))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}'  +
                        f'valid_ACC: {(correct/total).cpu().detach().data.numpy():.5f}')
        




        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(1/valid_ACC, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    # torch.save(net, 'freeze.pth')


    net.load_state_dict(torch.load('checkpoint.pt'))
    return  net, avg_train_losses, avg_valid_losses



if __name__ == "__main__":
    
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define some parameters

    batch_size = 16

    GPU_usage_and_test()
    # mv_frames_to_dir()
    # make test set
    # moveFile("frame_data/pytorch_data/","frame_data/test/")

    train_data = myDataSet("frame_data/pytorch_data",data_transform)
    test_data = myDataSet("frame_data/test",data_transform)
s
    # train_data = myDataSet("frame_framelevel_data/train",data_transform)
    # test_data = myDataSet("frame_framelevel_data/test",data_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    print("test")
    print(len(trainloader.dataset))
    print(len(testloader.dataset))


    # you can also try to use vgg or resnet models 
    # from torchvision_vgg import VGG,vgg16_bn1
    # vgg16 = vgg16_bn1()
    # # Freeze training for all layers
    # for name, param in vgg16.features.named_parameters():
    #     param.require_grad = False  
    # # vgg16 = nn.DataParallel(vgg16)
    # vgg16 = vgg16.cuda()


    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


    # optimizer for psc2code is rmsprop 
    optimizer = optim.Adam()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    y_predict = []
    y_true = []

    model,avg_train_losses,avg_valid_losses = train_model1(model, feature_extractor, criterion, optimizer, num_epochs=100)
    torch.save(model.state_dict(), "experiment1_update.pth")
    