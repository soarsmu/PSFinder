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
from typing import Union, List, Dict, Any, cast
# from torchvision_vgg import vgg16_bn1
# import torchvision_vgg
from torch.utils.model_zoo import load_url as load_state_dict_from_url
plt.ion()  


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # CIFAR 10 (7, 7) to (1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 1 * 1, 4096),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        #/root/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth
        state_dict = torch.load(
           "/root/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth", map_location=device
        )
        model.load_state_dict(state_dict)
    return model

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}






def vgg16_bn1(pretrained=False, progress=True, device="cpu", **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, device, **kwargs)







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
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate=0.1    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)


def mv_frames_to_dir(valid_path = "/home/PSC2CODE/chengran/data_copy/valid_frame_update_data",invalid_path = "/home/PSC2CODE/chengran/data_copy/invalid_frame_data",validdir = "/home/PSC2CODE/chengran/frame_data/valid_pytorch",invaliddir = "/home/PSC2CODE/chengran/frame_data/invalid_pytorch"):
    count = 0
    for video in os.listdir(valid_path):
        for img in os.listdir(os.path.join(valid_path,video)):
            # print(img)
            # print(os.path.join(valid_path,video,img))
            if not os.path.exists("/home/PSC2CODE/chengran/frame_data/pytorch_data/"+img):
                copyfile(os.path.join(valid_path,video,img),"/home/PSC2CODE/chengran/frame_data/pytorch_data/"+"valid"+img)
            count+=1
    print("the number of valid frames are "+str(count))
    count = 0
    for video in os.listdir(invalid_path):
        for img in os.listdir(os.path.join(invalid_path,video)):
            if not os.path.exists("/home/PSC2CODE/chengran/frame_data/pytorch_data/"+img):
                copyfile(os.path.join(invalid_path,video,img),"/home/PSC2CODE/chengran/frame_data/pytorch_data/"+"other"+img)
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
    # moveFile("/home/PSC2CODE/chengran/frame_data/pytorch_data/","/home/PSC2CODE/chengran/frame_data/test/")
    train_data = myDataSet("/home/PSC2CODE/chengran/frame_data/pytorch_data",data_transform)
    test_data = myDataSet("/home/PSC2CODE/chengran/frame_data/test",data_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=0)

    #vgg
    # vgg16 = models.vgg16_bn()
    # vgg16.load_state_dict(torch.load('vgg16_bn-6c64b313.pth'))
    # # vgg16 = models.vgg16(pretrained=True)
    # # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    # print(vgg16.classifier[6].out_features) # 1000 
    # # Freeze training for all layers
    # for param in vgg16.features.parameters():
    #     param.require_grad = False
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, 2)]) # Add our layer with 4 outputs
    # vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    # print(vgg16)
    # net = vgg16().to(device)



    print(VGGNet())
    net = VGGNet().to(device)
    # net = vgg16_bn1(pretrained=True,progress=True)
    
    # optimizer for psc2code is rmsprop 
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
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
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0     
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
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
       
    # 测试准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                
                maxk = max((1,5))
                label_resize = labels.view(-1, 1)
                _, predicted = outputs.topk(maxk, 1, True, True)
                total += labels.size(0)
                correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
                
                y_predict.append(predicted)
                y_true.append(labels)
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    torch.save(net.state_dict(), 'VGG_model.pth')


