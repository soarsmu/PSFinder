
import os
import subprocess
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
from PIL import Image

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


def video_predict(list1,list2,duplicate):

    overall = list1 + list2
    overall_new = [int(x) for x in overall]
    list1 = [int(x) for x in list1]
    list1.sort()
    overall_new.sort()
    pre_pointer = 0
    label = 0
    for i in range(len(overall_new)-2):
        for j in range(len(list1)-2):
            # print("%d %d %d"%(overall_new[i],overall_new[i+1],overall_new[i+2]))
            # print("%d %d %d"%(list1[j],list1[j+1],list1[j+2]))
            if list1/(list1+list2)>=0.5:
                if overall_new[i]==list1[j] and overall_new[i+1]==list1[j+1] and overall_new[i+2]==list1[j+2] and overall_new[i+3]==list1[j+3]:
                    label = 1
                    # print(list1[j])
                    # print(list1[j+1])
                    # print(list1[j+2])
                    return label
    return label
def callsubprocess(video_path):

    videopath = video_path
    videoname = video_path.split("/")[-1][:-4]
    
    # outputpath = "evaluation/sample/"
    # outputpath = "evaluation/sample_psc2code/"
    outputpath = "evaluation/sample_otheride/"

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    if not os.path.exists(outputpath+videoname):
        os.makedirs(outputpath+videoname)
    cmds = ["ffmpeg","-i", videopath,"-r", "1/60", "-f","image2", "-vf", "fps=fps=1", "-q:v", "2",outputpath+videoname+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)

    
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

data_transform  = transforms.Compose([
    transforms.Resize((300,300)),

    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

])

# def ifvideovalid(dic):
#     # print(list(dic.keys()))
#     list1 = ['valid','valid','valid','valid']
#     list2 = list(dic.values())
#     flag = False
#     for i in range(len(list2) - len(list1) + 1):
#         if list2[i: i+len(list1)] == list1:
#             flag = True
#             break
#     return flag


# def predict_sample():

#     class_names = ['invalid',"valid"]

#     for data in os.listdir("evaluation/sample"):
#         video_path = os.path.join("evaluation/sample",data)
#     # net= torch.load("experiment2.pth")
#         from torchvision_vgg import VGG,vgg16_bn1
#         net = vgg16_bn1()
#         net = nn.DataParallel(net)
#         net.load_state_dict(torch.load("experiment2_1.pth"))
#         net = net.cuda()
#         print(video_path)
#         valid = []
#         invalid = []
#         with torch.no_grad():
#             net.eval()
#             for imgs in os.listdir(video_path):
#                 if ".txt" not in imgs:
#                     img = Image.open(os.path.join(video_path,imgs))
#                     img_ = data_transform(img).unsqueeze(0) 
#                     img_ = img_.cuda()
#                     outputs = net(img_)

#                     _, indices = torch.max(outputs,1)
#                     percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
#                     perc = percentage[int(indices)].item()
#                     result = class_names[indices]
#                     number = imgs[:-4]
#                     # print('predicted:', result,number)
#                     if result == "valid":
#                         valid.append(int(number))
#                     else:
#                         invalid.append(int(number))
#             print(sorted(valid))
#             print(sorted(invalid))

def diff_frames(frame_folder, thre=0.05, metric="NRMSE"):
    '''

    The duplicated frames would be deleted and the file 'frames.txt' contains the name of the filtered frames
    @param frame_folder: the location of video frames 
    @param thre: threhold of dissimilarity
    @param metris: NRMSE or SSIM
    '''
    
    print("used parameters: ", thre, "/", metric == 'SSIM')
    fout = open("%s/frames.txt" % frame_folder, "w")

    frame_seq = []
    for frame in os.listdir(frame_folder):
        if not frame.endswith(".png"):
            continue
        frame_seq.append(int(frame[0:-4]))

    frame_seq = sorted(frame_seq)
    print(frame_seq)


    filter_frames = []
    pre_img = None
    for frame in frame_seq:
        img = cv2.imread("%s/%d.png" % (frame_folder, frame))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if pre_img is not None:
            if metric == 'SSIM':
                sim = 1 - skimage.measure.compare_ssim(pre_img, img_gray)
            else:
                sim = skimage.measure.compare_nrmse(pre_img, img_gray)

            if sim > thre:
                pre_img = img_gray
                filter_frames.append(frame)
            else:
                os.remove("%s/%d.png" % (frame_folder, frame))
        else:
            pre_img = img_gray
            filter_frames.append(frame)
    
    print("filtered frame number")
    fout.write(" ".join([str(f) for f in filter_frames]))
    fout.close()



def predict_sample_afterdelete():

    class_names = ['invalid',"valid"]
    validnumber = 0
    invalidnumber = 0

    return_dic = {}

    for data in os.listdir("evaluation_data/sample"):
        video_path = os.path.join("evaluation_data/sample",data)
        from transformers import ViTFeatureExtractor, ViTForImageClassification
    # net= torch.load("experiment2.pth")
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        net = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        net.load_state_dict(torch.load("experiment1_update.pth"))
        # net = nn.DataParallel(net)
        # net.load_state_dict(torch.load("experiment2_1.pth"))
        net = net.cuda()
        print(video_path)
        valid = []
        invalid = []
        duplicate = []
        with torch.no_grad():
            net.eval()
            for imgs in os.listdir(video_path):
                if ".txt" not in imgs:
                    img = Image.open(os.path.join(video_path,imgs))
                    img_ = data_transform(img).unsqueeze(0) 
                    img_ = img_.cuda()
                    outputs = net(img_)

                    _, indices = torch.max(outputs,1)
                    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    perc = percentage[int(indices)].item()
                    result = class_names[indices]
                    number = imgs[:-4]
                    # print('predicted:', result,number)
                    if result == "valid":
                        valid.append(int(number))
                    if result =='invalid':
                        invalid.append(int(number))
                    else:
                        duplicate.append(int(number))
            print(sorted(valid))
            print(sorted(invalid))
            validnumber += len(valid)
            invalidnumber += len(invalid)
            predict_result = video_predict(valid,invalid,duplicate)
            print(data)
            # print(validnumber)
            # print(invalidnumber)
            with open("evaluation/log/videopredict.txt","a")as f:
                f.write(video_path+" "+str(predict_result)+"\n")
            # if os.path.exists("data_copy/test/non-screencast/"+data[-11:]):
                # return_dic[video_path] = [len(os.listdir("data_copy/test/non-screencast/"+data[-11:])),predict_result,0]

def predict_psc2code_afterdelete():

    return_dic = {}

    class_names = ['invalid',"valid"]

    validnumber = 0
    invalidnumber = 0

    for data in os.listdir("evaluation_data/sample_psc2code"):
        video_path = os.path.join("evaluation_data/sample_psc2code",data)
    # net= torch.load("experiment2.pth")
        from torchvision_vgg import VGG,vgg16_bn1
        net = vgg16_bn1()
        net.load_state_dict(torch.load("experiment1_update.pth"))
        # net.load_state_dict(torch.load("experiment2_1.pth"))

        net = net.cuda()
        print(video_path)
        valid = []
        invalid = []
        with torch.no_grad():
            net.eval()
            for imgs in os.listdir(video_path):
                if ".txt" not in imgs:
                    img = Image.open(os.path.join(video_path,imgs))
                    img_ = data_transform(img).unsqueeze(0) 
                    img_ = img_.cuda()
                    outputs = net(img_)

                    _, indices = torch.max(outputs,1)
                    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    perc = percentage[int(indices)].item()
                    result = class_names[indices]
                    number = imgs[:-4]
                    # print('predicted:', result,number)
                    if result == "valid":
                        valid.append(int(number))
                    else:
                        invalid.append(int(number))
            print(sorted(valid))
            validnumber += len(valid)
            invalidnumber += len(invalid)
            print(sorted(invalid))
            predict_result = video_predict(valid,invalid)
            print(data)
    print(validnumber)
    print(invalidnumber)
            # with open("evaluation/log/visdeopredict.txt","a")as f:
                # f.write(video_path+" "+str(video_predict(valid,invalid))+"\n")
            # return_dic[video_path] = [len(os.listdir("data_copy/test/psc2code/"+data)),predict_result,0]
    # return return_dic

def predict_otheride_afterdelete():

    class_names = ['invalid',"valid"]
    validnumber = 0
    invalidnumber = 0
    for data in os.listdir("evaluation_data/sample_otheride"):
        video_path = os.path.join("evaluation_data/sample_otheride",data)
    # net= torch.load("experiment2.pth")
        from torchvision_vgg import VGG,vgg16_bn1
        net = vgg16_bn1()
        net.load_state_dict(torch.load("experiment1_update.pth"))
        net = net.cuda()
        print(video_path)
        valid = []
        invalid = []
        with torch.no_grad():
            net.eval()
            for imgs in os.listdir(video_path):
                if ".txt" not in imgs:
                    img = Image.open(os.path.join(video_path,imgs))
                    img_ = data_transform(img).unsqueeze(0) 
                    img_ = img_.cuda()
                    outputs = net(img_)

                    _, indices = torch.max(outputs,1)
                    percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    perc = percentage[int(indices)].item()
                    result = class_names[indices]
                    number = imgs[:-4]
                    # print('predicted:', result,number)
                    if result == "valid":
                        valid.append(int(number))
                    else:
                        invalid.append(int(number))
            print(sorted(valid))
            print(sorted(invalid))
            validnumber += len(valid)
            invalidnumber += len(invalid)
            print(video_predict(valid,invalid))
            print(data)
            # with open("evaluation/log/videopredict.txt","a")as f:
                # f.write(video_path+" "+str(video_predict(valid,invalid))+"\n")
    print(validnumber)
    print(invalidnumber)


# for videodir in os.listdir("video_data/codeless_video"):
#     # print(videodir)
#     if ".md" not in videodir and videodir in os.listdir("data_copy/test/non-screencast"):
#         video = os.listdir(os.path.join("video_data/codeless_video",videodir))[0]
#         videopath = os.path.join("video_data/codeless_video",videodir,video)
#         callsubprocess(videopath)
    

# predict_sample()



# for data in os.listdir("evaluation_data/sample"):
#     video_path = os.path.join("evaluation_data/sample",data)
#     diff_frames(video_path, thre=0.05, metric="NRMSE")
# 预测non-screencasts


print(predict_sample_afterdelete())



