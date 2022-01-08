
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
import random
import numpy
# give a list for psc2code video dealing time
# def getlist_psc2code():
#     output = {}
#     rightlist = []
#     faultlist = []
#     database = "data_copy/test"
#     for category in os.listdir(database):
#         # print(category) 
#         if category == "non-screencast":
#             for video in os.listdir(os.path.join(database,category)):
#                 rightlist.append(video)
#                 for img in os.listdir(os.path.join(database,category,video)):
#                     # if "frame" not in img:
#                         # print(img)
#                     output[video] = int(1.6*len(os.listdir(os.path.join(database,category,video))))
#         else:
#             for video in os.listdir(os.path.join(database,category)):
#                 faultlist.append(video)
#                 for img in os.listdir(os.path.join(database,category,video)):
#                     # if "frame" not in img:
#                         # print(img)
#                     output[video] = len(os.listdir(os.path.join(database,category,video)))
#     return output,rightlist,faultlist


# psc2codelist,nonlist,yeslist = getlist_psc2code()
# count = 0
# for video in psc2codelist.keys():
#     psc2codelist[video] = int(psc2codelist[video]*1.88)
#     count+= psc2codelist[video]

# print(len(nonlist))
# print(len(yeslist))

# # wrong one: MkJ1jY8ubKs
# # 2.5倍
# def random1():
#     count = 0
#     test_positive = random.sample(list(psc2codelist), k=13)
#     test_negative = list(set(list(psc2codelist)).difference(set(test_positive)))
#     # print(len(list(psc2codelist)))
#     # print(len(test_positive))
#     # print(len(test_negative))
#     tp = 0
#     fp = 0
#     fn = 0
#     for item in test_positive:
#         if item in yeslist:
#             tp += 1
#         if item in nonlist:
#             fp+=1
#     for item in test_negative:
#         if item in yeslist:
#             fn+=1
#     return tp,fp,fn
# allpre = []
# allrecall = []
# allfi = []
# for i in range(20):
#     tp,fp,fn = random1()
#     pre = float(tp)/(tp+fp)
#     allpre.append(pre)
#     recall = float(tp)/(tp+fn)
#     allrecall.append(recall)
#     allfi.append(2*recall*pre/(recall+pre))
#     # print("%d,%d,%d"%(tp,fp,fn))
# print("the pre is %.3f, the recall is %.3f, the F-1 is %.3f"%(numpy.mean(allpre),numpy.mean(allrecall),numpy.mean(allfi))) 

def random1():
    count = 0
    psc2codelist = [i for i in range(22)]
    print(psc2codelist)
    yeslist = [i for i in range(14)]
    nonlist = [i for i in range(14,22)]
    print(yeslist)
    print(nonlist)
    test_positive = random.sample(list(psc2codelist), k=11)
    test_negative = list(set(list(psc2codelist)).difference(set(test_positive)))
    # print(len(list(psc2codelist)))
    # print(len(test_positive))
    # print(len(test_negative))
    tp = 0
    fp = 0
    fn = 0
    for item in test_positive:
        if item in yeslist:
            tp += 1
        if item in nonlist:
            fp+=1
    for item in test_negative:
        if item in yeslist:
            fn+=1
    return tp,fp,fn
allpre = []
allrecall = []
allfi = []
for i in range(20):
    tp,fp,fn = random1()
    pre = float(tp)/(tp+fp)
    allpre.append(pre)
    recall = float(tp)/(tp+fn)
    allrecall.append(recall)
    allfi.append(2*recall*pre/(recall+pre))
    # print("%d,%d,%d"%(tp,fp,fn))
print("the pre is %.4f, the recall is %.4f, the F-1 is %.4f"%(numpy.mean(allpre),numpy.mean(allrecall),numpy.mean(allfi))) 