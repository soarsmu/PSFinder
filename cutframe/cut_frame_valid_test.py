# -*- coding: UTF-8 -*-
# firstly should read the json file to get the labels.
import json
import subprocess
import sys
import os
import time
from json import load
from shutil import copyfile

def read_video():
    videopath = "/home/PSC2CODE/chengran/video_data/psc2code_video"
    videodir = os.listdir(videopath)
    return videodir

def find_json():
    # video name :'Tutorial 14 - Creare un menù con lo switch (Java).mp4'
    jsonpath = "/home/PSC2CODE/webapp/labels/"
    jsondir = os.listdir(jsonpath)
    return jsondir

def read_json(jsontxt):
    jsonpath = "/home/PSC2CODE/webapp/labels/"+jsontxt
    print(os.path.exists(jsonpath))
    with open(jsonpath) as f:
        data = load(f)
        # print(data["video_name"])
        # if data.has_key("video"):
        if "video" in data.keys():
            data["video_name"]=data['video']
        if os.path.exists("/home/PSC2CODE/chengran/video_data/psc2code_video/"+data["video_name"]+".mp4"):   
            return data["labels"],data["video_name"]
        else:
            return 0,0


# shoud cut the frames by ffmpeg
# download and setting the ffmpeg version 4.1.6
# note that ffmpeg is external programming, so we need use subprocess
def callsubprocess(video_path,output_mid_path):
    # video_path = "/home/PSC2CODE/chengran/video_for_test/Tutorial 14 - Creare un menù con lo switch (Java).mp4"
    print(os.path.exists(video_path))
    outputpath = output_mid_path
    cmds = ["ffmpeg","-i", video_path, "-r", "1", "-f", "image2",outputpath+"%d.png", "-nostdin"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)


# translate time stamp to time 
def timestamp_to_time(timestamp):
    h_time = timestamp//3600
    # print(h_time)
    m_time = (timestamp-h_time*3600)//60
    # print(m_time)
    s_time = (timestamp-h_time*3600-m_time*60)
    # print(s_time)
    return str("%d:%d:%d"%(h_time,m_time,s_time))

def callsubprocess_test():
    video_path = "/home/PSC2CODE/chengran/video_data/psc2code_video/Java Video Tutorial 11.mp4"
    print(os.path.exists(video_path))
    outputpath = "/home/PSC2CODE/chengran/test"
    cmds = ["ffmpeg","-i", video_path, "-r", "1", "-f", "image2",outputpath+"%d.png", "-nostdin"]
    # ffmpeg -ss 01:23:45 -i input -vframes 1 -q:v 2 output.jpg
    # cmds = ["ffmpeg","-ss","00:11:11", "-i", video_path, "-vframes", "1", "-q:v", "2",outputpath]
    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)




# cutting whole process
# mkdir 1 2 3 4 for lables
# cp those frames into that lables
# for test, only need one video file
if __name__ == "__main__":
    count = 0
    jsondir = find_json()
    for json in jsondir:
        print(json)
        label, video_name = read_json(json)
        print(video_name)

        if video_name!=0:

            print("vidoe name is "+str(video_name))
            if os.path.exists("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name)==False:
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/origin/"+video_name)
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name)
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/1/")
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/2/")
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/3/")
                os.mkdir("/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/4/")
        #     # delete those video_json that don't have download file    
            videopath = "/home/PSC2CODE/chengran/video_data/psc2code_video/"+video_name+".mp4"
            outputpath = "/home/PSC2CODE/chengran/experiment2_data/origin/"+video_name+"/"



            # if os.path.exists(outputpath):
            #     for img in os.listdir(outputpath):
            #         if img[0:-4] in label.keys():
            #             number = img[0:-4]
            #             print(label[number])
            #         if int(label[number]) == 1:
            #             print("yes")

            callsubprocess(videopath,outputpath)
            # usefuldir = []

            # # 删除没有标签的frame
            for img in os.listdir(outputpath):
                if img[0:-4] not in label.keys():
                    os.remove(outputpath+img)
                else:
                    number =  img[0:-4]
                    if int(label[number]) == 1:
                        copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/1/"+img)
                    if int(label[number]) == 2:
                        copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/2/"+img)
                    if int(label[number]) == 3:
                        copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/3/"+img)
                    if int(label[number]) == 4:
                        copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/4/"+img)

                    # print(outputpath+img)
            #     # 复制到实验2中去
            #     # 在实验2中需要这个数据
            #     # else:
            #     # print(label)
            #     else:
            #         number = img[0:-4]
            #         print(label[number])

            #     if label[number] == 1:
            #         copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/1/"+img)
            #     if label[number] == 2:
            #         copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/2/"+img)
            #     if label[number] == 3:
            #         copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/3/"+img)
            #     if label[number] == 4:
            #         copyfile(outputpath+img,"/home/PSC2CODE/chengran/experiment2_data/psc2code_data/"+video_name+"/4/"+img)
            #             # os.remove(outputpath+img)
            #     else:
            #         print("fault")
            # # for timestamp in list(label.keys()):

            # #     # if os.path.exists(videopath):
            # #     #     print("true")
            # #     # timeset = timestamp_to_time(int(timestamp))
            # #     print(timestamp)
            # #     print(label[timestamp])


