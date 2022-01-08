# firstly should read the json file to get the labels.
import json
import subprocess
import sys
import os
import time

def read_json():
    # video name :'Tutorial 14 - Creare un menù con lo switch (Java).mp4'
    jsonpath = "/home/PSC2CODE/webapp/lables/Tutorial 14 - Creare un men con lo switch (Java)_SsyblLU4fZg.json"
    with open(jsonpath) as f:
        data = json.load(f)
        # print(data["labels"])
        # print(list(data["labels"].keys()))
        return data["labels"]


# shoud cut the frames by ffmpeg
# download and setting the ffmpeg version 4.1.6
# note that ffmpeg is external programming, so we need use subprocess
def callsubprocess(timeset,label,item):
    video_path = "video_for_test/Tutorial 14 - Creare un menù con lo switch (Java).mp4"
    print(os.path.exists(video_path))
    outputpath = ""+label+"/"+item+".png"
    cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-vframes", "1", "-q:v", "2",outputpath]
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
    video_path = "video_for_test/Tutorial 14 - Creare un menù con lo switch (Java).mp4"
    print(os.path.exists(video_path))
    outputpath = ""+label+"/"+item+".png"
    cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-vframes", "1", "-q:v", "2",outputpath]
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
    dic = read_json()

    # callsubprocess_test()


    # whole process
    for item in list(dic.keys()):
        timeset = timestamp_to_time(int(item))
        # print(timeset)
        # print(dic[item])
        callsubprocess(timeset,dic[item],item)
