# python3

from pytube import YouTube
import json
import os
import io
# 读取labels里所有的视频文件
def read_video_records():
    videopath = "webapp/lables/"
    listdir = os.listdir(videopath)
    return listdir


# 读取视频的名字和hash
def readvideos(videopath,count):
    # print(os.path.exists(videopath))
    with open(videopath) as f :
        data = json.load(f)

        #
        # this part have been done manully
        #

        # 有的有video name， 有的只有video，在只有video的情况下需要补足video hash
        # 先将video替换为 video_name

        # 替换
        # test = "video_hash"
        # if test not in data:
        #     print(videopath)
        #     data["video_name"] = data.pop("video")
    # return data["video_hash"], data["video_name"]
        return data["video_hash"]



def download_video():
    hash, name = readvideos()
    download_path = "video/psc2code_video"
    download_url = "http://youtube.com/watch?v="+str(hash)
    print(download_url)
    yt = YouTube(download_url)
    print(yt.streams)

def testdownload(hash):
    download_path = ""
    download_url = "http://youtube.com/watch?v="+str(hash)
    print(download_url)
    yt = YouTube(download_url)

    # this step could be imporved by using more high quality videos 
    # note that 13 or 14 videos don't have 720p version videos, so I don't download them

    yt.streams.get_by_itag(22).download(download_path)


def truetestdownload():
    download_url = "http://youtube.com/watch?v=SsyblLU4fZg"
    print(download_url)
    yt = YouTube(download_url)
    # yt.streams.get_audio_only().download(download_path)    
    yt.streams.first().download()    
    yt.streams.get_by_itag(22).download()


if __name__ == '__main__':
    listdir = read_video_records()
    count = 0
    for video in listdir:
        videopath = "webapp/lables/"+video
        video_hash = readvideos(videopath,count)
        print(video_hash)   
        try:
            testdownload(video_hash)
        except:
            print("error/n/n/n")
            count+=1
    print("count is :"+str(count))

