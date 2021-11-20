# this file is to download videos that don't contain code but the topic is related to java and programming.
from pytube import YouTube

# store place : video_data/codeless_video

def download(download_hash):
    download_url = "http://youtube.com/watch?v="+download_hash
    download_path = yourpath+download_hash
    # download_url = "http://youtube.com/watch?v="+str(hash)
    print(download_url)
    yt = YouTube(download_url)

    # itag 22 means video has 720p quality and the format is mp4
    yt.streams.get_by_itag(22).download(download_path,filename =download_path)

def download_IDE(download_hash,ide):
    download_url = "http://youtube.com/watch?v="+download_hash
    download_path = yourpath+ide+"/"+download_hash
    # download_url = "http://youtube.com/watch?v="+str(hash)
    print(download_url)
    yt = YouTube(download_url)

    # itag 22 means video has 720p quality and the format is mp4
    yt.streams.get_by_itag(22).download(download_path,filename =download_hash)



#
# 记录视频信息
#
# note that only need the video hash
download_hash = input("download url is :")
# download_ide = input("IDE is")
download_IDE(download_hash,"VSCode")
# download(download_hash)
