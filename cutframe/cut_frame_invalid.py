
# -*- coding: UTF-8 -*-

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


def callsubprocess(video_path,video):
    videoname = os.listdir(video_path)
    videopath = "/home/PSC2CODE/chengran/video_data/codeless_video/"+video+"/"+videoname[0]
    print(videopath)


    outputpath = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/"+video

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", videopath,"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",outputpath+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)


def callsubprocess_for_psc2code(video_path,video):
    os.mkdir("/home/PSC2CODE/chengran/test_for_psc2code")
    videopath = video_path
    print(videopath)


    outputpath = "/home/PSC2CODE/chengran/test_for_psc2code"

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", videopath,"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",outputpath+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)
 
# 对图片进行统一化处理
def get_thum(image, size=(64,64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
 
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res
 
def getthesim(image1add,image2add):
    image1 = Image.open(image1add)
    image2 = Image.open(image2add)
    cosin = image_similarity_vectors_via_numpy(image1, image2)
    # print('图片余弦相似度',cosin)
    return cosin

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

def callsubprocess_for_otherIDE(videoname,video_path,output_path):

    print(videoname)
    print(video_path)
    print(output_path)
    video = os.listdir(video_path)[0]
    
    # videopath = "/home/PSC2CODE/chengran/video_data/codeless_video/"+video+"/"+videoname[0]
    # print(videopath)


    # outputpath = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/"+video

    # # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", os.path.join(video_path,video),"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",output_path+"/%d.png"]
    # # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    # print("\n")
    print(cmds)
    # print("\n")
    subprocess.call(cmds)

if __name__ == "__main__":

    # 切割图片
    source_path = "/home/PSC2CODE/chengran/video_data/valid_video_otherIDE/"
    output_path = "/home/PSC2CODE/chengran/data_copy/valid_otherIDE"
    whole_dir = os.listdir(source_path)

    for video_dir in whole_dir:
        print("the ide file is "+video_dir)
        for video in os.listdir(os.path.join(source_path,video_dir)):
            path = os.path.join(source_path,video_dir,video)
            # print(path)
            video_name = path[-11:]
            if not os.path.exists(os.path.join(output_path,video_name)):
                os.mkdir(os.path.join(output_path,video_name))
            callsubprocess_for_otherIDE(video_name,path,os.path.join(output_path,video_name))
        #     path = "/home/PSC2CODE/chengran/video_data/codeless_video/"+video
        #     if path != "/home/PSC2CODE/chengran/video_data/codeless_video/videolog.md":
        #         # 切割
        #         if not os.path.exists("/home/PSC2CODE/chengran/frame_data/invalid_frame_data/"+video):
        #             os.mkdir("/home/PSC2CODE/chengran/frame_data/invalid_frame_data/"+video)
        #             callsubprocess(path,video)
        #         if not os.path.exists("/home/PSC2CODE/chengran/frame_data/invalid_frame_data/"+video+"/frames.txt"):
            diff_frames(os.path.join(output_path,video_name))
            



    # for test
    # callsubprocess_for_psc2code("/home/PSC2CODE/chengran/video_data/psc2code_video/Java operators.mp4","Java operators.mp4")
    # diff_frames("/home/PSC2CODE/chengran/test_for_psc2code")


    # # 计算相似度
    # print("test")
    # count = 0
    # # image1 = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/0VX66NUoBRo/1.png"
    # # image2 = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/0VX66NUoBRo/2.png"
    # path = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/0VX66NUoBRo/"
    # for frame in os.listdir(path):
    #     image1 = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/0VX66NUoBRo/1.png"
    #     image2 = "/home/PSC2CODE/chengran/frame_data/invalid_frame_data/0VX66NUoBRo/"+frame
    #     cosin = getthesim(image1,image2)
    #     if cosin <= 0.9:
    #         count+=1
    # print(count)

