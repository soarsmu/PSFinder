
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
    videopath = "video_data/codeless_video/"+video+"/"+videoname[0]
    print(videopath)


    outputpath = "frame_data/invalid_frame_data/"+video

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", videopath,"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",outputpath+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)


def callsubprocess_for_psc2code(video_path,video):
    os.mkdir("test_for_psc2code")
    videopath = video_path
    print(videopath)


    outputpath = "test_for_psc2code"

    # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", videopath,"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",outputpath+"/%d.png"]
    # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    print("\n")
    print(cmds)
    print("\n")
    subprocess.call(cmds)
 
def get_thum(image, size=(64,64), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image
 
# calculate the distance for images
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

        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
 
def getthesim(image1add,image2add):
    image1 = Image.open(image1add)
    image2 = Image.open(image2add)
    cosin = image_similarity_vectors_via_numpy(image1, image2)
    return cosin

def diff_frames(frame_folder, thre=0.05, metric="NRMSE"):
    '''
    The duplicated frames would be deleted and the file 'frames.txt' contains the name of the filtered frames and the name of the duplicate frames
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
    duplicated_frame = []
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
                duplicated_frame.append(frame)
        else:
            pre_img = img_gray
            filter_frames.append(frame)
    
    print("Finished")
    fout.write(" ".join([str(f) for f in filter_frames]))
    fout.write("\n")
    fout.write(" ".join([str(f) for f in duplicated_frame]))

    fout.close()

def callsubprocess_for_otherIDE(videoname,video_path,output_path):

    print(videoname)
    print(video_path)
    print(output_path)
    video = os.listdir(video_path)[0]
    
    # videopath = "video_data/codeless_video/"+video+"/"+videoname[0]
    # print(videopath)


    # outputpath = "frame_data/invalid_frame_data/"+video

    # # cmds = ["ffmpeg","-i", videopath,"-r","1", "-vframes", "1", "-q:v", "2",outputpath+"/%d.png"]
    cmds = ["ffmpeg","-i", os.path.join(video_path,video),"-f","image2", "-vf", "fps=fps=1", "-q:v", "2",output_path+"/%d.png"]
    # # cmds = ["ffmpeg","-ss",timeset, "-i", video_path, "-r", "1", "-f", "image2", outputpath, "-nostdin"]

    # print("\n")
    print(cmds)
    # print("\n")
    subprocess.call(cmds)

if __name__ == "__main__":
    source_path = "video_data/valid_video_otherIDE/"
    output_path = "data_copy/valid_otherIDE"
    whole_dir = os.listdir(source_path)

    for video_dir in whole_dir:
        print("the ide file is "+video_dir)
        for video in os.listdir(os.path.join(source_path,video_dir)):
            path = os.path.join(source_path,video_dir,video)
            video_name = path[-11:]
            if not os.path.exists(os.path.join(output_path,video_name)):
                os.mkdir(os.path.join(output_path,video_name))
            callsubprocess_for_otherIDE(video_name,path,os.path.join(output_path,video_name))

            diff_frames(os.path.join(output_path,video_name))
            




