import os, random, shutil
import json
import math
from shutil import copyfile


def get_valid_positive_psc2code_data_train():
    sourcepath = "/home/PSC2CODE/chengran/experiment2_data/psc2code_data"
    videodir = os.listdir(sourcepath)
    image_path = []
    for file in videodir:
        # print(file)
        if file in os.listdir("/home/PSC2CODE/chengran/data_copy/valid_frame_update_data/"):
            # print(file)

            for subfile in os.listdir(os.path.join(sourcepath,file)):
                # print(subfile)
                if subfile=="1":
                    for img in  os.listdir(os.path.join(sourcepath,file,subfile)):
                        name = "positive"+file+img
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/train/"+name)
                if subfile=="3":
                    for img in  os.listdir(os.path.join(sourcepath,file,subfile)):
                        name = "nagetive"+file+img
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/train/"+name)
                #     print("2 is yes")
            # for img in os.listdir(os.path.join(sourcepath,video)):
            #     print(os.path.join(sourcepath,video,img))
            #     image_path.append(os.path.join(sourcepath,video,img))

        if file not in os.listdir("/home/PSC2CODE/chengran/data_copy/valid_frame_update_data/") and file not in os.listdir("/home/PSC2CODE/chengran/data_copy/test/psc2code/"):
            print("there must be some fault")
            print(file)



    return image_path


def get_valid_positive_psc2code_data_test():
    sourcepath = "/home/PSC2CODE/chengran/experiment2_data/psc2code_data"
    videodir = os.listdir(sourcepath)
    image_path = []
    for file in videodir:
        # print(file)
        if file in os.listdir("/home/PSC2CODE/chengran/data_copy/test/psc2code/"):
            # print(file)

            for subfile in os.listdir(os.path.join(sourcepath,file)):
                # print(subfile)
                if subfile=="1":
                    for img in  os.listdir(os.path.join(sourcepath,file,subfile)):
                        name = "positive"+file+img
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/test/"+name)
                if subfile=="3":
                    for img in  os.listdir(os.path.join(sourcepath,file,subfile)):
                        name = "nagetive"+file+img
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/test/"+name)
                #     print("2 is yes")
            # for img in os.listdir(os.path.join(sourcepath,video)):
            #     print(os.path.join(sourcepath,video,img))
            #     image_path.append(os.path.join(sourcepath,video,img))

        if file not in os.listdir("/home/PSC2CODE/chengran/data_copy/valid_frame_update_data/") and file not in os.listdir("/home/PSC2CODE/chengran/data_copy/test/psc2code/"):
            print("there must be some fault")
            print(file)



    return image_path


def shutil_data():
    sourcepath = "/home/PSC2CODE/chengran/experiment2_data/tmp/"
    filedir = os.listdir(sourcepath) 
    img_list = []
    train = []
    for img in filedir:
        img_list.append(os.path.join(sourcepath,img))
    train = random.sample(img_list,math.ceil(len(filedir)*0.9))
    for img in train:
        outputpath = "/home/PSC2CODE/chengran/experiment2_data/train/"
        shutil.copyfile(img,outputpath+img.split("/")[-1])

    test = list(set(img_list)-set(train))
    for img in test: 
        outputpath = "/home/PSC2CODE/chengran/experiment2_data/test/"
        # print(img.split("/")[-1])  
        shutil.copyfile(img,outputpath+img.split("/")[-1])


def get_valid_train_otheride_data():
    sourcepath = "/home/PSC2CODE/chengran/experiment2_data/train_otheride"
    videodir = os.listdir(sourcepath)
    image_path = []
    for file in videodir:
        if file[:11] in os.listdir("/home/PSC2CODE/chengran/data_copy/valid_otherIDE"):
            for subfile in os.listdir(os.path.join(sourcepath,file)):
                if subfile =="valid":
                    # print(subfile)
                    for img in os.listdir(os.path.join(sourcepath,file,subfile)):
                        # print(img)
                        name = "positive"+file+img
                        # print(name)
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/train/"+name)
    for file in videodir:
        if file[:11] in os.listdir("/home/PSC2CODE/chengran/data_copy/valid_otherIDE"):
            for subfile in os.listdir(os.path.join(sourcepath,file)):
                if subfile =="invalid":
                    # print(subfile)
                    for img in os.listdir(os.path.join(sourcepath,file,subfile)):
                        # print(img)
                        name = "nagetive"+file+img
                        print(name)
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/train/"+name)

def get_valid_test_otheride_data():
    sourcepath = "/home/PSC2CODE/chengran/experiment2_data/test_otheride"
    videodir = os.listdir(sourcepath)
    image_path = []
    for file in videodir:
        if file[:11] in os.listdir("/home/PSC2CODE/chengran/data_copy/test/otheride") and "check" not in file:
            for subfile in os.listdir(os.path.join(sourcepath,file)):
                if subfile =="valid":
                    # print(subfile)
                    for img in os.listdir(os.path.join(sourcepath,file,subfile)):
                        # print(img)
                        name = "positive"+file+img
                        print(name)
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/test/"+name)
    for file in videodir:
        if file[:11] in os.listdir("/home/PSC2CODE/chengran/data_copy/test/otheride"):
            for subfile in os.listdir(os.path.join(sourcepath,file)):
                # print(subfile)
                if subfile =="invalid":
                    # print(subfile)
                    for img in os.listdir(os.path.join(sourcepath,file,subfile)):
                        # print(img)
                        name = "nagetive"+file+img
                        print(name)
                        copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/test/"+name)


        # if "check" not in file:
        #     for subfile in os.listdir(os.path.join(sourcepath,file)):
        #         if "valid" in subfile and "invalid" not in subfile:
        #             for img in os.listdir(os.path.join(sourcepath,file,subfile)):
        #                 name = "positive"+file+img  
        #                 copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/tmp/"+name)
        #         if "invalid"  in subfile:
        #             for img in os.listdir(os.path.join(sourcepath,file,subfile)):
        #                 name = "negative"+file+img
        #                 copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/tmp/"+name)
            

# get_valid_positive_psc2code_data_train()
get_valid_positive_psc2code_data_test()
# get_valid_train_otheride_data()
get_valid_test_otheride_data()









# get_valid_nagative_psc2code_data()
            
    #     for subfile in os.listdir(os.path.join(sourcepath,file)):
    #         # print(subfile)
    #         if subfile=="1":
    #             for img in  os.listdir(os.path.join(sourcepath,file,subfile)):
    #                 name = "positive"+file+img
    #                 copyfile(os.path.join(sourcepath,file,subfile,img),"/home/PSC2CODE/chengran/experiment2_data/tmp/"+name)
    #         # if subfile=="2":
    #         #     print("2 is yes")
    #     # for img in os.listdir(os.path.join(sourcepath,video)):
    #     #     print(os.path.join(sourcepath,video,img))
    #     #     image_path.append(os.path.join(sourcepath,video,img))
    # return image_path