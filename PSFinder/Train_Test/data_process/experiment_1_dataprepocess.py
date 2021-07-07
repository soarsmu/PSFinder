import os, random, shutil
import json
import math


def get_valid_psc2code_data():
    sourcepath = "psc2code_data/"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        for subfile in os.listdir(os.path.join(sourcepath,video)):
            if subfile == "1" or subfile == "3":
                for img in os.listdir(os.path.join(sourcepath,video,subfile)):
                    print(os.path.join(sourcepath,video,subfile,img))
                    image_path.append(os.path.join(sourcepath,video,subfile,img))
    return image_path

def get_invalid_psc2code_data():
    sourcepath = "psc2code_data/"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        for subfile in os.listdir(os.path.join(sourcepath,video)):
            if subfile == "4":
                for img in os.listdir(os.path.join(sourcepath,video,subfile)):
                    print(os.path.join(sourcepath,video,subfile,img))
                    image_path.append(os.path.join(sourcepath,video,subfile,img))
    return image_path

def get_valid_otherIDE_data():
    sourcepath = "data_copy/valid_otherIDE"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        for img in os.listdir(os.path.join(sourcepath,video)):
            print(os.path.join(sourcepath,video,img))
            image_path.append(os.path.join(sourcepath,video,img))
    return image_path

def getinvalid_data_small():
    sourcepath = "data_copy/invalid_frame_data"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        for img in os.listdir(os.path.join(sourcepath,video)):
            print(os.path.join(sourcepath,video,img))
            image_path.append(os.path.join(sourcepath,video,img))
    return image_path

def getinvalid_data_big():
    sourcepath = "data_copy/invald_too_largefile"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        for img in os.listdir(os.path.join(sourcepath,video)):
            print(os.path.join(sourcepath,video,img))
            image_path.append(os.path.join(sourcepath,video,img))
    return image_path

# 随机抽取
def random_select(filedir,picknumber):
    filenumber = len(filedir)
    # picknumber = 1000
    # ratio
    # rate = 0.1
    # picknumber = rate * filenumber
    sample = random.sample(filedir,picknumber)

# every video extract 1000 from big file
def random_invalid_select():
    sourcepath = "data_copy/invald_too_largefile"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        random1 = []
        for img in os.listdir(os.path.join(sourcepath,video)):
            random1.append(os.path.join(sourcepath,video,img))
        # select 1000
        if len(random1)>=600:
            sample = random.sample(random1,600)
        else:
            sample = random.sample(random1,len(random1))
        image_path.extend(sample)   
    return image_path

def random_normal_select(sourcepath):
    # sourcepath = "data_copy/invald_too_largefile"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        random1 = []
        for img in os.listdir(os.path.join(sourcepath,video)):
            random1.append(os.path.join(sourcepath,video,img))
        # select 1000
        if len(random1)>=1000:
            sample = random.sample(random1,1000)
        else:
            sample = random.sample(random1,len(random1))
        image_path.extend(sample)   
    return image_path


def random_valid_select():
    sourcepath = "experiment1_data/label_otheride_experiment1"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        if "check" in video:
            print(video)
            random1 = []
            for img in os.listdir(os.path.join(sourcepath,video)):
                random1.append(os.path.join(sourcepath,video,img))
            # select 1000
            if len(random1)>=600:
                sample = random.sample(random1,600)
            else:
                sample = random.sample(random1,len(random1))
            image_path.extend(sample)   
        else:
            for subfile in os.listdir(os.path.join(sourcepath,video)):
                if subfile =="invalid" or subfile =="valid":
                    for img in os.listdir(os.path.join(sourcepath,video,subfile)):
                        image_path.append(os.path.join(sourcepath,video,subfile,img))   
                        # print(os.path.exists(os.path.join(sourcepath,video,subfile,img)))
    return image_path

def random_invalid_otheride_select():
    sourcepath = "experiment1_data/label_otheride_experiment1"
    videodir = os.listdir(sourcepath)
    image_path = []
    for video in videodir:
        if "check" not in video:
            for subfile in os.listdir(os.path.join(sourcepath,video)):
                if subfile =="unrelated":
                    for img in os.listdir(os.path.join(sourcepath,video,subfile)):
                        image_path.append(os.path.join(sourcepath,video,subfile,img))   
                        # print(os.path.exists(os.path.join(sourcepath,video,subfile,img)))
    return image_path

def move_image(filedir,label,logfile):
    dic = {}
    output_path = "frame_data/pytorch_data/"
    for img in filedir:
        if img.endswith(".png"):
            imgname = img.split("/")[-3]+label+img.split("/")[-2]+img.split("/")[-1]
            print(imgname)
            dic[img]=imgname
            shutil.copyfile(img,output_path+imgname)
    json_log = json.dumps(dic)
    with open("frame_data/log/"+logfile+".json","w") as f:
        f.write(json_log)

def move_image_test(filedir,label,logfile):
    dic = {}
    output_path = "frame_data/test/"
    for img in filedir:
        if img.endswith(".png"):
            imgname = label+img.split("/")[-2]+img.split("/")[-1]
            print(imgname)
            dic[img]=imgname
            shutil.copyfile(img,output_path+imgname)
    json_log = json.dumps(dic)
    with open("frame_data/log/"+logfile+"_test_.json","w") as f:
        f.write(json_log)

def divide_train():
    sourcepath = "frame_framelevel_data/pytorch_data/"
    filedir = os.listdir(sourcepath)
    # print(len(filedir))
    img_list = []
    train = []
    for img in filedir:
        img_list.append(os.path.join(sourcepath,img))
    train = random.sample(img_list,math.ceil(len(filedir)*0.8))

    file = open("frame_framelevel_data/log/train.txt","w")
    file.write(str(train))
    file.close() 
    for img in train: 
        outputpath = "frame_framelevel_data/train/"
        # print(img.split("/")[-1])  
        shutil.copyfile(img,outputpath+img.split("/")[-1])
    # print(len(train))
    test = list(set(img_list)-set(train))
    # print(len(test))
    test = list(set(img_list)-set(train))

    file = open("frame_framelevel_data/log/test.txt","w")
    file.write(str(test))
    file.close()    
    for img in test: 
        outputpath = "frame_framelevel_data/test/"
        # print(img.split("/")[-1])  
        shutil.copyfile(img,outputpath+img.split("/")[-1])
    # print(len(train))


if __name__ == "__main__":
    #use for train dataset
    #
    #

    # # img file from valid other ide
    # random_valid_ide = random_valid_select()
    # # print(len(random_valid_ide))
    # move_image(random_valid_ide,"valid",logfile = "valid_otheride")

    # # img file from invalid other ide
    # random_invalid_ide = random_invalid_otheride_select()
    # print(len(random_invalid_ide))
    # move_image(random_invalid_ide,"invalid",logfile = "invalid_otheride")

    # # # img file from valid psc2code
    # psc2code_data = get_valid_psc2code_data()
    # move_image(psc2code_data,"valid",logfile ="valid_psc2code")

    # # img file from invalid psc2code
    # psc2code_data = get_invalid_psc2code_data()
    # move_image(psc2code_data,"invalid",logfile ="invalid_psc2code")

    # # # img file from invalid small
    # invalid_small = getinvalid_data_small()
    # move_image(invalid_small,"invalid",logfile = "invalid_samll")

    # # # img file from invalid big
    # random_invalid = random_invalid_select()
    # move_image(random_invalid,"invalid",logfile = "invalid_big")

    # divide_train()

    # use for test dastaset
    #
    #

    # img file from valid other ide
    # random_valid_ide = random_normal_select("data_copy/test/otheride")
    # print(len(random_valid_ide))
    # move_image_test(random_valid_ide,"valid",logfile = "valid_ide")

    # # img file from valid psc2code
    # psc2code_data = random_normal_select("data_copy/test/psc2code")
    # move_image_test(psc2code_data,"valid",logfile ="valid_psc2code")

    # # img file from invalid small
    # invalid_small = random_normal_select("data_copy/test/non-screencast")
    # move_image_test(invalid_small,"invalid",logfile = "invalid_samll")

    count = 0 
    path = "frame_data/pytorch_data"
    for img in os.listdir(path):
        if "invalid" in img:
            count+=1
    print(count)