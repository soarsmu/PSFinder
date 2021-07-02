import cv2
import PIL
import numpy as np
import keras.preprocessing
from matplotlib import pyplot as plt
import os
from keras.preprocessing.image import load_img


sourcepath = "/home/PSC2CODE/chengran/experiment2_data/test"

# sourcepath = "/home/PSC2CODE/chengran/data_copy/valid_frame_update_data/Java operators"

model = keras.models.load_model('weights-new.h5')
imgs = []
label = []
name = []
for img in os.listdir(sourcepath):
    if "positive" in img:
        label.append(int(0))
    else:
        label.append(int(1))
    img_keras = np.array(load_img(os.path.join(sourcepath,img),target_size=(300,300,3)))
    imgs.append(img_keras)
    name.append(img)
predicted = model.predict(np.array(imgs))
# print "predict: ", predicted
result = np.argmax(model.predict(np.array(imgs)), axis=1)

tp = 0
fp = 0
fn = 0
tn = 0
total = len(label)

for i in range(len(label)):

    print("\n")
    if int(label[i])==0 and int(result[i])==0:
        tp +=1
    if int(label[i])==1 and int(result[i])==0:
        fp +=1
    if int(label[i])==0 and int(result[i])==1:
        fn +=1
    if int(label[i])==1 and int(result[i])==1:
        tn +=1

print(tp)
print(fp)
print(fn)
print(tn)
print(total)

print("acc is %.3f" % (correct/len(label)))













