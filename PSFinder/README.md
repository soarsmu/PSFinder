# PSFinder
- [PSFinder](#psfinder)

Replication Package for the paper "Efficient Search of Live-Coding Screencasts from Online Videos".

PSFinder aims to identify if a video is a live-coding screencasts. Programming videos on the Internet are valuable resources for learning programming skills. To find relevant videos, developers typically search online video platforms (e.g., YouTube) with keywords on topics they wish to learn. Developers often look for live-coding screencasts, in which the videos' authors perform live coding. 
Yet, not all programming videos are live-coding screencasts. In this work, we develop a tool PSFinder to identify live-coding screencasts. \toolname{} leverages a classifier to identify whether a video frame contains an IDE window. It uses a sampling strategy to pick a number of frames from an input video, runs the classifer on these frames, and then determines whether the video is a live-coding screencast based on frames classified as containing IDE window. In our preliminary experiment, PSFinder can effectively identify live-coding screencasts as it achieves an F1-score of 0.97.

***
# download videos
Our tool could download videos from YouTube based on [PyTube](https://pytube.io/en/latest/).
Related functions are in /download/download_invalid_java.py:
+ input the specific url for YouTube videos
+ output the highest resolution videos
# PSFinder
## Pre-process
+ Anaconda
+ Pytorch
+ Creat Anaconda environment with dependent libraries (environment.yml):
  ```
  conda env create -f environment.yml
  ```
## Pipeline
User can run the whole pipeline on run.py.

For running in stages, firstly 
+ run /cutframe/cut_frame_invalid.py
  
  to extracting every frame per minute and delete the most similar frames.
+ run our model to get results

  the model could be downloaded [here](https://drive.google.com/file/d/1De6yEzqOdMFn3htw3FZS96-6hvTxbZ1d/view?usp=sharing). You can run this script

  /eval/evaluate_PSFinder.py
  
  The input should be the output file path in above step, the output is the classification results. 
# Train
The train script is in the file: 

/Train_Test 

The modeil defination is in /Train_Test/torchvision_vgg.py

