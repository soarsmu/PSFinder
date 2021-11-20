# PSFinder
- [PSFinder](#psfinder)

Replication Package for PSFinder, an automatic tool to identify live-coding screencasts.

# Download Videos
Our tool could download videos from YouTube based on [PyTube](https://pytube.io/en/latest/).
Related functions are in /download/download_invalid_java.py:
+ input the specific url for YouTube videos
+ output the highest resolution videos

# PSFinder
## Pre-process
+ Anaconda
+ Pytorch
+ Creat Anaconda environment with dependent libraries (environment.yml):
  
  ```conda env create -f environment.yml```

## Pipeline
Before running the pipeline, you may want to check the sub-folder

```/download```

for automatically downloading videos from YouTube. 

If you prefer to use YouTube official API to search and download videos, please kindly check the official [API documentation](https://developers.google.com/youtube/v3) further to sign in and employ the service.

To run our pipeline in stages,  
+ For *frame sampler*, which extract each sampled frame and mark the duplicate frames:
  
  ```/cutframe/cut_frame_invalid.py```
  
+ For *video classifier*, which identify whether a video is live-coding screencast: 

  To fine-tune the pre-trained model, the script is in the sub-folder:

  ```/Train_Test```

  one will also find the source code we employ to implement data preprocess.

  The script for *video-level classification strategy* part is in:

  ```/eval/evaluate_PSFinder.py```
  
  The input images should be the same as the file path of above model output. 

