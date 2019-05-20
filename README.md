# Drone-Shadow-Tracking
Xiaoyan Zou, Ruofan Zhou, Majed El Helou, Sabine Süsstrunk  

## Citation
```
update this later
```

## Code
requirement: python, opencv, matlab, photoshop
The codes are tested on Ubuntu 16.04, matlab R2017b and python 3.6.5

preparation:
Please prepare files in order.
1) The video

2) The video frames.
toFrames.py which will create a "VIDEO_NAME_frames" folder in the folder "data_frames" at root and put automatically video frames in the folder. The frames are used for shadow detection and the initial bounding box of the drone.
   command in terminal: python toFrames.py -n VIDEO_NAME.mp4

3) The ground truth image
Take the first frame, and use PhotoShop to obtain a ground truth of the drone. Note that the background should be white and the drone's shadow should be black. Please name the ground truth image as video name and place it at root.
eg. video_name = vid.mp4, ground truth img_name = vid.png

4) The location of the initial bounding box of the drone.
Apply the ground truth image to "box_location.py" to create the "VIDEO_NAME.txt" at root which represents the location of the initial bounding box of the drone.
   command in terminal: python box_location.py -run.py -n VIDEO_NAME.IMAGE_FORMAT -o OUTPUT_FOLDER

5) The shadow detection masks
Open "create_shadow_mask.m" and modify frame_folder and masks_folder. And then run the codes to obtain the shadow detection masks.


how to run the code:
Having above files ready, we can run the main codes.
  command in terminal: python run.py -n VIDEO_NAME.mp4 -o OUTPUT_FOLDER
  To save video and frames: python run.py -n VIDEO_NAME.VIDEO_FORMAT -o OUTPUT_FOLDER -sv true -sf true


### References: 
MOSSE Tracking Algorithm, original implementation
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - [Visual object tracking using adaptive correlation filters](https://ieeexplore.ieee.org/document/5539960/).
Code link: https://github.com/TianhongDai/mosse-object-tracking

Shadow Detection code:
Derek Bradley and Gerhard Roth, “Adaptive thresholding using the integral image,” Journal of Graphics Tools,vol. 12, no. 2, pp. 13–21, 2007.
Given by Vatsal Shah and Vineet Gandhi, “An iterative approach for shadow removal in document images,” in In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 1892–1896.




