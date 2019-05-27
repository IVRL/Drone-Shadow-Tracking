# Drone Shadow Tracking
Xiaoyan Zou, Ruofan Zhou, [Majed El Helou](https://majedelhelou.github.io/) and Sabine Süsstrunk.

This is a Python implementation for the Drone Shadow Tracking [paper](https://infoscience.epfl.ch/record/265717/).


## Codes
### Dependencies
- Python 3.6.5
- Opencv 3.4.3
- MATLAB R2017b

### Quick start (Demo)
In `demo` folder, you can easily reproduce ths results reported in the paper by simply run the following command: 
```
python run.py -n VIDEO_NAME -o OUTPUT_FOLDER
```
*VIDEO_NAME* can be the following:
1_simple.mp4, 2_bird.mp4, 3_comFace.mp4, 3_four.mp4, 4_newspaper.mp4, 5_bag.mp4, 5_grass.mp4, 5_people.mp4


### How to run on your own video
Note: all prepared files should be placed at root folder.

#### Step 1: obtain video frames
`toFrames.py` which will create video frames to the `VIDEO_NAME_frames` folder in `data_frames` folder at root. The video frames are used for shadow detection and the initial bounding box of the drone.
```
python toFrames.py -n VIDEO_NAME.mp4
```

#### Step 2: create shadow detection results
Open `create_shadow_mask.m` and modify *frame_folder* and *masks_folder*. Then run it to obtain the shadow detection masks.


#### Step 3: ground truth image (for the first frame)
Take the first frame of the video, and use PhotoShop to obtain a ground truth of the drone shadow. Note that the background should set to white and the drone's shadow should set to black. Please name the ground truth image as video name and place it at root.
- eg. video_name = vid.mp4  -->  ground truth img_name = vid.png


#### Step 4: initial bounding box of the drone
Apply the ground truth image to `box_location.py` to create the `VIDEO_NAME.txt` at root which represents the location of the initial bounding box of the drone.
```
python box_location.py -n VIDEO_NAME.IMAGE_FORMAT
```


#### Step 5: run tracking
Having above files ready, we can run the main codes.
```
python run.py -n VIDEO_NAME.mp4 -o OUTPUT_FOLDER
```

To save video and frames: 
```
python run.py -n VIDEO_NAME.VIDEO_FORMAT -o OUTPUT_FOLDER -sv true -sf true
```

  To save video and frames: python run.py -n VIDEO_NAME.VIDEO_FORMAT -o OUTPUT_FOLDER -sv true -sf true


### References: 
MOSSE Tracking Algorithm, original implementation
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the **python** implementation of the - [Visual object tracking using adaptive correlation filters](https://ieeexplore.ieee.org/document/5539960/).
Code link: https://github.com/TianhongDai/mosse-object-tracking

Shadow Detection code:
Derek Bradley and Gerhard Roth, “Adaptive thresholding using the integral image,” Journal of Graphics Tools,vol. 12, no. 2, pp. 13–21, 2007.
Given by Vatsal Shah and Vineet Gandhi, “An iterative approach for shadow removal in document images,” in In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 1892–1896.




