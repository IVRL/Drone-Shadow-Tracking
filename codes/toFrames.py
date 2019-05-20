import cv2
import argparse
import os

parse = argparse.ArgumentParser()
parse.add_argument("-n", "--video_name", required=True, help="video name")
args = vars(parse.parse_args())
video_name = args['video_name']
vidcap = cv2.VideoCapture('./'+video_name)

vname = video_name.split('.')[0]
#creat paths
frames_path = './data_frames/'+vname+'_frames/'
if not os.path.exists(frames_path):
	os.mkdir(frames_path)

success,image = vidcap.read()
count = 0
success = True
while success:
  s=vname+"{0:0>3}".format(count)
  cv2.imwrite(frames_path+vname+s+'.png', image)
  success,image = vidcap.read()
  count += 1

