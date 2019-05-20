import argparse
import cv2
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument("-o", "--output_folder", required=False, default='.', help="output video file")
parse.add_argument("-n", "--img_name", required=True, help="image name")
args = vars(parse.parse_args())
output_folder = args['output_folder']
img_name = args['img_name']
margin = 5
x_min = 1000000
y_min = 1000000
x_max = 0
y_max = 0

I = cv2.imread(img_name)

h=I.shape[0]
w=I.shape[1]
m=I[:,:,2]  # [0 0 0]  or [255 255 255]
for i in range(0,h):
	for j in range(0,w):
		if m[i,j] == 0:
			y_min = min(y_min,i)
			x_min = min(x_min,j)
			y_max = max(y_max,i)
			x_max = max(x_max,j)

print("Without margin: "+str(x_min)+", "+str(y_min)+", "+str(x_max-x_min)+", "+str(y_max-y_min))
print(margin)
print("With margin: "+str(x_min-margin)+", "+str(y_min-margin)+", "+str(x_max-x_min+margin+margin)+", "+str(y_max-y_min+margin+margin))
out_name = output_folder+'/'+img_name.split('.')[0] + ".txt"
f = open(out_name,'w') 
f.write(str(x_min-margin)+", "+str(y_min-margin)+", "+str(x_max-x_min+margin+margin)+", "+str(y_max-y_min+margin+margin))
                                              # width              height
f.close()
print("done")