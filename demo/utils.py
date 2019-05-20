import numpy as np
import cv2
import os
from scipy import signal

# used for linear mapping
def linear_mapping(images):
    max_value = images.max()
    min_value = images.min()
    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a
    image_after_mapping = parameter_a * images + parameter_b
    return image_after_mapping

# pre-processing the image
def pre_process(img):
	# get the size of the img
	height = img.shape[0]
	width = img.shape[1]
	img = np.log(img + 1) 
	img = (img - np.mean(img)) / (np.std(img) + 1e-5)
	# use the hanning window
	window = window_func_2d(height, width)
	img = img * window
	return img

# create window
def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row
    return win

# for rotate
def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the image
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

# pre train the filter on the first frame
def _pre_training(args, init_frame, G):
	height = G.shape[0]
	width = G.shape[1]
	fi = cv2.resize(init_frame, (width, height))
	# pre-process img
	fi = pre_process(fi)
	Ai = G * np.conjugate(np.fft.fft2(fi))
	Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
	for _ in range(args['num_pretrain']):
		if args['rotate']:
			fi = pre_process(random_warp(init_frame))
		else:
			fi = pre_process(init_frame)
		Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
		Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
	return Ai, Bi

# get the ground-truth gaussian reponse
def _get_gauss_response(args, img, gt):
	# get the shape of the image
	height = img.shape[0]
	width = img.shape[1]
	# get the mesh grid
	xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object
	center_x = gt[0] + 0.5 * gt[2]
	center_y = gt[1] + 0.5 * gt[3]
	# cal the distance
	dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * args['sigma'])
	# get the response map
	response = np.exp(-dist)
	# normalize
	response = linear_mapping(response)
	return response

# it will get the first ground truth of the video
def _get_init_ground_truth(args, img_path):
	gt_path = os.path.join(img_path, 'groundtruth.txt')
	with open(gt_path, 'r') as f:
		# just read the first frame
		line = f.readline()
		gt_pos = line.split(',')
	return [float(element) for element in gt_pos]

# load images from a folder
def load_images_from_folder(folder):
	images = []
	for filename in sorted(os.listdir(folder)):
		#names.append(filename)
		img = cv2.imread(os.path.join(folder,filename))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret,img=cv2.threshold(img,127,1,cv2.THRESH_BINARY) # 0=black
		img ^= 1 # 1=black
		if img is not None:
			images.append(img)
	return images

# to determine whether MOSSE will lose the target or not
def failOrNot(old_surface, mask_filtered, pos):
	q = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2RGB)
	gray = cv2.cvtColor(np.multiply(q,255), cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) # either 0 or 255
	img, contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	new_surface = 0
	center = False
	recoverable = False

	#find the contour of the drone
	for j in range(len(contours)):
		dist = cv2.pointPolygonTest(contours[j],(int(pos[0]+pos[2]/2), int(pos[1]+pos[3]/2)),False)
		
		if (dist == 1): # when shadow contains center point
			center = True
			# threshold
			new_surface = cv2.contourArea(contours[j]) # update new_surface
			if (new_surface >= 2.5 * old_surface):
				return [True, old_surface] 

	if not center: # when shadow does NOT contain center point
		gray_cut = gray[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]]
		ret, binary = cv2.threshold(gray_cut,127,255,cv2.THRESH_BINARY) # either 0 or 255
		img, contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		#print(contours[0])
		#print(gray_cut.shape)
		center_pos = [int(pos[3]/2.0),int(pos[2]/2.0)]
		dist = float('inf')
		index = -1

		for j in range(len(contours)):
			x,y,w,h = cv2.boundingRect(contours[j])
			d1 = (x- center_pos[0])**2    +  (y- center_pos[1])**2
			d2 = (x+w- center_pos[0])**2  +  (y- center_pos[1])**2
			d3 = (x- center_pos[0])**2    +  (y+h- center_pos[1])**2
			d4 = (x+w- center_pos[0])**2  +  (y+h- center_pos[1])**2
			dmin = np.minimum(d1, np.minimum(d2, np.minimum(d3, d4 ) ) )
			if (dmin < dist):
				index = j
				dist = dmin

		if index != -1:
			new_surface = cv2.contourArea(contours[index])  # update new_surface
			if (new_surface >= 2.5 * old_surface):
				return [True, old_surface] 

	return [False, new_surface] # not fail


def lmosse(args,pos,prev_frame,frame_gray,mask_filtered):
	margin = (pos[2]+pos[3])/3
	lpos = pos.copy()
	ix,iy,w,h = lpos
	ix = ix- margin
	iy = iy- margin
	w = w + 2*margin
	h = h + 2*margin
	lregion = [int(ix),int(iy),int(w),int(h)]
	# start to draw the gaussian response
	response_map = _get_gauss_response(args, prev_frame, lregion)
	# start to create the training set
	# get the goal
	g = response_map[lregion[1]:lregion[1]+lregion[3], lregion[0]:lregion[0]+lregion[2]]
	b_fi = prev_frame[lregion[1]:lregion[1]+lregion[3], lregion[0]:lregion[0]+lregion[2]]
	#heatmap of the drone
	heatmap = signal.fftconvolve(b_fi, b_fi, mode = 'same')
	G = np.fft.fft2(g)
	# pre-training
	Ai, Bi = _pre_training(args,b_fi, G)
	pos1 = lregion.copy()
	clip_pos = np.array([pos1[0], pos1[1], pos1[0]+pos1[2], pos1[1]+pos1[3]]).astype(np.int64)
	Hi = Ai / Bi
	b_fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
	Gi = Hi * np.fft.fft2(b_fi)
	gi = linear_mapping(np.fft.ifft2(Gi))

	# mosse anwser combine with shadow mask and heatmap
	mask_filtered = mask_filtered[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
	gi = np.multiply(heatmap,np.multiply(gi,mask_filtered))

	# find the max pos
	max_value = np.max(gi)
	max_pos = np.where(gi == max_value)
	dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
	dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
	lpos[0] = lpos[0] + dx  
	lpos[1] = lpos[1] + dy
	return lpos,lregion





