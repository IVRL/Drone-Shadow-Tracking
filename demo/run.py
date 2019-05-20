import numpy as np 
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from utils import linear_mapping, pre_process, random_warp, _pre_training, _get_gauss_response,_get_init_ground_truth, load_images_from_folder, failOrNot, lmosse

#some default params
initTracking = False
ix, iy = -1, -1
w, h = 0, 0
inteval = 30  
lr=0.325
sigma=100
num_pretrain=128
count = -1
old_surface = float('inf')
fail = False
prev_frame = []
kernel = np.ones((3,3), np.uint8) 

# main function
if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parse.add_argument("-o", "--output_folder", required=False, default='./', help="output video file")
	parse.add_argument("-n", "--video_name", required=True, help="video name")
	#mosse
	parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
	parse.add_argument('--sigma', type=float, default=100, help='the sigma')
	parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
	parse.add_argument('--rotate', action='store_true', help='if rotate image during pre-training.')
	parse.add_argument("-sv",'--svideo', default = True, help='save the output as a video')
	parse.add_argument("-sf",'--sframe', default = True, help='save result frames')
	
	# to complete 
	args = vars(parse.parse_args())
	output_folder = args['output_folder']
	v_name = args['video_name']
	video_name = v_name.split('.')[0]
	lr= args['lr']
	sigma= args['sigma']
	num_pretrain= args['num_pretrain']
	svideo = args['svideo']
	sframe = args['sframe']

	#creat paths
	out_path = './'+output_folder+'/'
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	#center position output_folder+'/'
	mosse_txt_name = out_path +video_name + "_centerPos_mosse.txt"
	fm = open(mosse_txt_name, 'w')	
	
	# load shadow masks
	mask_path = './data/'+video_name+'_mask/'
	masks = load_images_from_folder(mask_path)
	
	# begin the algorithm
	cap = cv2.VideoCapture('./data/' + v_name)
	
	# get ground truth box postion
	f = open('./data/' + video_name+".txt","r")
	line = f.readline().split(',')
	ix = int(line[0])
	iy = int(line[1])
	w = int(line[2])
	h = int(line[3])
	f.close()

	initTracking = True

	while(cap.isOpened()):
		count += 1

		ret, frame = cap.read()
		if not ret:
			break

		# get the image of the first frame (read as gray scale image)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if(initTracking):
			# create the video 
			if svideo:
				fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
				s= './'+output_folder+'/'+video_name.split('.')[0]+'_result'+".mp4"
				out = cv2.VideoWriter(s, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

			# initialise mosse
			init_gt = (ix,iy,w,h)
			init_gt = np.array(init_gt).astype(np.int64)
			# start to draw the gaussian response
			response_map = _get_gauss_response(args, frame_gray, init_gt)
			# start to create the training set 
			# get the goa
			g = response_map[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
			fi = frame_gray[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
			G = np.fft.fft2(g)

			# pre-training
			Ai, Bi = _pre_training(args,fi, G)# Ai:corr of G and fi  Bi:spectrum of fi
			# first frame
			Ai = lr * Ai
			Bi = lr * Bi
			pos = init_gt.copy()
			clip_pos = np.array([pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]]).astype(np.int64)
			initTracking = False

			# prepare for Lmosse
			prev_frame = frame_gray.copy()
			# prepare for original mosse response
			mosse_fi = fi.copy()
			mosse_Ai = Ai.copy()
			mosse_Bi = Bi.copy()
			mosse_clip_pos = clip_pos.copy()
			mosse_pos = pos.copy()
		else:
			#start tracking	
			frame_gray = frame_gray.astype(np.float32)
			#filter noise
			mask_filtered = cv2.dilate(cv2.erode(masks[count], kernel, iterations=1), kernel, iterations=1)
			fail, old_surface = failOrNot(old_surface, mask_filtered, pos)

			if fail: 

				lpos, lregion = lmosse(args,pos,prev_frame,frame_gray,mask_filtered)

				# draw
				cv2.rectangle(frame, (lpos[0], lpos[1]), (lpos[0]+lpos[2], lpos[1]+lpos[3]), (0, 255, 255), 2) # yello
				cv2.circle(frame,(int(lpos[0]+lpos[2]/2),int(lpos[1]+lpos[3]/2)), 3, (0,0,255), -1) # red
				cv2.rectangle(frame, (lregion[0], lregion[1]), (lregion[0]+lregion[2], lregion[1]+lregion[3]), (255, 255, 255), 2) # white

				# write down the bounding box location
				fm.write(str(lpos))
				fm.write('\n')

			else: 
				# mosse
				Hi = Ai / Bi
				# subWindow
				fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
				# keep win size unchanged
				fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
				Gi = Hi * np.fft.fft2(fi)  
				gi = linear_mapping(np.fft.ifft2(Gi))
				# find the max pos
				max_value = np.max(gi)
				max_pos = np.where(gi == max_value)
				dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
				dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
				# update the position
				pos[0] = pos[0] + dx  
				pos[1] = pos[1] + dy
				# trying to get the clipped position [xmin, ymin, xmax, ymax]
				clip_pos[0] = np.clip(pos[0], 0, frame.shape[1])
				clip_pos[1] = np.clip(pos[1], 0, frame.shape[0])
				clip_pos[2] = np.clip(pos[0]+pos[2], 0, frame.shape[1])
				clip_pos[3] = np.clip(pos[1]+pos[3], 0, frame.shape[0])
				clip_pos = clip_pos.astype(np.int64)
				# get the next fi using the new bounding box
				fi = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
				fi = pre_process(cv2.resize(fi, (init_gt[2], init_gt[3])))
				# update
				Ai = lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Ai
				Bi = lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - lr) * Bi
				# green bounding box when original mosse succeeds
				cv2.rectangle(frame, (pos[0], pos[1]), (pos[0]+pos[2], pos[1]+pos[3]), (0, 255, 255), 2) 
				# write down the bounding box location
				fm.write(str(pos))
				fm.write('\n')

			prev_frame = frame_gray.copy()


			#--------start ORIGINAL MOSSE RESPONSE--------

			mosse_Hi = mosse_Ai / mosse_Bi
			mosse_fi = frame_gray[mosse_clip_pos[1]:mosse_clip_pos[3], mosse_clip_pos[0]:mosse_clip_pos[2]]
			mosse_fi = pre_process(cv2.resize(mosse_fi, (init_gt[2], init_gt[3])))
			mosse_Gi = mosse_Hi * np.fft.fft2(mosse_fi)
			mosse_gi = linear_mapping(np.fft.ifft2(mosse_Gi))
			# find the max pos...
			mosse_max_value = np.max(mosse_gi)
			mosse_max_pos = np.where(mosse_gi == mosse_max_value)
			mosse_dy = int(np.mean(mosse_max_pos[0]) - mosse_gi.shape[0] / 2)
			mosse_dx = int(np.mean(mosse_max_pos[1]) - mosse_gi.shape[1] / 2)
			# update the position...
			mosse_pos[0] = mosse_pos[0] + mosse_dx
			mosse_pos[1] = mosse_pos[1] + mosse_dy
			# trying to get the clipped position [xmin, ymin, xmax, ymax]
			mosse_clip_pos[0] = np.clip(mosse_pos[0], 0, frame.shape[1])
			mosse_clip_pos[1] = np.clip(mosse_pos[1], 0, frame.shape[0])
			mosse_clip_pos[2] = np.clip(mosse_pos[0]+mosse_pos[2], 0, frame.shape[1])
			mosse_clip_pos[3] = np.clip(mosse_pos[1]+mosse_pos[3], 0, frame.shape[0])
			mosse_clip_pos = mosse_clip_pos.astype(np.int64)
			# get the current fi..
			mosse_fi = frame_gray[mosse_clip_pos[1]:mosse_clip_pos[3], mosse_clip_pos[0]:mosse_clip_pos[2]]
			mosse_fi = pre_process(cv2.resize(mosse_fi, (init_gt[2], init_gt[3])))
			# online update...
			mosse_Ai = lr * (G * np.conjugate(np.fft.fft2(mosse_fi))) + (1 - lr) * mosse_Ai
			mosse_Bi = lr * (np.fft.fft2(mosse_fi) * np.conjugate(np.fft.fft2(mosse_fi))) + (1 - lr) * mosse_Bi
			# visualize the tracking process...
			cv2.rectangle(frame, (mosse_pos[0], mosse_pos[1]), (mosse_pos[0]+mosse_pos[2], mosse_pos[1]+mosse_pos[3]), (255, 0, 0), 2)

			#--------end ORIGINAL MOSSE RESPONSE--------


			# put frame number
			cv2.putText(frame, 'Frame number: '+str(count), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2) # yellow text
			cv2.imshow('Tracking', frame)
			# write the txt file of the center position of the bounding box of each frame while tracking

			# if record, save the frames
			if svideo: 
				out.write(frame) 
			
			# save frames with bounding box
			if sframe :
				s = "{0:0>3}".format(count)
				path_box = out_path + '/'+video_name.split('.')[0]+'_framesWithBox/'
				if not os.path.exists(path_box):
					os.mkdir(path_box)
				cv2.imwrite(path_box +s+ '_box.png', frame)

		c = cv2.waitKey(inteval) & 0xFF
		if c==27 or c==ord('q'):
			break
		

	
	
	cap.release()
	if svideo:
		out.release()
	fm.close()
	cv2.destroyAllWindows()
