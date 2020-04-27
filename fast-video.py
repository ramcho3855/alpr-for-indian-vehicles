import sys
import cv2
import numpy as np
import traceback
import os
import darknet.darknet as dn
import time

from glob 					import glob
from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir, isfile
from src.drawing_utils 		import draw_label,  write2img
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder, nms
from darknet.darknet 		import detect
from src.label 				import dknet_label_conversion, lread, Label
from src.load_model  		import load_system


def detect_vehicle(img_path, output_dir, loaded_models,bname):
	print('Searching for vehicles...')

	print('\tScanning %s' % img_path)


	vehicle_net, vehicle_meta, vehicle_threshold = loaded_models[0]

	R, _ = detect(vehicle_net, vehicle_meta, img_path.encode('utf-8'), thresh=vehicle_threshold)

	R = [r for r in R if r[0].decode(encoding='utf-8') in ['car']]

	print('\t\t%d cars found' % len(R))

	if len(R):
		Iorig = cv2.imread(img_path)
		WH = np.array(Iorig.shape[1::-1], dtype=float)

		for i,r in enumerate(R):
			cx, cy, w, h = (np.array(r[2])/np.concatenate((WH,WH))).tolist()
			tl = np.array([cx - w/2., cy - h/2.])
			br = np.array([cx + w/2., cy + h/2.])
			label = Label(0, tl, br)
			Icar = crop_region(Iorig, label)
			cv2.imwrite('%s/%s_%dcar.png' % (output_dir, bname, i), Icar)


def detect_lp(output_dir,loaded_models,Iorig_name):

	imgs_paths = glob('%s/%s_*car.png' % (output_dir, Iorig_name))
	print('Searching for license plates...')

	for i, img_path in enumerate(imgs_paths):


		bname = splitext(basename(img_path))[0]
	
		lp_net, lp_meta, lp_threshold = loaded_models[1]
		
		R, _ = detect(lp_net, lp_meta, img_path.encode('utf-8'), thresh=lp_threshold)

		if len(R):
			Iorig = cv2.imread(img_path)
			WH = np.array(Iorig.shape[1::-1], dtype=float)

			for i, r in enumerate(R):
				cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label = Label(0, tl, br)
				Ilp = crop_region(Iorig, label)
				cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp)


def ocr_lp(output_dir, loaded_models,Iorig_path):

	Iorig_name = basename(splitext(Iorig_path)[0])
	imgs_paths = sorted(glob('%s/%s_*_lp.png' % (output_dir, Iorig_name)))

	print('Performing Character Recognition...')
	Iorig = cv2.imread(Iorig_path)
	
	all_lp_str = ''
	
	for i, img_path in enumerate(imgs_paths):


		bname = basename(splitext(img_path)[0])
		
		ocr_net, ocr_meta, ocr_threshold = loaded_models[2]
		
		R, (width, height) = detect(ocr_net, ocr_meta, img_path.encode('utf-8'), thresh=ocr_threshold, nms=None)

		if len(R):

			L = dknet_label_conversion(R, width, height)
			L = nms(L, .45)

			L.sort(key=lambda x: x.tl()[0])
			lp_str = ''.join([chr(l.cl()) for l in L])
			all_lp_str += lp_str + " " 
			print('\t\tLP: %s' % lp_str)

		else:

			print('No characters found')
	

	write2img(Iorig, Label(0), all_lp_str)
	cv2.imwrite('%s/%s_output.png' % (output_dir,Iorig_name ), Iorig)
	

def finish_frame(output_dir, show):
	for f in glob(output_dir + "/*_lp.png"):
		os.remove(f)
	for f in glob(output_dir + "/*car.png"):
		os.remove(f)

	

if __name__ == '__main__':
	
	try:
	
		input_dir  = sys.argv[1]
		video_path = sys.argv[2]
		output_dir = sys.argv[3]
		csv_file = sys.argv[4]
		show = int(sys.argv[5])
		if not isdir(output_dir):
			makedirs(output_dir)
		
		loaded_models = load_system()
		cap= cv2.VideoCapture(video_path)
		video_name = splitext(basename(video_path))[0]
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		i=1
		start = time.time()
		while(cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
				break
			img_path = input_dir + '/' + str(i)+'.jpg'
			cv2.imwrite(img_path,frame)
			bname = str(i)
			detect_vehicle(img_path, output_dir, loaded_models,bname)
			detect_lp(output_dir,loaded_models,bname)
			ocr_lp(output_dir,loaded_models,img_path)
			
			
			#if show:
			#	for f in glob(output_dir + '/*.png'):
			#		frame = cv2.imread(f)
			#		cv2.imshow("frame",frame)
			#		key = cv2.waitKey(1)
			#		if key & 0xFF == ord('q'):
			#			break
			#	for f in glob(output_dir + '/*.png'):
			#		os.remove(f)
			i+=1
		#os.system("ffmpeg -framerate {0} -i {1}/%01d_output.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {1}/{2}_output.mp4".format(fps, output_dir,video_name))
		#for f in glob(output_dir + "/*_output.png"):
		#	os.remove(f)
		print("FPS of video: {:5.2f}".format(i/(time.time()-start)))
		finish_frame(output_dir, show)
		for f in glob(input_dir + '/*.jpg'):
			os.remove(f)
		

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
