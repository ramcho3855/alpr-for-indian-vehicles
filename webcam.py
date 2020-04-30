import sys
import cv2
import numpy as np
import traceback
import os
import darknet.darknet as dn

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

	vehicle_net, vehicle_meta, vehicle_threshold = loaded_models[0]

	R, _ = detect(vehicle_net, vehicle_meta, img_path.encode('utf-8'), thresh=vehicle_threshold)

	R = [r for r in R if r[0].decode(encoding='utf-8') in ['car']]

	print('%d cars found' % len(R))

	if len(R):
		Iorig = cv2.imread(img_path)
		WH = np.array(Iorig.shape[1::-1], dtype=float)
		Lcars = []

		for i,r in enumerate(R):
			cx, cy, w, h = (np.array(r[2])/np.concatenate((WH,WH))).tolist()
			tl = np.array([cx - w/2., cy - h/2.])
			br = np.array([cx + w/2., cy + h/2.])
			label = Label(0, tl, br)
			Icar = crop_region(Iorig, label)
			Lcars.append(label)
			cv2.imwrite('%s/%s_%dcar.png' % (output_dir, bname, i), Icar)
		
		lwrite('%s/%s_cars.txt' % (output_dir, bname), Lcars)


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
			Llp = []

			for i, r in enumerate(R):
				cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
				tl = np.array([cx - w / 2., cy - h / 2.])
				br = np.array([cx + w / 2., cy + h / 2.])
				label = Label(0, tl, br)
				Ilp = crop_region(Iorig, label)
				Llp.append(label)
				cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp)
			
			lwrite('%s/%s_lp.txt' % (output_dir, bname), Llp)
		else:
			print('No license plate found')


def ocr_lp(output_dir, loaded_models,Iorig_path):

	Iorig_name = basename(splitext(Iorig_path)[0])
	imgs_paths = sorted(glob('%s/%s_*_lp.png' % (output_dir, Iorig_name)))

	Iorig = cv2.imread(Iorig_path)
	

	
	for i, img_path in enumerate(imgs_paths):


		bname = basename(splitext(img_path)[0])
		
		ocr_net, ocr_meta, ocr_threshold = loaded_models[2]
		
		R, (width, height) = detect(ocr_net, ocr_meta, img_path.encode('utf-8'), thresh=ocr_threshold, nms=None)

		if len(R):

			L = dknet_label_conversion(R, width, height)
			L = nms(L, .45)

			L.sort(key=lambda x: x.tl()[0])
			lp_str = ''.join([chr(l.cl()) for l in L])
			
			with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
				f.write(lp_str + '\n')
			
			print('\t\tLP: %s' % lp_str)

		else:

			print('No characters found')
	

def gen_output(output_dir):

	YELLOW = (0, 255, 255)
	RED = (0, 0, 255)

	bname = splitext(basename(img_path))[0]

	I = cv2.imread(img_path)

	detected_cars_labels = '%s/%s_cars.txt' % (output_dir, bname)

	Lcar = lread(detected_cars_labels)

	if Lcar:

		for i, lcar in enumerate(Lcar):

			draw_label(I, lcar, color=YELLOW, thickness=3)

			lp_label = '%s/%s_%dcar_lp.txt' % (output_dir, bname, i)
			lp_label_str = '%s/%s_%dcar_lp_str.txt' % (output_dir, bname, i)

			C = cv2.imread('%s/%s_%dcar.png' % (output_dir, bname, i))

			if isfile(lp_label):
				Llp = lread(lp_label)
				for j, llp in enumerate(Llp):
					draw_label(I, llp, color=RED, thickness=3, lp=True, lcar=lcar, C=C)
					if isfile(lp_label_str):
						with open(lp_label_str, 'r') as f:
							lp_str = f.read().strip()
						cwh = np.array(C.shape[1::-1]).astype(float)
						iwh = np.array(I.shape[1::-1]).astype(float)
						tl = tuple(np.add((llp.tl() * cwh).tolist(), (lcar.tl() * iwh).tolist()) / iwh)
						br = tuple(np.add((llp.br() * cwh).tolist(), (lcar.tl() * iwh).tolist()) / iwh)
						temp_label = Label(0, tl=tl, br=br)
						write2img(I, temp_label, lp_str)

	return I


def finish_frame(output_dir,img_path):
	for f in glob(output_dir + '/*'):
		os.remove(f)
	os.remove(img_path)
	

if __name__ == '__main__':
	
	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]
		

		if not isdir(output_dir):
			makedirs(output_dir)
		
		loaded_models = load_system()
		
		
#		url = "http://192.168.43.1:8080/video"
#		cap = cv2.VideoCapture(url)
			
		cap= cv2.VideoCapture(0)
		if not (cap.isOpened()):
			sys_exit(0)

		i = 1			
		fps = 30										#fps of webcam
			
		while(cap.isOpened()):
			
			ret, frame = cap.read()
			if ret == False:
				break
				
			#it will only process 3 frames each second
			if (i % (fps//3) == 0):
				img_path = input_dir + '/' + str(i)+'.jpg'
				cv2.imwrite(img_path,frame)
				bname = str(i)
				detect_vehicle(img_path, output_dir, loaded_models,bname)
				detect_lp(output_dir,loaded_models,bname)
				ocr_lp(output_dir,loaded_models,img_path)
				cv2.imshow('webcam', gen_output(output_dir))
				finish_frame(output_dir, img_path)
					
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
					
			i += 1

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)


