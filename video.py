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



def detect_vehicle(img_path, input_dir, output_dir, loaded_models):
	print('Searching for vehicles...')

	print('\tScanning %s' % img_path)

	bname = basename(splitext(img_path)[0])
					

	if not isdir(output_dir):
		makedirs(output_dir)
	vehicle_net, vehicle_meta, vehicle_threshold = loaded_models[0]

	R, _ = detect(vehicle_net, vehicle_meta, img_path.encode('utf-8'), thresh=vehicle_threshold)
	
	R = [r for r in R if r[0].decode(encoding='utf-8') in ['car']]

	print('\t\t%d cars found' % len(R))

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


def detect_lp(input_dir,loaded_models):

	output_dir = input_dir


	imgs_paths = glob('%s/*car.png' % input_dir)
	print('Searching for license plates...')

	for i, img_path in enumerate(imgs_paths):

		print('\t Processing %s' % img_path)

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


def ocr_lp(input_dir, loaded_models):

	output_dir = input_dir

	imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

	print('Performing Character Recognition...')

	for i, img_path in enumerate(imgs_paths):

		print('\tScanning %s' % img_path)

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


def gen_output(input_dir, output_dir, csv_file):
	YELLOW = (0, 255, 255)
	RED = (0, 0, 255)

	csv = open(csv_file,'a')
	img_files = image_files_from_folder(input_dir)

	for img_file in img_files:

		bname = splitext(basename(img_file))[0]

		I = cv2.imread(img_file)

		detected_cars_labels = '%s/%s_cars.txt' % (output_dir, bname)

		Lcar = lread(detected_cars_labels)

		csv.writelines('%s' % bname)

		if Lcar:

			for i, lcar in enumerate(Lcar):

				draw_label(I, lcar, color=YELLOW, thickness=3)

				lp_label = '%s/%s_%dcar_lp.txt' % (output_dir, bname, i)
				lp_label_str = '%s/%s_%dcar_lp_str.txt' % (output_dir, bname, i)

				imgs_path = glob('%s/%s_%dcar.png' % (input_dir, bname, i))

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

							csv.writelines(',%s' % lp_str)
		cv2.imwrite('%s/%s_output.png' % (output_dir, bname), I)
		csv.writelines('\n')
	csv.close()
	

def finish_frame(input_dir, output_dir):
	for f in glob(output_dir + "/*_lp.png"):
		os.remove(f)
	for f in glob(output_dir + "/*car.png"):
		os.remove(f)
	for f in glob(output_dir + "/*_cars.txt"):
		os.remove(f)
	for f in glob(output_dir + "/*_lp.txt"):
		os.remove(f)
	for f in glob(output_dir + "/*_str.txt"):
		os.remove(f)
	for f in glob(input_dir + '/*.jpg'):
		os.remove(f)
	

		
		
	



if __name__ == '__main__':
	
	try:
	
		input_dir  = sys.argv[1]
		video_path = sys.argv[2]
		output_dir = sys.argv[3]
		csv_file = sys.argv[4]

		
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
			#frame = cv2.resize(frame,())
			cv2.imwrite(input_dir + '/' + str(i)+'.jpg',frame)
			img_path = input_dir + '/' + str(i)+'.jpg'
			detect_vehicle(img_path, input_dir, output_dir, loaded_models)
			detect_lp(output_dir,loaded_models)
			ocr_lp(output_dir,loaded_models)
			gen_output(input_dir, output_dir,csv_file)
			finish_frame(input_dir, output_dir)
			
			i+=1
		print("FPS of video: {:5.2f}".format(i/(time.time()-start)))
		os.system("ffmpeg -framerate {0} -i {1}/%01d_output.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {1}/{2}_output.mp4".format(fps, output_dir,video_name))
		for f in glob(output_dir + "/*_output.png"):
			os.remove(f)
		
		

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
