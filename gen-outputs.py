import sys
import cv2
import numpy as np
import os

from glob 				import glob
from os.path 			import splitext, basename, isfile
from src.utils 			import image_files_from_folder
from src.drawing_utils 	import draw_label,  write2img
from src.label 			import lread, Label

YELLOW = (0, 255, 255)
RED = (0, 0, 255)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

img_files = image_files_from_folder(input_dir)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	I = cv2.imread(img_file)

	detected_cars_labels = '%s/%s_cars.txt' % (output_dir, bname)

	Lcar = lread(detected_cars_labels)

	sys.stdout.write('%s' % bname)

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

						sys.stdout.write(',%s' % lp_str)
	cv2.imwrite('%s/%s_output.png' % (output_dir, bname), I)
	sys.stdout.write('\n')
