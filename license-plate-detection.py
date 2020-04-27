import sys, os
import cv2
import traceback
import numpy as np

import darknet.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from src.utils 				import crop_region
from darknet.darknet 		import detect
from glob 					import glob

if __name__ == '__main__':

	try:

		input_dir = sys.argv[1]
		output_dir = input_dir

		lp_threshold = .25

		lp_weights = 'data/lp-detector/lpd.weights'
		lp_netcfg = 'data/lp-detector/lpd.cfg'
		lp_dataset = 'data/lp-detector/lpd.data'

		lp_net = dn.load_net(lp_netcfg.encode('utf-8'), lp_weights.encode('utf-8'), 0)
		lp_meta = dn.load_meta(lp_dataset.encode('utf-8'))

		imgs_paths = glob('%s/*car.png' % input_dir)

		print('Searching for license plates...')

		for i, img_path in enumerate(imgs_paths):

			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]

			R, _ = detect(lp_net, lp_meta, img_path.encode('utf-8'), thresh=lp_threshold)
			#R = [r for r in R if r[0] in ['lp']]

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

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
