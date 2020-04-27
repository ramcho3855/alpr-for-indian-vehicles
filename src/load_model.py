import sys
import cv2
import numpy as np
import traceback

import darknet.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.darknet 		import detect


def load_system():
	
	try:
	
		loaded_models = []
		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/vehicle-detection.weights'
		vehicle_netcfg = 'data/vehicle-detector/vehicle-detection.cfg'
		vehicle_dataset = 'data/vehicle-detector/vehicle-detection.data'
		
		lp_threshold = .25

		lp_weights = 'data/lp-detector/lpd.weights'
		lp_netcfg = 'data/lp-detector/lpd.cfg'
		lp_dataset = 'data/lp-detector/lpd.data'
		
		ocr_threshold = .4

		ocr_weights = 'data/cr/cr.weights'
		ocr_netcfg = 'data/cr/cr.cfg'
		ocr_dataset = 'data/cr/cr.data'
		
		vehicle_net = dn.load_net(vehicle_netcfg.encode('utf-8'), vehicle_weights.encode('utf-8'), 0)
		vehicle_meta = dn.load_meta(vehicle_dataset.encode('utf-8'))
		loaded_models.append([vehicle_net, vehicle_meta, vehicle_threshold])
		
		lp_net = dn.load_net(lp_netcfg.encode('utf-8'), lp_weights.encode('utf-8'), 0)
		lp_meta = dn.load_meta(lp_dataset.encode('utf-8'))
		loaded_models.append([lp_net, lp_meta, lp_threshold])
		
		ocr_net = dn.load_net(ocr_netcfg.encode('utf-8'), ocr_weights.encode('utf-8'), 0)
		ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))
		loaded_models.append([ocr_net, ocr_meta, lp_threshold])
		
		return loaded_models
	
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	

if __name__ == '__main__':
	print('hello')
	load_system()
	

