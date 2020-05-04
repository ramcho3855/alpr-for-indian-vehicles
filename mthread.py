from queue import Queue
from threading import Thread, Event
from time import sleep,time
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
from time import clock
import multiprocessing as mp

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

cap = ''

cut_neg=lambda a: (abs(a)+a)//2
# Load names of classes
vclassesFile = "data/vehicle-detector/vehicle-detection.names"
lclassesFile = "data/lp-detector/lpd.names"
crclassesFile = 'data/cr/cr.names'

vclasses = None
lclasses = None
crclasses = None
with open(vclassesFile, 'rt') as f:
    vclasses = f.read().rstrip('\n').split('\n')
    
with open(lclassesFile, 'rt') as f:
    lclasses = f.read().rstrip('\n').split('\n')
    
with open(crclassesFile, 'rt') as f:
    crclasses = f.read().rstrip('\n').split('\n')


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame,left, top, right, bottom,label):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

    if len(label):
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
            1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
    return frame


def postprocess(frame, outs,stage,offset=None):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if stage == 3:
        output = {}
    else:
        output = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if stage == 3:
            output[left] = crclasses[classIds[i]]        
        else:
            output.append([left,top,width,height])
    return output


def show(q2,s,outputFile):
	global cap


	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
	while True:
		start = time.time()
		frame = q2.get()
		if frame == 'END':
			print("All done ..........................................")
			return
#		cv.imshow("frame",frame)
		print("FPS : {}".format(1/(time.time()-start)))
		vid_writer.write(frame.astype(np.uint8))
		if cv.waitKey(1) & 0xFF == ord('q'):
			s.set()
			return
		

def lp(q1,q2,s):
	modelConfiguration = "data/lp-detector/lpd.cfg"
	modelWeights = "data/lp-detector/lpd.weights"
	net2 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net2.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net2.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
	modelConfiguration = "data/cr/cr.cfg"
	modelWeights = "data/cr/cr.weights"

	net3 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net3.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net3.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

	while True and not s.is_set():
		data = q1.get()
		if data == 'END':
			print("LP Recognition done..........................................")
			q2.put("END")
			return
		frame = data[0]
		cars = data[1]
		if True:
			for car in cars:
				cv.rectangle(frame, (car[0], car[1]), (car[0]+car[2], car[1]+car[3]), (0, 255, 255), 3)
				Icar = frame[cut_neg(car[1]):cut_neg(car[1])+cut_neg(car[3]), cut_neg(car[0]):cut_neg(car[0])+cut_neg(car[2])]
				blob =  cv.dnn.blobFromImage(Icar, 1/255, (416, 416), [0, 0, 0], 1, crop=False)
				net2.setInput(blob)
				outs = net2.forward(getOutputsNames(net2))
				lps = postprocess(Icar,outs,2)
				print("lps", lps)
				if len(lps):
					for lp in lps:
						Ilp = Icar[cut_neg(lp[1]):cut_neg(lp[1])+cut_neg(lp[3]), cut_neg(lp[0]):cut_neg(lp[0])+cut_neg(lp[2])]
						blob =  cv.dnn.blobFromImage(Ilp, 1/255, (256, 96), [0, 0, 0], 1, crop=False)
						net3.setInput(blob)
						outs = net3.forward(getOutputsNames(net3))
						offsetx = lp[0] + car[0]
						offsety = lp[1] + car[1]
						offset = [offsetx,offsety]
						chars = postprocess(Ilp,outs,3,offset)
						lp_str = ''
						for char in sorted(chars):
							lp_str += chars[char]
						print(lp_str)
						frame = drawPred(frame,offsetx,offsety,offsetx+lp[2],offsety+lp[3],lp_str)
			q2.put(frame)
				


def vehicle(q,q1,s):

	modelConfiguration = "data/vehicle-detector/vehicle-detection.cfg"
	modelWeights = "data/vehicle-detector/vehicle-detection.weights"

	net1 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # Create a 4D blob from a frame.
	while not s.is_set():
		frame = q.get()
		if frame == 'END':
			print("Vehicle detection done...............................")
			q1.put('END')
			return

		#print(frame)
		blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
		net1.setInput(blob)

    # Runs the forward pass to get output of the output layers
		outs = net1.forward(getOutputsNames(net1))

    # Remove the bounding boxes with low confidence

	#get coordinates of predicted bounding boxes
		cars = postprocess(frame, outs,1)
		print("cars:",cars)
		vehicle = [frame,cars]
		q1.put(vehicle)

#		if data == -1:
#			break
#		else:
#			print("received : {}".format(data) )

def sender(q,s,video):
	
	global cap
	
	cap = cv.VideoCapture(video)
	wf = 30
	i = 1
	while(cap.isOpened()) and not s.is_set():
		ret, frame = cap.read()
		if not ret:
			print("done reading frames....")
			q.put("END",block=False)
			break
#		if(i%(wf//5) == 0):
		q.put(frame,block=False)
#		i+=1


	
q = Queue(maxsize=0)
q1 = Queue(maxsize=0)
q2 = Queue(maxsize=0)
q3 = Queue(maxsize=0)
	
if __name__ == "__main__":

	parser = argparse.ArgumentParser(
    	description='ALPR')
	parser.add_argument('--video', help='Path to video file.')
	args = parser.parse_args()

	if (args.video):
    	# Open the video file
		if not os.path.isfile(args.video):
		    print("Input video file ", args.video, " doesn't exist")
		    sys.exit(1)
		input_path = args.video
		outputFile = args.video[:-4]+'alpr_result.avi'
	else:
		input_path = 0
		outputFile = 'webcam.avi'
    	


	stop_event= Event()
	a = Thread(target=sender, args=(q,stop_event,input_path))
	b = Thread(target=vehicle, args=(q,q1,stop_event))
	c = Thread(target=lp, args=(q1,q2,stop_event))
	d = Thread(target=show,args=(q2,stop_event,outputFile))
	a.start()
	sleep(1)
	start = time.time()
	b.start()
	c.start()
	d.start()
	a.join()
	b.join()
	c.join()
	d.join()
	print("FPS final: {}".format(113/(time.time()-start)) )
	
