import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
import imutils


confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 288  # 608     # Width of network's input image
inpHeight = 288  # 608     # Height of network's input image

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

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
    

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "data/vehicle-detector/vehicle-detection.cfg"
modelWeights = "data/vehicle-detector/vehicle-detection.weights"

net1 = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(left, top, right, bottom,label):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
        1.5*labelSize[0]), top + baseLine), (255, 0, 255), cv.FILLED)
    cv.putText(frame, label, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)


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






outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

cut_neg=lambda a: (abs(a)+a)//2

overall_time = time.time()
total_frames = 0

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    #frame = imutils.resize(frame,height = 960,width=540)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    start = time.time()
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net1.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net1.forward(getOutputsNames(net1))

    # Remove the bounding boxes with low confidence

	#get coordinates of predicted bounding boxes
    cars = postprocess(frame, outs,1)
    
    print("cars:",cars)
    

    blob = None

    if len(cars):
        for car in cars:
            drawPred(car[0],car[1],car[0]+car[2],car[1]+car[3],"car")
            Icar = frame[cut_neg(car[1]):cut_neg(car[1])+cut_neg(car[3]), cut_neg(car[0]):cut_neg(car[0])+cut_neg(car[2])]
            blob =  cv.dnn.blobFromImage(Icar, 1/255, (192, 192), [0, 0, 0], 1, crop=False)
            net2.setInput(blob)
            outs = net2.forward(getOutputsNames(net2))
            lps = postprocess(Icar,outs,2)
            print("lps", lps)
            
            if len(lps):
                for lp in lps:
                    Ilp = Icar[cut_neg(lp[1]):cut_neg(lp[1])+cut_neg(lp[3]), cut_neg(lp[0]):cut_neg(lp[0])+cut_neg(lp[2])]
                    blob =  cv.dnn.blobFromImage(Ilp, 1/255, (224, 64), [0, 0, 0], 1, crop=False)
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
                    drawPred(offsetx,offsety,offsetx+lp[2],offsety+lp[3],lp_str)
    cv.imshow('frame',frame)
    total_frames += 1
    duration = time.time() - start
    print("FPS : {}".format(1/duration))
    
        # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))
print("overall Fps :", total_frames/(time.time() - overall_time) )
   
