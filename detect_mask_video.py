# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from imutils.video import VideoStream

import numpy as np
import argparse
import imutils
import time
import cv2
import os

# BASED ON
# https://github.com/thegopieffect/computer_vision

##########################
#                        #
# Mask detecton function #
#                        #
##########################

def detectFaceAndMask(frame, faceNet, maskNet):
	# get frame dimensions
	(h, w) = frame.shape[:2]
	# create blob from image with mean subtraction (104.0, 177.0, 123.0)
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# detect faces in blob
	faceNet.setInput(blob)
	detections = faceNet.forward()
	
	# prepare data variables
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			
			# extract the face
			face = frame[startY:endY, startX:endX]
			# convert from BGR to RGB
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			# resize to 224x224 as this is how mask model is prepared
			face = cv2.resize(face, (224, 224))
			# add to array
			face = img_to_array(face)
			# preprocess
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# save face and bounding boxes to variables
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	
			# only check for mask if at least one face was detected
			if len(faces) > 0:
				pred = maskNet.predict(face)
				preds.append((pred[0][0], pred[0][1]))
	
	# return a 2-tuple of the face locations and detection probabilities
	return (locs, preds)



####################
#                  #
# Arguments Parser #
#                  #
####################



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face",
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model",
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float,
	default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load face model from disk
print("-I-: loading face detector model")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load mask model from disk
print("-I-: loading mask detector model")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("-I-: loading video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# setup variables
filterBW = False
filterSharpen = False
filterDenoise = False
drawContours = True



#############
#           #
# Main loop #
#           #
#############



# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# apply filters
	if filterBW:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
	
	if filterDenoise:
		frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 15)
	
	if filterSharpen:
		frameBlurred = cv2.GaussianBlur(frame, (0, 0), 9);
		frame = cv2.addWeighted(frame, 1.5, frameBlurred, -0.5, 0);
	
	# detect faces and check for mask
	(locs, preds) = detectFaceAndMask(frame, faceNet, maskNet)
	
	# loop over the detections
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, mask*255, withoutMask*255) if label == "Mask" else (0, mask*255, withoutMask*255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
		line_length = 3
		line_width = 1
		
		# draw contours
		if drawContours:
			edged = cv2.Canny(frame[startY:endY, startX:endX], 64, 192)
			for y in range(startY, endY):
				for x in range(startX, endX):		
					if edged[y - startY, x - startX]:
						#frame[y, x, 0] = frame[y, x, 0] - edged[y - startY, x - startX]
						frame[y, x] = color
		else:
			line_length = 10
			line_width = 2
				
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		
		# ellipse around face
		#centerEllipse = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
		#radiusEllipse = (int((endX - startX) / 2), int((endY - startY) / 2))
		#cv2.ellipse(frame, centerEllipse, radiusEllipse, 0, 0, 360, color)
		
		# corner lines on face bounding box
		cv2.rectangle(frame, (startX, startY), (startX, startY + line_length), color, line_width) # top-left
		cv2.rectangle(frame, (startX, startY), (startX + line_length, startY), color, line_width)
		
		cv2.rectangle(frame, (startX, endY), (startX, endY - line_length), color, line_width) # top-right
		cv2.rectangle(frame, (startX, endY), (startX + line_length, endY), color, line_width)
		
		cv2.rectangle(frame, (endX, startY), (endX, startY + line_length), color, line_width) # bottom-left
		cv2.rectangle(frame, (endX, startY), (endX - line_length, startY), color, line_width)
		
		cv2.rectangle(frame, (endX, endY), (endX, endY - line_length), color, line_width) # bottom-right
		cv2.rectangle(frame, (endX, endY), (endX - line_length, endY), color, line_width)
			
	# show the output frame
	cv2.imshow("Mask Detector", frame)
	key = cv2.waitKey(1) & 0xFF



#############
#           #
# Controlls #
#           #
#############



	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		
	# if the `b` key was pressed, enable BW filter
	if key == ord("b"):
		filterBW = not filterBW
	
	# if the `s` key was pressed, enable Sharpen filter
	if key == ord("s"):
		filterSharpen = not filterSharpen
		
	# if the `d` key was pressed, enable Denoising filter
	if key == ord("d"):
		filterDenoise = not filterDenoise
		
	# if the `c` key was pressed, enable Denoising filter
	if key == ord("c"):
		drawContours = not drawContours

# cleanup
cv2.destroyAllWindows()
vs.stop()