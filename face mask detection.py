# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


x='deploy.prototxt.txt'
y='res10_300x300_ssd_iter_140000.caffemodel'

#Load Model
print("Loading model...................")
net = cv2.dnn.readNetFromCaffe(x,y)
model=load_model('mask_recog_ver2.h5')
# initialize the video stream to get the video frames
#width= int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
#height= int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (600,540))
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop the frams from the  VideoStream
while True :
    #Get the frams from the video stream and resize to 400 px
        frame = vs.read()
        frame = imutils.resize(frame,width=400)

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
        (h, w) = frame.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
        net.setInput(blob)
        detections = net.forward()

    # loop over the detections
        for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
            confidence = detections[0, 0, i, 2]
	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
            if confidence > 0.3:
		# compute the (x, y)-coordinates of the bounding box for
		# the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
		# ensure the bounding boxes fall within the dimensions of
		# the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        # extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
		# pass the face through the model to determine if the face
		# has a mask or not
                (mask, withoutMask) = model.predict(face)[0]# determine the class label and color we'll use to draw
		# the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # show the output frame
            #cv2.imwrite("Frame", frame)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
            if key == ord("q"):
             break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()