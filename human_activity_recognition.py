
# import the necessary packages
from collections import deque
import numpy as np
import imutils
import cv2

#Give the path of pretrained model and its classes

model = "activityrecognition.onnx"
classes =  open("Human_actions.txt","r")
# read the classses and store them in alist
CLASSES = classes.read().strip().split("\n")
print(CLASSES)

SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

#Deque (Doubly Ended Queue) in python is 
#implemented using the module “collections“.
#Deque is preferred over list in the cases 
#where we need quicker append and pop operations
# from both the ends of container, as deque provides 
#an O(1) time complexity for append and pop operations
# as compared to list which provides O(n) time complexity.
#limit the amount of items a deque can hold. 
frames = deque(maxlen=SAMPLE_DURATION)

# load the pretrained RestNet-34 kinetics pretrained model
print("loading human activity recognition model...")
net = cv2.dnn.readNet(model)

# capture the video frame either throuh video or webcam
print("accessing video stream...")
vs = cv2.VideoCapture(r"activities_trim.mp4")
#vs = cv2.VideoCapture(0)

# loop over frames from the video stream
while True:
    # read a frame from the video stream
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed then we've reached the end of
    # the video stream so break from the loop
    if not grabbed:
        print("no frame read from stream")
        break

    # resize the frame (to ensure faster processing) and add the
    # frame to our queue
    frame = imutils.resize(frame, width=800)
    frames.append(frame)

    # if our queue is not filled to the sample size, continue back to
    # the top of the loop and continue polling/processing frames
    if len(frames) < SAMPLE_DURATION:
        continue

    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0,
        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
        swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # pass the blob through the network to obtain our human activity
    # recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    # draw the predicted activity on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2)

    # display the frame to our screen
    cv2.imshow("Activity Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()

