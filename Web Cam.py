#importing Libraries
import cv2
from random import randrange as r

#dataset load
trainData=cv2.CascadeClassifier('Face.xml')

#choose a image
webcam = cv2.VideoCapture(0)

while True:

    success, img = webcam.read()

    #conversion To gray scale
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Detect Faces
    faceCoordinates = trainData.detectMultiScale(grayimg)

    for x,y,w,h in faceCoordinates:
        cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255),2))

    #Show the image
    cv2.imshow('FACE DETECTION',img)

    key = cv2.waitKey(1)
    if (key == 81 or key == 113):
        break

webcam.release()

