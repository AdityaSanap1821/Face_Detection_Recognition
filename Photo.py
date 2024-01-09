#importing Libraries
import cv2
from random import randrange as r
#dataset load
trainData=cv2.CascadeClassifier('Face.xml')

#choose a image
img = cv2.imread('Face1.jpg')

#conversion To gray scale
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect Faces
faceCoordinates = trainData.detectMultiScale(grayimg)
#[[856 107 295 295]]

x,y,w,h = faceCoordinates[0]

cv2.rectangle(img,(x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255)),2)

#Show the image
cv2.imshow('FACE DETECTION',img)

cv2.waitKey()
