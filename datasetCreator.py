import cv2
import numpy as numpy
from random import randrange as r

faceDetect = cv2.CascadeClassifier('Face.xml')
cam = cv2.VideoCapture(0)

id = input("Enter User Id: ")

sampleNum = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite(r"dataSet\User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255),2))
        cv2.waitKey(100)
    cv2.imshow("FACE",img)
    cv2.waitKey(1)
    if (sampleNum > 20):
        break
cam.release()
cv2.destroyAllWindows()