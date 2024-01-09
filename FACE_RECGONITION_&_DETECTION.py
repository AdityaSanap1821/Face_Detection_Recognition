# importing libraries
import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
from random import randrange as r


window = tk.Tk()
window.title("FACE RECGONITION AND DETECTION")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
    window, text="FACE RECGONITION & DETECTION SYSTEM",
    bg="yellow", fg="red", width=55,
    height=2, font=('Times New Roman', 40, 'bold italic'))
message.place(x=0, y=20)

lbl = tk.Label(window, text="FACE DETECTION ",
               width=20, height=2, fg="red",
               bg="white", font=('Times New Roman', 25, ' bold italic '))
lbl.place(x=250, y=345)

lbl3 = tk.Label(window, text="FACE RECGONITION ",
               width=20, height=2, fg="red",
               bg="white", font=('Times New Roman', 25, ' bold italic '))
lbl3.place(x=250, y=520)

lbl1 = tk.Label(window, text="ID.",
               width=20, height=2, fg="red",
               bg="white", font=('Times New Roman', 15, ' bold '))
lbl1.place(x=400, y=200)
 
txt = tk.Entry(window,
               width=20, bg="white",
               fg="green", font=('Times New Roman', 15, ' bold '))
txt.place(x=700, y=215)
 
lbl2 = tk.Label(window, text="Name",
                width=20, fg="red", bg="white",
                height=2, font=('Times New Roman', 15, ' bold '))
lbl2.place(x=400, y=300)
 
txt2 = tk.Entry(window, width=20,
                bg="white", fg="green",
                font=('Times New Roman', 15, ' bold '))
txt2.place(x=700, y=315)
 

def Photo():
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

def Video():
    #dataset load
    trainData=cv2.CascadeClassifier('Face.xml')

    #choose a image
    cam = cv2.VideoCapture('Clip.mp4')

    while True:

        success, img = cam.read()

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
    cam.release()


def Webcam():
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

def Create_DataSet():

        faceDetect = cv2.CascadeClassifier('Face.xml')
        cam = cv2.VideoCapture(0)

        Id = (txt.get())
        name = (txt2.get())
        sampleNum = 0

        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray,1.3,5)
            for (x, y, w, h) in faces:
                sampleNum = sampleNum + 1
                cv2.imwrite(r"dataSet\ "+name + "."+Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255),2))
                cv2.waitKey(100)
            cv2.imshow("FACE",img)
            cv2.waitKey(1)
            if (sampleNum > 20):
                break
        cam.release()
        cv2.destroyAllWindows()

def trainImg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = r'dataSet'

    def getImageWithId(path):
        ImagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for ImagePath in ImagePaths:
            faceImg = Image.open(ImagePath).convert('L')
            faceNp = np.array(faceImg,'uint8')
            ID = int(os.path.split(ImagePath)[-1].split('.')[1])
            faces.append(faceNp)
            print(ID)
            IDs.append(ID)
            cv2.imshow("TRAINING",faceNp)
            cv2.waitKey(10)
        return IDs, faces

    IDs, faces = getImageWithId(path)
    recognizer.train(faces, np.array(IDs))
    recognizer.save(r'recognizer/trainingData.yml')
    cv2.destroyAllWindows()


def detector():
    faceDetect = cv2.CascadeClassifier('Face.xml')
    cam = cv2.VideoCapture(0)
    Id = (txt.get())
    name = (txt2.get())
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(r"C:\KJ SOMAIYA\KJ SOMAIYA SEM 3\PYTHON\Python Mini Project\recognizer\trainingData.yml")

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)

    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(r(0,255),r(0,255),r(0,255),2))
            id, conf = rec.predict(gray[y:y+h,x:x+w])
            cv2.putText(img,str(name),(x,y+h),fontFace,fontScale,fontColor)
        cv2.imshow("FACE",img)
        if(cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()



takephoto = tk.Button(window, text="TAKE IMAGE",
                    command=Create_DataSet, fg="white", bg="green",
                    width=20, height=3,
                    font=('Times New Roman', 15, ' bold '))
takephoto.place(x=200, y=600)

trainImg = tk.Button(window, text="TRAIN IMAGE",
                     command=trainImg, fg="white", bg="green",
                     width=20, height=3,
                     font=('Times New Roman', 15, ' bold '))
trainImg.place(x=500, y=600)

trackImg = tk.Button(window, text="RECOGNIZE",
                     command=detector, fg="white", bg="green",
                     width=20, height=3,
                     font=('Times New Roman', 15, ' bold '))
trackImg.place(x=800, y=600)

photo = tk.Button(window, text="PHOTO",
                    command=Photo, fg="white", bg="black",
                    width=20, height=3,
                    fon=('times', 15, ' bold '))
photo.place(x=370, y=420)

video = tk.Button(window, text="VIDEO",
                     command=Video,fg="white", bg="black",
                     width=20, height=3,
                     font=('times', 15, ' bold '))
video.place(x=670, y=420)

webcam = tk.Button(window, text="WEB CAMERA",
                     command=Webcam,fg="white", bg="black",
                     width=20, height=3,
                     font=('times', 15, ' bold '))
webcam.place(x=970, y=420)

quitWindow = tk.Button(window, text="Quit",
                       command=window.destroy, fg="white", bg="green",
                       width=20, height=3,
                       font=('Times New Roman', 15, ' bold '))
quitWindow.place(x=1100, y=600)

window.mainloop()