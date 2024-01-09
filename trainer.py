import os
import cv2
import numpy as np
from PIL import Image

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