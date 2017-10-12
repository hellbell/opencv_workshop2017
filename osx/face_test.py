import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
datapath = "dataset/"

people_list = os.listdir(datapath)

cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.5, minSize=(70,70))

    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print id, conf
        if(conf > 10):
            name = people_list[id]
        else:
            name = "Unknown"        
        cv2.putText(im, name, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
    cv2.imshow('img',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

