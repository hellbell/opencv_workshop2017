import cv2
import os

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name=raw_input('enter your name: ')
sampleNum=0

dataset_path = os.path.join('dataset',name)
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.5, minSize=(70,70))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #incrementing sample number 
        sampleNum = sampleNum + 1
        #saving the captured face in the dataset folder
        cv2.imwrite(dataset_path + "/" + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
    
    cv2.imshow('frame',img)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is more than 30
    elif sampleNum > 30:
        break
