import cv2,os
import glob
import numpy as np
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
datapath = 'dataset'

people_list = os.listdir(datapath)
images = []
labels = []

label = 0
for name in people_list:
    image_paths = glob.glob(os.path.join(datapath, name, '*.jpg'))                
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, 'uint8')
        images.append(image_np)
        labels.append(label)
        print label
        cv2.imshow("Adding faces to training set...", image_np)
        cv2.waitKey(30)
    label = label + 1

recognizer.train(images, np.array(labels))
recognizer.write('face_recognizer.yml')
cv2.destroyAllWindows()

