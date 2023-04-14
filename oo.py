import cv2
import numpy as np
import face_recognition

imgmark = face_recognition.load_image_file('imagess/mark.jpg')
imgmark = cv2.cvtColor(imgmark,cv2.COLOR_BGR2RGB)

imghelmy = face_recognition.load_image_file('imagess/helmy.jpg')
imghelmy = cv2.cvtColor(imghelmy,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgmark)[0]
encoding = face_recognition.face_encodings(imgmark)[0]
cv2.rectangle(imgmark,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc =face_recognition.face_locations(imghelmy)[0]
encoding = face_recognition.face_encodings(imghelmy)[0]
cv2.rectangle(imghelmy,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)


cv2.imshow('mark',imgmark)
cv2.imshow('helmy',imghelmy)
cv2.waitKey(0)

        