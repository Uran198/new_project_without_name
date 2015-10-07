import cv2
import numpy as np


cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for x,y,w,h in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
	cv2.imshow('frame', frame)
	k = cv2.waitKey(1)
	if  k & 0XFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()