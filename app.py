import cv2
from model import transform
from model import get_features
from sklearn.linear_model import LogisticRegression
import pandas as pd
from time import time

df = pd.read_csv("data.csv", header=None)
X = transform(df[0])
y = df[1]

t0 = time()
clf = LogisticRegression()
clf.fit(X,y)

print("Trained in", time()-t0)

cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		print("Not ret: Something went wrong!")
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	features = get_features(gray)
	pred = clf.predict([features])
	print(pred)
	cv2.imshow("win", frame)
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cap.release()
