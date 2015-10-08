from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import cv2
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt

def make_test_data():
	cap = cv2.VideoCapture(0)
	prefix = "imgs/full/img_"
	for i in range(60,100):
		ret, frame = cap.read()
		if not ret:
			print("ERROR RET")
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(prefix + str(i) + '.png', gray)
	cap.release()

prefix = 'imgs/'
one = prefix + 'full/'
zero = prefix + 'empty/'

ones = [one + f for f in os.listdir(one)]
tPath = ones[-1]
#ones = ones[:-1]
testIm = cv2.imread(tPath, cv2.IMREAD_GRAYSCALE)
zeros = [zero + f for f in os.listdir(zero)]
files = ones + zeros

labels = ([1] * len(ones)) + ([0] * len(zeros))
imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files]
for i in range(len(imgs)):
	imgs[i] = cv2.resize(imgs[i], (200,200))

X = [y.flatten() for y in imgs]
print(len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2)

n_classes = 300

t0 = time()
pca = RandomizedPCA(n_classes, whiten =True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Found PCA in %0.3f" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


#
#
#cap = cv2.VideoCapture(0)
#fFrame = cv2.imread('first.png', cv2.IMREAD_GRAYSCALE)
#fFrame = None
#
#arr = []
#pres = []
#
#num = 0
#while True:
#	ret, frame = cap.read()
#	if not ret: break
#
#	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#	gray = cv2.GaussianBlur(gray, (21,21), 0)
#	if fFrame is None:
#		fFrame = gray
#		cv2.imwrite('first.png', fFrame)
#		continue
#
#
#	sz = 6
#	arr.append(gray)
#	arr = arr[-sz:]
#	mn = arr[0]/sz
#	for i in range(1, len(arr)): mn = mn + arr[i]/sz
#	mn = mn.astype(np.uint8)
#	fFrame = mn
#	cv2.imshow('MN', mn)
#
#
#	delta = cv2.absdiff(fFrame, gray)
#	thresh = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)
#	#print(thresh[0])
#	#print(len(thresh))
#	thresh = thresh[1]
#	thresh = cv2.dilate(thresh, None, iterations = 3)
#	_, cnts, _ = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
#
#	for x,y,w,h in faces:
#		cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
#		print(w*h)
#
#	#if (len(cnts) > 0):
#	#	mx = max([cv2.contourArea(c) for c in cnts])
#	#	if mx > 100000:
#	#		print(num)
#	#		num+=1
#	#		print(mx)
#
#	cv2.imshow('frame', frame)
#	cv2.imshow('Thresh', thresh)
#
#	#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#	#faces = face_cascade.detectMultiScale(frame, 1.3, 5)
#	##frame = gray
#	#for x,y,w,h in faces:
#	#	cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
#	#denoice = cv2.fastNlMeansDenoisingColored(frame, hColor = 10, h = 10, templateWindowSize=7, searchWindowSize = 21)
#	#laplacian = cv2.Sobel(frame, cv2.CV_8U, 1, 0, ksize=5)
#	#cv2.imshow('Denoice', denoice)
#	##cv2.imshow('Original', frame)
#	#cv2.imshow('Laplacian', laplacian)
#	key = chr(cv2.waitKey(1) & 0xFF)
#	if  key == 'q':
#		break
#cap.release()
#cv2.destroyAllWindows()
