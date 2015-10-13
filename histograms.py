import cv2
import matplotlib.pyplot as plt

im1 = cv2.imread("imgs/full/img_0.png")
im2 = cv2.imread("imgs/empty/img_0.png")
ims = [im1, im2]

ims = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in ims]

color = list("b")
plt.xlim([0,256])
for j,im in enumerate(ims):
	plt.figure(j)
	for i,col in enumerate(color):
		hist = cv2.calcHist(im, [i], None, [256], [0,256])
		plt.plot(hist,color=col)

plt.show()
