import cv2

img = cv2.imread('imgs/full/img_0.png', cv2.IMREAD_GRAYSCALE)
ratio = 3
kernel_size = 3
cap = cv2.VideoCapture(0)
sz = 3
g_thresh = 0

def draw():
	#dst = cv2.blur(img, (sz,sz))
	dst = cv2.bilateralFilter(img, sz, 75, 75)
	dst = cv2.Canny(dst, g_thresh, g_thresh*ratio, kernel_size)
	cv2.imshow("Win", dst)

def cannyThreshold(thresh,*args):
	global g_thresh
	g_thresh = thresh
	draw()

def blur_change(blur):
	global sz
	sz = max(1,blur)
	draw()

cv2.namedWindow("Win")
cv2.createTrackbar("Thresh", "Win", 0, 100, cannyThreshold)
cv2.createTrackbar("Blur size", "Win", 0, 100, blur_change)
cannyThreshold(0,0)
cv2.waitKey(0)
cap.release()
