import cv2

ratio = 3
kernel_size = 3
sz = 8
thresh = 7

def edges(img):
	#dst = cv2.blur(img, (sz,sz))
	dst = cv2.bilateralFilter(img, sz, 75, 75)
	dst = cv2.Canny(dst, thresh, thresh*ratio, kernel_size)
	return dst

if __name__ == "__main__":
	em = cv2.imread('imgs/empty/img_0.png', cv2.IMREAD_GRAYSCALE)
	fl = cv2.imread('imgs/full/img_0.png', cv2.IMREAD_GRAYSCALE)

	cap = cv2.VideoCapture(0)
	em = edges(em)
	fl = edges(fl)

	while True:
		#ret, fl = cap.read()
		#fl = edges(fl)

		cv2.imshow("Win1", em)
		cv2.imshow("Win2", fl)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break
		
	cap.release()
	cv2.destroyAllWindows()

