import cv2

def make_test_data(prefix = 'temp/img_', cnt = 30):
	cap = cv2.VideoCapture(0)
	for i in range(cnt):
		ret, frame = cap.read()
		if not ret:
			print("ERROR RET")
			break
		cv2.imwrite(prefix + str(i) + '.png', frame)
		cv2.imshow("win", frame)
		cv2.waitKey(1)
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	#make_test_data(prefix="imgs/full/img_", cnt=100)
	#make_test_data(prefix="imgs/empty/img_", cnt=100)
	pass
