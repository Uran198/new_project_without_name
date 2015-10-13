import cv2
import numpy as np

def do_it(im1, im2):
	orb = cv2.ORB_create()
	m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
	kp1, desc1 = orb.detectAndCompute(im1, None)
	kp2, desc2 = orb.detectAndCompute(im2, None)
	matches = m.match(desc1, desc2)
	matches = sorted(matches, key=lambda x:x.distance)
	res = cv2.drawMatches(im1, kp1, im2, kp2, matches[:20], None, flags=2)
	cv2.imshow('res', res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

im1 = cv2.imread('imgs/empty/img_1.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('imgs/full/img_1.png', cv2.IMREAD_GRAYSCALE)

do_it(im1, im2)
