import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

normal_img = cv.imread('images//empty_puzzle.jpg')
gray_img = cv.cvtColor(normal_img, cv.COLOR_BGR2GRAY)
median_blurred_img = cv.medianBlur(gray_img,5)
gaussian_blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)
edged = cv.Canny(gaussian_blurred_img, 50, 300)
flag, threshold = cv.threshold(edged, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

cnts, _	 = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
print("Detected contours", len(cnts))

contours_img = cv.drawContours(normal_img.copy(), cnts, -1, (0,0,255), 2)

contour_img = normal_img.copy()
for cnt in cnts:
	if (cv.arcLength(cnt, True) > 700):
		perimeter = cv.arcLength(cnt, True)
		print("Perimeter: ", perimeter)
		M = cv.moments(cnt)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
		cv.circle(contour_img, (cX, cY), 7, (255, 255, 255), -1)
		cv.putText(contour_img, str(perimeter),  (cX - 20, cY - 20),
		cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv.imshow("Single contour image", contour_img)	
		cv.waitKey(0)

#cv.imshow("Normal", normal_img)
# cv.imshow("Gray", gray_img)
# cv.imshow("Median Blurred", median_blurred_img)
# cv.imshow("Gaussian Blurred", gaussian_blurred_img)
#cv.imshow("Edged", edged)
#	cv.imshow("Threshold", threshold)

cv.destroyAllWindows()

# ret,th1 = cv.threshold(blurred,127,255,cv.THRESH_BINARY)
# th2 = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,11,4)

# # Find contours
# contours, hierachy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# print("Number of detected contours:" , len(contours))

# cv.drawContours(normal_img, contours, -1, (0,0,255), 2)

# cv.imshow("Contours", normal_img)
# cv.imshow("Adaptive Threshold", th2)
# cv.waitKey(0)
# cv.destroyAllWindows()