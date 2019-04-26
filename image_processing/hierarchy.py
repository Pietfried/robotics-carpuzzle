import cv2
import numpy as np
from matplotlib import pyplot as plt

#Building images
normal_img = cv2.imread('images/image.jpg')
gray_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
edged_img = cv2.Canny(blurred_img, 50, 250)

contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

def show_images():
    cv2.imshow("normal_img", normal_img)
    cv2.waitKey(0)
    cv2.imshow("edged_img", edged_img)
    cv2.waitKey(0)

def show_contours_onebyone():
    i = 0
    for contour in contours:
        cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)
        print("contour number:", i)
        print(hierarchy[0][i])
        cv2.imshow("Contour image", normal_img)
        cv2.waitKey(0)
        i = i + 1

def find_board_contour(contours):
    for contour in contours:
        if (cv2.arcLength(contour, True) > 2200):
            print("Contour found. Contour is", contour)
            cv2.drawContours(normal_img, [contour], -1, (0, 255, 0), 2)


find_board_contour(contours)

cv2.imshow("contours", normal_img)
cv2.waitKey(0)