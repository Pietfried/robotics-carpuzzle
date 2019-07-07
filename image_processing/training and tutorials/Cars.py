from imutils import contours
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread("lol.jpg")
cv2.imshow("image", image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
edged = cv2.Canny(blurred, 50, 300)
cv2.imshow("edged", edged)
cv2.waitKey(0)
flag, thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

docCnt = []

periList = []
if len(cnts) > 0:

    # print cnts

    for c in cnts:

        peri = cv2.arcLength(c, True)
        periList.append(peri)
        if peri > 0:
            docCnt.append(c)

# print periList
periList.sort(reverse=True)
print periList[0:11]

cv2.drawContours(image, docCnt, -1, (0, 0, 0), 3)
cv2.imshow("Contours", image)
cv2.waitKey(0)
#sort contours and number them from left to right
