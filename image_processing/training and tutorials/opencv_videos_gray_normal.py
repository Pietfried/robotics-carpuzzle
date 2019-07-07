import cv2

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    normal_img = frame
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing normal", normal_img)
    cv2.imshow("Capturing gray", gray_img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
