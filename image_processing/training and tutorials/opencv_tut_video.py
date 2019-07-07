import cv2, time

face_cascade = cv2.CascadeClassifier("C:\\Users\\piet.goempel\\AppData\\Local\\Programs\\Python\\Python37-32\Files\\haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

a = 1

while True:
    a = a + 1
    check, frame = video.read()
    print(frame)

    #Converting to gray img
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting the face
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)

    for x,y,w,h in faces:
        img = cv2.rectangle(gray_img, (x,y), (x+w, y+h), (255,255,0), 3)

    #Displaying the image with face detection
    cv2.imshow("Capturing", img)

    key = cv2.waitKey(1)

    #Closing window on pressing q
    if key == ord('q'):
        break

print(a) #This will print the number of frames

video.release()
cv2.destroyAllWindows()
