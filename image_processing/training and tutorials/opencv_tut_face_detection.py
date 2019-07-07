import cv2

face_cascade = cv2.CascadeClassifier("C:\\Users\\piet.goempel\\AppData\\Local\\Programs\\Python\\Python37-32\Files\\haarcascade_frontalface_default.xml")

img = cv2.imread("C:\\Users\\piet.goempel\\Desktop\\Fotos\\IMG_7712.jpg", 1)

resized_img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(resized_img, (x,y), (x+w, y+h), (255,255,0), 3)

cv2.imshow("Facedetection", img)
cv2.waitKey(0)
