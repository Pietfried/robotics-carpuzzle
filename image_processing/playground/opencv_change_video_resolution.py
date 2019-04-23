import cv2

# methods to change resolution
# recommendation is to only do downscale

def make_1080p():
    video.set(3,1920)
    video.set(4, 1080)

def make_720p():
    video.set(3,1280)
    video.set(4, 720)

def make_480p():
    video.set(3,640)
    video.set(4, 480)

def change_res(width, height):
    video.set(3,width)
    video.set(4, height)

def rescale_frame(frame, percent=75):
    scale_percent = percent
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

# Capturing the video

video = cv2.VideoCapture(0)

# make_480p()
# make_720p()
# make_1080p()
#change_res(400, 600)

while True:
    check, frame = video.read()
    
    frame1 = rescale_frame(frame)
    frame2 = rescale_frame(frame, percent=100)

    cv2.imshow("Capturing 75 percent", frame1)
    cv2.imshow("Capturing 100 percent", frame2)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
