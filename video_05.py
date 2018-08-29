import cv2


def nothing(emp):
    pass


video = '01.mp4'

cv2.namedWindow('video')

cap = cv2.VideoCapture(video)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


loop_flag = 0

pos = 0

cv2.createTrackbar('time', 'video', 0, frames, nothing)

while 1:

    if loop_flag == pos:

        loop_flag = loop_flag + 1

        cv2.setTrackbarPos('time', 'video', loop_flag)

    else:

        pos = cv2.getTrackbarPos('time', 'video')

        loop_flag = pos

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    ret, img = cap.read()
    img =  cv2.flip(img, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('video', gray)
    k = cv2.waitKey(1)

    if k & loop_flag == frames:
        break
    if k == ord('w'):
        cv2.waitKey(0)