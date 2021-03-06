# https://google.github.io/mediapipe/solutions/hands.html

import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myLists = os.listdir(folderPath)
print(myLists)
overlayList = []
for myList in myLists:
    fullPath = f'{folderPath}/{myList}'
    image = cv2.imread(fullPath)
    print(fullPath)
    overlayList.append(image)

print(len(overlayList))
pTime = 0
# hand tracking instance
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
tmp = -1

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []

        # Thumb, end of thumb is righter than 2nd point ==> thumb is opened
        # X axis, thumb is moving way of X axis direction
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):  # except Thumb
            # end of finger is higher than 3rd point ==> finger is opened
            # Y axis
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)

        if tmp != totalFingers:
            tmp = totalFingers
            print(time.strftime('%Y-%m-%d %X', time.localtime(time.time())), totalFingers)

        # 숫자 이미지를 cap 이미지에 합치기
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 15)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
