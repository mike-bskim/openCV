import cv2
import mediapipe as mp
import time

print('openCV version: ', cv2.__version__)
cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imgRGB)
  # print(results.multi_hand_landmarks)

  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

  cTime = time.time()

  cv2.imshow('Image', img)
  cv2.waitKey(1)
