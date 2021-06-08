# https://google.github.io/mediapipe/solutions/pose.html

import cv2
import mediapipe as mp
import time


class poseDetector():
  def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.upBody = upBody
    self.smooth = smooth
    self.detectionCon = detectionCon
    self.trackCon = trackCon

    self.mpDraw = mp.solutions.drawing_utils
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                self.detectionCon, self.trackCon)

  def findPose(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)

    if self.results.pose_landmarks:
      if draw:
      # 각 관절의 포인트를 연결한다.
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

    return img


  def findPosition(self, img, draw=True):

    lmList = []
    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h, w, c = img.shape
      # print(id, lm, h, w, c)
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    return lmList




def main():
  cTime = 0
  pTime = 0
  cap = cv2.VideoCapture(0)
  # cap = cv2.VideoCapture('D:\\youtube\\댄싱9 S2이윤지.mp4')
  detector = poseDetector()
  
  while True:
    seccess, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
      # print(lmList)
      # 특정 포인트만 생상 변경
      cv2.circle(img, (lmList[19][1], lmList[19][2]), 15, (255, 0, 0), cv2.FILLED)
      cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime -pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(10)    

if __name__ == '__main__':
  main()