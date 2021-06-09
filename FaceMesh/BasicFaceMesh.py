import cv2
import mediapipe as mp
import time
import imutils

cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('D:\\youtube\\1.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=3)

# static_image_mode=False,
# max_num_faces=1,
# min_detection_confidence=0.5,
# min_tracking_confidence=0.5):

# 프레임 너비/높이, 초당 프레임 수 확인
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS)  # 또는 cap.get(5)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d (전체 프레임 수: %d)' % (width, height, fps, length))

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                  drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):  # 468 points
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # print(id, x, y)


    cTime = time.time()
    gTime = cTime - pTime
    fps = 1 / gTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    # print('cTime({}), gTime({}), pTime({})'.format(cTime, gTime, pTime))

    if gTime > 1./length:
        pTime = cTime
        img = imutils.resize(img, width=1024)
        cv2.imshow('image', img)
        cv2.waitKey(1)

cap.release()
cv2.destoryAllWindows()
