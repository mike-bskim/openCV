import cv2
import mediapipe as mp
import time
import imutils


class faceMeshDetector():

    def __init__(self, staticMmode=False, maxFaces=3, minDectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMmode
        self.maxFaces = maxFaces
        self.minDectionCon = minDectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=3)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):  # 468 points
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                    # print(id, x, y)

                faces.append(face)

        return img, faces


def main() :

    cTime = 0
    pTime = 0
    faceCnt = 0
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('D:\\youtube\\1.mp4')
    detector = faceMeshDetector(maxFaces=1)

    # 프레임 너비/높이, 초당 프레임 수 확인
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 또는 cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 또는 cap.get(5)
    print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d (전체 프레임 수: %d)' % (width, height, fps, length))

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        cTime = time.time()
        gTime = cTime - pTime
        fps = 1 / gTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        # print('cTime({}), gTime({}), pTime({})'.format(cTime, gTime, pTime))

        if gTime > 1./length:
            if (len(faces)) != 0 and faceCnt != len(faces):
                faceCnt = len(faces)
                local_time = time.strftime('%Y-%m-%d %X / %Z', time.localtime(time.time()))
                # print('지금시간: {}, 사람수: {}'.format(local_time.decode('cp949').encode('utf8'), faceCnt))
                print('지금시간: {}, 사람수: {}'.format(local_time, faceCnt))
                print((faces[0]))
            pTime = cTime
            img = imutils.resize(img, width=1024)
            cv2.imshow('image', img)
            cv2.waitKey(1)

    cap.release()
    cv2.destoryAllWindows()


if __name__ == '__main__':
    main()
