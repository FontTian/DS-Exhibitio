# -*- coding:utf-8 -*-
__author__ = 'FontTian'
__Date__ = '2017/5/6'
import cv2


def generate():
    # 加载已经训练好的haar
    #
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load('./haarcascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while (True):
        ret, frame = camera.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                f = cv2.resize(gray[y:y + h, x:x + w], (300, 300))

                cv2.imwrite('./data/at/tfs/%s.pgm' % str(count), f)
                count += 1

            cv2.imshow('camera', frame)
        else:
            print("no ret")

        if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


def readVideo():
    # cap = cv2.VideoCapture("fenlei.mp4")
    # cap = cv2.VideoCapture("lisaru.mp4")
    cap = cv2.VideoCapture("susuan.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        # 这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
        if ret == True:
            # gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("frame" , gray)
            cv2.imshow("frame", frame)
        else:
            break
        # 因为视频是10帧每秒，因此每一帧等待100ms
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate()
