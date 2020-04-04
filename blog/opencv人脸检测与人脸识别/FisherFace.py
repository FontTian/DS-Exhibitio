import cv2
import numpy as np
images = []

images.append(cv2.imread("data/at/fonttian/15.pgm",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("data/at/fonttian/17.pgm",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("data/at/fonttian/55.pgm",cv2.IMREAD_GRAYSCALE))

images.append(cv2.imread("data/at/zjz/15.pgm",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("data/at/zjz/17.pgm",cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread("data/at/zjz/60.pgm",cv2.IMREAD_GRAYSCALE))

# 创建标签
labels = [0,0,0,1,1,1]

# 创建测试文件路径
imgPath = "data/at/fonttian/570.pgm"

# 传入测试图片路径，识别器，原始人脸数据库，人脸数据库标签
def showConfidence(imgPath,recognizer,images,labels):
    # 训练识别器
    recognizer.train(images,np.array(labels))
    # 加载测试图片
    predict_image=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    # 预测并打印结果
    labels,confidence = recognizer.predict(predict_image)
    print("label=",labels)
    print("conficence=",confidence)

# EigenFace(PCA，5000以下判断可靠）
recognizer = cv2.face.EigenFaceRecognizer_create()
showConfidence(imgPath,recognizer,images,labels)
# LBPH（局部二值模式直方图，0完全匹配，50以下可接受，80不可靠）
recognizer = cv2.face.LBPHFaceRecognizer_create()
showConfidence(imgPath,recognizer,images,labels)
# Fisher(线判别分析 ， 5000以下判断为可靠）
recognizer = cv2.face.FisherFaceRecognizer_create()
showConfidence(imgPath,recognizer,images,labels)



# FisherFaceRecognizer_create