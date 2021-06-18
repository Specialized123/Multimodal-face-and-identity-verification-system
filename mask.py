import time
import cv2 as cv
import numpy as np

# 摄像头
cap = cv.VideoCapture(0)
# 创建检测器
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# 加载 model
face_recognizer.read('F:/AI Study/FaceRecognition-main/FaceRecognition-main/fm_model.xml')
nose_cascade=cv.CascadeClassifier('F:/AI Study/opencv/pack/share/OpenCV_xml/haarcascade_mcs_nose.xml')
face_cascade=cv.CascadeClassifier('E:/VS2017/Anaconda3_64/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade=cv.CascadeClassifier('E:/VS2017/Anaconda3_64/Lib/site-packages/cv2/data/haarcascade_eye.xml')
mouth_cascade=cv.CascadeClassifier('F:/AI Study/opencv/pack/share/OpenCV_xml/haarcascade_mcs_mouth.xml')

# 画框
def draw_rectangle(img, rect):
    for x,y,w,h in rect:
        cv.rectangle(img, (x, y), (x+w, y+h), (128, 128, 0), 2)

# 写文字
def draw_text(img, text, rect):
    for x,y,w,h in rect:
        cv.putText(img, text, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

def cut_face(image):
    results=face_cascade.detectMultiScale(gray,1.1,5)
    if results != ():
        for x,y,w,h in results:
            return results,gray[y:y+h, x:x+w]
    else:
        return [[0, 0, 0, 0]], None
    
# 口罩判别
def Mask_detection(image,rect):
    nose = nose_cascade.detectMultiScale(gray,1.3,5)
    eye = eye_cascade.detectMultiScale(gray,1.1,5)
    mouth = mouth_cascade.detectMultiScale(gray,1.1,5)
    #判断是否带口罩
    if nose!=() and mouth!=() and rect[0][0]+rect[0][2] != 0:
        return "NO"
    elif nose==() and mouth==() and rect[0][0]+rect[0][2] != 0: 
        return "YES"
    return '**'

# 此函数识别传递的图像中的人物并在检测到的脸部周围绘制一个矩形及其名称
# 人脸识别 预测
def predict(image):
    if image is not None:
        #预测人脸
        results = face_recognizer.predict(image)
        print(results[1])
        #置信度阈值
        if results[1] < 50:
            label_text = face_recognizer.getLabelInfo(results[0])
        else:
            label_text = 'stranger'
        return label_text
    else:
    #print eye
        return 'not whole face'

train_faces = []
train_labels = []
num = -1
train_face_num = 15
while( cap.isOpened() ):
    # USB摄像头工作时,读取一帧图像
    ret, image = cap.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rect,cut_img = cut_face(gray)
        
    # 检测 返回是否带口罩
    label_mask = Mask_detection(gray,rect)
    # 输出人脸框 及相关的文字信息
    draw_rectangle(image, rect)
    draw_text(image, "mask:"+label_mask, [[10,40,10,10]])
    
    key = cv.waitKey(1) & 0xFF


            
# 释放资源和关闭窗口
cap.release()
cv.destroyAllWindows()