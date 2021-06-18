import time
import cv2 as cv

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#name = face_recognizer.getLabelInfo(label)
face_recognizer.write('fm_model.xml')