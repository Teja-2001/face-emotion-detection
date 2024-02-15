# face-emotion-detection
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import cv2
from deepface import DeepFace

cap=cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
while True:
    _,frame=cap.read()
    faces=cascade.detectMultiScale(frame,1.1,3)
    predection=DeepFace.analyze(frame,actions=["emotion"],enforce_detection=False)
    for x,y,h,w in faces:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
    cv2.putText(frame,predection[0]["dominant_emotion"],(10,50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)
    cv2.imshow("window",frame)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
