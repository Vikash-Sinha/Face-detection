# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
'''
from os import listdir
from os.path import isfile,join
'''
path='/home/vikash/Windows_files_and_folder/Vikash_Sinha/d_drive/python/open_cv/face_Detection/sample/'
onlyPath=[i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i))]
train_Data,lable=[],[]
for i,file in enumerate(onlyPath):
    img_path=path+onlyPath[i]
    img= cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    train_Data.append(np.asarray(img, np.uint8))
    lable.append(i)

lable=np.asarray(lable,np.int32)
model=cv2.face.LBPHFaceRecognizer_create ()
model.train(np.asarray(train_Data), np.asarray(lable))
print("ho gya= ",model,"    ",type(model))




path_classifier= cv2.CascadeClassifier('/home/vikash/anaconda3/lib/python3.8/site-packages/cv2/data/'+'haarcascade_frontalface_default.xml')
count=0
def face_dect(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=path_classifier.detectMultiScale(gray,1.5,5)
    if face == ():
        return img ,[]
    for(x,y,h,w) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),3)
        crp_img=img[y:y+h,x:x+w]
        
    return img,crp_img
    
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    img,face=face_dect(frame)
    try:
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face=cv2.flip(face,1)
        result=model.predict(face)
        if result[1] < 400:
            confi= int(100*(1-result[1]/300))
            disp=str(confi)+' % match'
            cv2.putText(frame, disp, (50,50), cv2.FONT_HERSHEY_COMPLEX , 1, (0,255,0),3)
            if confi > 75:
                disp="Vikash  Sinha Jii  Hai Ye "
                cv2.putText(frame, disp, (150,450), cv2.FONT_HERSHEY_COMPLEX , 1, (15, 193, 248),3)
                cv2.imshow('vikash', frame)
            else:
                disp="Koi Dusra H Ye "
                cv2.putText(frame, disp, (150,450), cv2.FONT_HERSHEY_COMPLEX , 1, (0,0,255),3)
                cv2.imshow('vikash', frame)
            
    except:
        disp="Face Nhi Deakh Rha H "
        cv2.putText(frame, disp, (150,450), cv2.FONT_HERSHEY_COMPLEX , 1, (255,0,231),3)
       
        cv2.imshow('vikash', frame)
        pass
 
    if cv2.waitKey(1)==ord('c'):
        break;

cap.release()
cv2.destroyAllWindows()



