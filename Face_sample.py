# -*- coding: utf-8 -*-
import cv2
import time
path_classifier= cv2.CascadeClassifier('/home/vikash/anaconda3/lib/python3.8/site-packages/cv2/data/'+'haarcascade_frontalface_default.xml')
count=0
def face_rec(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=path_classifier.detectMultiScale(gray,1.5,3)
    if face is():
        return None
    for(x,y,h,w) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),5)
        crp_img=img[y:y+h,x:x+w]
        
    return crp_img
    
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()   
    try:
        frame=cv2.flip(frame,1)
        if face_rec(frame) is not None:
            count+=1
            face=cv2.resize(face_rec(frame),(200,200))
            face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.putText(frame, str(count), (150,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
            cv2.imwrite('/home/vikash/Windows_files_and_folder/Vikash_Sinha/d_drive/python/open_cv/face_Detection/sample/'+str(count)+'.jpg', face)
            
            
        else:
            print("not found")
            pass
        if count==140:
            cv2.putText(frame,'face reading complited' , (150,100), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            cv2.imshow('vikash',frame)
            print("ho rha h")
            cv2.waitKey(1)
            time.sleep(2)
            break
        
        cv2.imshow('vikash',frame)
    except:
        pass
   
 
    if cv2.waitKey(1)==ord('c'):
        break;

cap.release()
cv2.destroyAllWindows()