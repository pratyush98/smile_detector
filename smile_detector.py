import cv2
import numpy as np

smile_cascade = cv2.CascadeClassifier("smile.xml") 
face_cascade = cv2.CascadeClassifier("frontalface_default.xml")

#reading the video..
cap = cv2.VideoCapture('output.avi')

#getting the fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
ctr1=0

while(cap.isOpened()):

    ret, img = cap.read()
    if ret== True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #operating on the faces identified by the face-haarcascade..
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

            #extracting only the face for detection of smile..
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            smile = smile_cascade.detectMultiScale(roi_gray,9.4,1)

            #counting the frames where smile is detected..
            if len(smile) != 0:
                ctr1+=1
            for (ex, ey, ew, eh) in smile:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
                
        
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()



print("the time he was smiling was %f secs" %(float(ctr1)/fps))

