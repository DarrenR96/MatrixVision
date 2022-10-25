import cv2
from library import * 

vid = cv2.VideoCapture(0)
cv2.namedWindow('Matrix Vision')

while(True):
    ret, frame = vid.read()
    matrixFrame = YtoASCII(frame,t=85)
    height, width = frame.shape[0], frame.shape[1]
    frame = cv2.rectangle(frame,(0,600),(1280,720),(0,0,0),-1)
    cv2.putText(frame,'SigMedia Team, Electronic & Electrical Engineering', (60,675), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 1, 2)
    cv2.imshow('Camera Input', frame)
    cv2.imshow('Matrix Vision', matrixFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
