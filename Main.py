import numpy as np
import cv2
from collections import deque
 

Blower = np.array([100,60,60])
Bupper = np.array([140,255,255])

kernel = np.ones((5,5), np.uint8)

bpoints =[deque(maxlen= 512)]
ypoints =[deque(maxlen= 512)]
gpoints =[deque(maxlen= 512)]
rpoints = [deque(maxlen= 512)]
bindex = 0
yindex = 0
rindex = 0
gindex = 0

colors = [(255,200,0),(150,255,0),(0,0,200),(0,200,200)]
colorindex = 0

paintscreen = np.zeros((471,636,3))+255
paintscreen = cv2.rectangle(paintscreen,(40,1),(140,65),(0,0,0), 2)


paintscreen = cv2.rectangle(paintscreen,(160,1),(255,65),colors[0], -1)
paintscreen = cv2.rectangle(paintscreen,(275,1),(370,65),colors[1], -1)
paintscreen = cv2.rectangle(paintscreen,(390,1),(485,65),colors[2], -1)
paintscreen = cv2.rectangle(paintscreen,(505,1),(600,65),colors[3], -1)

cv2.putText(paintscreen, "Clear all",(49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintscreen, "Blue",(185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintscreen, "Green",(298,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintscreen, "Red",(420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintscreen, "Yellow",(520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

cv2.namedWindow("Webcam Paint",cv2.WINDOW_AUTOSIZE)

video = cv2.VideoCapture(0)

while True :
    success,frame = video.read()
    frame = cv2.flip(frame ,1)
    grayscaledframe = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    frame = cv2.rectangle(paintscreen,(40,1),(140,65),(0,0,0), 2)

    frame = cv2.rectangle(frame,(160,1),(255,65),colors[0], -1)
    frame = cv2.rectangle(frame,(275,1),(370,65),colors[1], -1)
    frame = cv2.rectangle(frame,(390,1),(485,65),colors[2], -1)
    frame = cv2.rectangle(frame,(505,1),(600,65),colors[3], -1)

    cv2.putText(frame, "Clear all",(49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Blue",(185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Green",(298,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Red",(420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Yellow",(520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    
    if not success :
        break

    bluemask = cv2.inRange(grayscaledframe,Blower,Bupper)
    bluemask = cv2.erode(bluemask,kernel,iterations=2)
    bluemask = cv2.morphologyEx(bluemask,cv2.MORPH_OPEN,kernel)
    bluemask = cv2.dliate(bluemask,kernel,iterations = 1)

    (cnts,_) = cv2.findContours(bluemask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center =  None

    if len(cnts>0) :
        cnt = sorted(cnts,key=cv2.contourArea,reverse=True)
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)

        cv2.circle(frame,(int(x), int(y)), int(radius), (0,255,255),2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01']/M['m00']))

    if center[1] <= 65 :
        if 40 <= center[0] <=140 :
            bpoints =[deque(maxlen= 512)]
            ypoints =[deque(maxlen= 512)]
            gpoints =[deque(maxlen= 512)]
            rpoints = [deque(maxlen= 512)]
            bindex = 0
            yindex = 0
            rindex = 0
            gindex = 0
            paintscreen[67:,:,:] = 255

        elif 160 <= center[0] <=255 :
            colorIndex = 0 
        elif 275 <= center[0] <=370 :
            colorIndex = 1 
        elif 390 <= center[0] <=485 :
            colorIndex = 2 
        elif 505 <= center[0] <=600:
            colorIndex = 3 

        else :
            if colorindex == 0 :
                bpoints[bindex].appendleft(center)
            elif colorindex == 1 :
                gpoints[gindex].appendleft(center)
            elif colorindex == 2 :
                rpoints[rindex].appendleft(center)    
            elif colorindex == 3 :
                ypoints[yindex].appendleft(center)            
        
    points = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)) :
        for j in range(len(points[i])) :
            for k in range(len(points[i][j])) :
                if points[i][j][k-1] is None or points[i][j][k] is None :
                    continue
                cv2.line(paintscreen,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
    cv2.imshow("Tracking",frame)
    cv2.imshow("Blankscreen",paintscreen)

    if cv2.waitKey(10) & 0xFF == ord('q') :
        break




video.release()
cv2.destroyAllWindows()

