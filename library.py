import numpy as np 
import cv2 
import itertools
import random 

asciiScale = """$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """[::-1]
streak = """JUvunft-_+~<;:,"^`'."""
steak = [*streak]

streaks = []
frame = 0

def RGBtoY(rgb):
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return y

def everyIthFrame(frame,n=10):
    if ((frame % n == 0) and (frame >0)):
        frame = 0 
        return True


def YtoASCII(y, n=4, t=90, streakProb=0.2):
    global frame
    global streaks
    canvas = np.zeros_like(y).astype(np.uint8)
    y = RGBtoY(y)
    yShape, xShape = y.shape[0], y.shape[1]
    #Downsample Image by a factor of n
    y = cv2.resize(y, (0,0), fx=1/n,fy=1/n)

    #Perform ASCII mapping 
    y = cv2.Sobel(src=y, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    y = (y - y.min())/(y.max()-y.min())
    thres = np.percentile(y,t)
    y = np.where(y<thres, 0, y)

    y_ascii = np.rint(y*(len(asciiScale)-1)).astype(np.uint8).ravel().tolist()
    y_ascii = [asciiScale[x] for x in y_ascii]
    x_coords = list(range(0,xShape,n))
    y_coords = list(range(0,yShape,n))
    xy_coords = list(itertools.product(y_coords, x_coords))

    _prob = random.random()
    if _prob < streakProb:
        streaks.append([0,random.randint(0,xShape)])

    y = np.rint(255*y).astype(np.uint8)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
    for (_ascii, _xy) in zip(y_ascii, xy_coords):
        cv2.putText(canvas, _ascii, (_xy[1], _xy[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (0,255,0), 1, cv2.LINE_AA)

    for _xy in streaks:
        offset=0
        random.shuffle(steak)
        for itr, word in enumerate(steak):
            offset += 4
            itr += 1 
            colorFactor = 1/itr
            cv2.putText(canvas, word, (_xy[1],_xy[0]+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.15, (0,(colorFactor*255),0), 1, cv2.LINE_AA)
        
    canvas = cv2.rectangle(canvas,(0,600),(1280,720),(0,0,0),-1)
    cv2.putText(canvas,'SigMedia Team, Electronic & Electrical Engineering', (60,675), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 1, 2)
    streaks = [[x[0]+10,x[1]] for x in streaks]
    for streak in streaks:
        if streak[0] >= 1280:
            streaks.remove(streak)

    frame += 1
    return canvas