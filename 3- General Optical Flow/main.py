
from opticalFlowLK import OpticalFlowLK
from util import *
import numpy as np
import cv2


cap = cv2.VideoCapture('slow_traffic_small.mp4')

#For keeping the last frame
last_grey = None
last_frame = None
last_corners = []
status = []

winsize = [25, 25]

while (cap.isOpened()):
    ret, frame = cap.read()
    scale = 0.8
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    

    #Get the corners from the current frame
    corners = cv2.goodFeaturesToTrack(grey, 100, 0.01, 10, mask = None, blockSize = 3, useHarrisDetector = True, k = 0.04)

    print('Number of corners detected: ', corners.shape[0])

    #Get the subpixel locations for corners
    subPixWinSize = (10, 10)
    zeroZone = (-1, -1)
    termcrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)

    cv2.cornerSubPix(grey, corners, subPixWinSize,zeroZone, termcrit)

    corners = np.array(corners)
    corners = np.squeeze(corners, axis=1)

    flowPoints = 0
    if len(last_corners) != 0:

        of = OpticalFlowLK(winsize, 0.03, 20)
        points, status = of.compute(last_grey, grey, np.copy(last_corners))
        

        for i in range(len(points)):

            if not status[i]:
                continue

            diff = points[i] - last_corners[i]
            distance = np.linalg.norm(diff)

            if distance > 15 or distance < 0.2:
                continue

            otherP = last_corners[i] + diff * 15
            flowPoints += 1

            color = tuple([0, 255, 0])
            
            cv2.circle(last_frame, tuple(last_corners[i].astype(int)), 1, color)
            cv2.line(last_frame, tuple(last_corners[i].astype(int)), tuple(otherP.astype(int)), color)

        cv2.imshow("out", last_frame)
        cv2.waitKey(1)

        print("[corners] moving/total: {} / {}".format(flowPoints, len(points)))
    last_corners = np.copy(corners)
    last_grey = np.copy(grey)
    last_frame = np.copy(frame)
