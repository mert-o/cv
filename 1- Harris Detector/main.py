import cv2
import numpy as np

img = cv2.imread('blox.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray Scale',gray)
#cv2.waitKey(0)

#normalize
gray = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))

#For filtering type converting
gray = gray.astype(np.float32)

#Harris Responses

#Derivatives
Ix = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
Iy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)


#Second Moment Matrix
Ixx = Ix * Ix
Iyy = Iy * Iy
Ixy = Ix * Iy

""" cv2.imshow("Ix",Ix)
cv2.imshow("Iy",Iy)
cv2.imshow("Ixx",Ixx)
cv2.imshow("Iyy",Iyy)
cv2.imshow("Ixy",Ixy)
cv2.waitKey(0) """

#Rebluring components of Matrix with bigger Gaussian for response calculation
#
#       M= [A   B]
#          [B   C]

#Response -> det(M) - alpha * trace(M)^2

A = cv2.GaussianBlur(src=Ixx,ksize=(3,3),sigmaX=1,sigmaY=1)
B = cv2.GaussianBlur(src=Iyy,ksize=(3,3),sigmaX=1,sigmaY=1)
C = cv2.GaussianBlur(src=Ixy,ksize=(3,3),sigmaX=1,sigmaY=1)


""" cv2.imshow("A",A)
cv2.imshow("B",B)
cv2.imshow("C",C)
cv2.waitKey(0) """



response = (A*B - C*C) - 0.06*(A+B)**2

#Normalize Response

dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
dbg = dbg.astype(np.float32)

""" cv2.imwrite("dIdx.png", (abs(Ix) * 255.0))
cv2.imwrite("dIdy.png", (abs(Iy) * 255.0))

cv2.imwrite("A.png", (abs(A) * 5 * 255.0))
cv2.imwrite("B.png", (abs(B) * 5 * 255.0))
cv2.imwrite("C.png", (abs(C) * 5 * 255.0))

cv2.imwrite("response.png", np.uint8(dbg * 255.0))
 """



#Corner keypoint generation by min/max

points = []


for i in range(response.shape[0]):
    for j in range(response.shape[1]):
        if response[i,j] > 0.1 :
            r_max = np.max(response[i-1:i+2,j-1:j+2])
            if r_max == response[i,j]:
                points.append(cv2.KeyPoint(j,i,1))


#Edge generation by vertical or horizontal min/max

result = img.copy()

for y in range(response.shape[0]):
    for x in range(response.shape[1]):
        
        if response[y,x] < -0.01:

            x_min = np.min(response[y-1:y+2,x])
            y_min = np.min(response[y,x-1:x+2])
            if response[y,x] == y_min or response[y,x]==x_min:
                result[y,x] = (0,0,255)



#Final

imgKeypoints = cv2.drawKeypoints(img, points,  outImage=None, color=(0, 255, 0))
cv2.imshow("Harris Corners", imgKeypoints)
cv2.imshow("Harris Edges", result)
cv2.waitKey(0)

#cv2.imwrite("edges.png", result)
#cv2.imwrite("corners.png", imgKeypoints)
