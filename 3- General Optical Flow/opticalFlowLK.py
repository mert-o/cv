
import numpy as np
import cv2

from util import *


class OpticalFlowLK:

    def __init__(self, winsize, epsilon, iterations):

        self.winsize = winsize
        self.epsilon = epsilon
        self.iterations = iterations

    def compute(self, prevImg, nextImg, prevPts):

        N = prevPts.shape[0]
        status = np.ones(N, dtype=int)
        nextPts = np.copy(prevPts)
        
        #Get the derivatives of the last frame
        prevDerivx = cv2.Scharr(prevImg,cv2.CV_64F,1,0, scale= 1/32)
        prevDerivy = cv2.Scharr(prevImg,cv2.CV_64F,0,1,scale= 1/32)

        halfWin = np.array([(self.winsize[0] - 1) * 0.5, (self.winsize[1] - 1) * 0.5])
        weights = computeGaussianWeights(self.winsize, 0.3)

        for ptidx in range(N):
            u0 = prevPts[ptidx]
            u0 -= halfWin


            u = u0
            iu0 = [int(np.floor(u0[0])), int(np.floor(u0[1]))]

            #Check if the point is in the window
            if iu0[0] < 0 or iu0[0] + self.winsize[0] >= prevImg.shape[1] - 1 or iu0[1] < 0 or (iu0[1] + self.winsize[1] >= (prevImg.shape[0] - 1)):
                status[ptidx] = 0
                continue

            bw = computeBilinerWeights(u0)

            bprev = np.zeros((self.winsize[0] * self.winsize[1], 1))

            #Points

            # W . A . x = W . b
            # W -> Gaussian Weight
            # A -> Constructed by Jacobian of previous image
            # x -> Displacment
            # b -> disparity between previous and the current image
            # Displacement becomes -> x = (A.T @ W @ A )^-1 @ A.T @ W @ b
            
            A = np.zeros((self.winsize[0] * self.winsize[1], 2))

            AtWA = np.zeros((2, 2))
            invAtWA = np.zeros((2, 2))


            foo = 0
            for y in range(self.winsize[1]):
                
                for x in range(self.winsize[0]):

                    #The real positions on image
                    gx = int(iu0[0] + x)
                    gy = int(iu0[1] + y)

                    #Compute the interpolated intensity
                    inter_point = (bw[0] * prevImg[gy, gx] +
                                bw[2] * prevImg[gy + 1, gx] +
                                bw[1] * prevImg[gy, gx + 1] +
                                bw[3] * prevImg[gy + 1, gx + 1])


                    bprev[foo][0] = inter_point
                    
                    # I multiplied the weights here
                    A[foo,0] = prevDerivx[gy,gx] * weights[y,x]
                    A[foo,1] = prevDerivy[gy,gx] * weights[y,x]
                
                    foo+=1


            #Weights multiplied by the window in the above for loop
            AtWA = A.T @ A

            invAtWA = invertMatrix2x2(AtWA)

            #Estimate the target point with the previous point
            u = u0

            # Lucas - Kanade
            for j in range(self.iterations):
                iu = [int(np.floor(u[0])), int(np.floor(u[1]))]

                #window function
                if iu[0] < 0 or iu[0] + self.winsize[0] >= prevImg.shape[1] - 1 or iu[1] < 0 or iu[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                    status[ptidx] = 0
                    break

                bw = computeBilinerWeights(u)

                AtWbnbp = [0, 0]

                bnext = np.zeros((self.winsize[1]*self.winsize[0],1))
                foo = 0
                for y in range(self.winsize[1]):
                    for x in range(self.winsize[0]):
                        gx = iu[0] + x
                        gy = iu[1] + y


                        inter_point = (bw[0] * nextImg[gy, gx] +
                                    bw[2] * nextImg[gy + 1, gx] +
                                    bw[1] * nextImg[gy, gx + 1] +
                                    bw[3] * nextImg[gy + 1, gx + 1])
                        
                        #Interpolated intensitiy in current image
                        bnext[foo][0] = inter_point 
                        foo+=1
                

                # A.T already multiplied with weights
                halfWin = -invAtWA @ A.T  @ (bnext - bprev)

                #Stoping condition
                if np.linalg.norm(halfWin)**2 > self.epsilon: break

            halfWin = np.array([halfWin[0][0],halfWin[1][0]])
            nextPts[ptidx] = u + halfWin
            

        return nextPts, status

