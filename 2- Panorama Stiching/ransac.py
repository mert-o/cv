
import numpy as np
from homography import computeHomography

def numInliers(points1, points2, H, threshold):

    inlierCount = 0

    #Count number of inliers with a given Homography matrix
    for i in range(len(points1)):
        x = np.array([points1[i][0],points1[i][1],1])

        x = x[None,:]

        pr = H @ x.T
        
        pr = pr[:-1]/pr[-1]

        p2 = np.array(points2[i])
        p2 = p2[None,:].T

        #Check for difference between tranformed at ground point
        df =  p2 - pr

        if np.linalg.norm(df) < threshold: 

            inlierCount += 1

    return inlierCount

def computeHomographyRansac(img1, img2, matches, iterations, threshold):

    #Ransac for finding the best homography
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(img1['keypoints'][matches[i].queryIdx].pt)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)

    bestInlierCount = 0
    bestH = 0
    for i in range(iterations):
        subset1 = []
        subset2 = []
        for i in range(4):
            f = np.random.randint(len(points1)-1)

            subset1.append(points1[f])
            subset2.append(points2[f])

        h = computeHomography(subset1,subset2)

        foo = numInliers(points1,points2,h,threshold)

        if foo > bestInlierCount: 
            bestInlierCount=foo
            bestH = h

    print ("Found " + str(bestInlierCount) + " RANSAC inliers.")
    return bestH
