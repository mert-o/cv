import cv2


def matchknn2(descriptors1, descriptors2):

    #Find the 2-Nearest Matches
    knnmatches = []
   

    for i in range(len(descriptors1)):
        knnmatches.append([cv2.DMatch( 0,0,float('inf')),cv2.DMatch( 0,0,float('inf'))])
        for j in range(len(descriptors2)):
            distance = cv2.norm(descriptors1[i],descriptors2[j],cv2.NORM_HAMMING)
            if distance < knnmatches[i][1].distance:
                knnmatches[i][1] = cv2.DMatch( i,j,distance )
                if distance < knnmatches[i][0].distance:
                    knnmatches[i][0], knnmatches[i][1] = knnmatches[i][1], knnmatches[i][0]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1,descriptors2)

    return knnmatches, matches

def ratioTest(knnmatches, ratio_threshold):
    matches = []

    #ratio test for cloes matches. above threshold is too ambigious and discarded
    for i in range(len(knnmatches)):

        if knnmatches[i][0].distance/knnmatches[i][1].distance < ratio_threshold:
            matches.append(knnmatches[i][0])

    return matches

def computeMatches(img1, img2):
    knnmatches, m = matchknn2(img1['descriptors'], img2['descriptors'])
    matches = ratioTest(knnmatches, 0.7)

    return matches
