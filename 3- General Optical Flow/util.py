import numpy as np

def computeBilinerWeights(q):

    #Compute bilinear interpolation
    #Entry 0 weight for pixel (x, y)
    #Entry 1 weight for pixel (x + 1, y)
    #Entry 2 weight for pixel (x, y + 1)
    #Entry 3 weight for pixel (x + 1, y + 1)

    x = np.floor(q[0])
    y = np.floor(q[1])

    a = q[0]-x
    b = q[1]-y

    weights = [1, 0, 0, 0]

    weights[0] = (1-a)*(1-b)
    weights[1] = a*(1-b)
    weights[2] = (1-a)*b 
    weights[3] = a*b

    return weights

def computeGaussianWeights(winsize, sigma):

    #Gauss weight
    weights = np.zeros(winsize)

    wCenter = (winsize[1]-1) / 2
    hCenter = (winsize[0]-1) / 2

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            x_ = (wCenter-j)/winsize[1]
            y_ = (hCenter-i)/winsize[0]

            weights[i,j] = np.exp(-(x_**2+y_**2)/(2*sigma**2))

    return weights

def invertMatrix2x2(A):

    a,b,c,d = A.flatten()

    F = np.array([[d, -b],[-c,a]])

    invA = 1/(a*d-c*b)
    invA *= F
    

    return invA

