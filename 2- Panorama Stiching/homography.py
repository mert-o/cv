import numpy as np

# Compute a homography matrix from 4 point matches
# Ax = 0 Solution

def computeHomography(points1, points2):

    A = np.zeros((8,9))
    # Create the matrix A from given points
    for i,_ in enumerate(A):
        if i%2 == 0:
            j = int(np.floor(i/2))
            A[i] = np.array([-points1[j][0],-points1[j][1],-1,0,0,0,points1[j][0]*points2[j][0],points1[j][1]*points2[j][0],points2[j][0]])
        else:
            A[i] = np.array([0,0,0,-points1[j][0],-points1[j][1],-1,points1[j][0]*points2[j][1],points1[j][1]*points2[j][1],points2[j][1]])

    U, s, V = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V)

    # Homography matrix is the last column
    h = V[:,-1]
    
    H = np.zeros((3, 3))
    H = h.reshape(3,3)

    #Normalize
    H *= (1/h[-1])


    return H

