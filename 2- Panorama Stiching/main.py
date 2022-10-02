import numpy as np
import cv2
import os

from matching import computeMatches
from ransac import computeHomographyRansac
from stitcher import createStichedImage

IMAGE_DIR = 'data/'


images = ["7.jpg",  "8.jpg",  "9.jpg",  "10.jpg", "11.jpg",
        "12.jpg", "13.jpg", "14.jpg", "15.jpg"]

image_data_dicts = []
for i, image_name in enumerate(images):
    image_data = {}
    image_data['file'] = image_name

    image_path = os.path.join(IMAGE_DIR, image_name)

    image_data['img'] = cv2.imread(image_path)
    image_data['img'] = cv2.resize(image_data['img'], None, fx=0.5, fy=0.5)

    image_data['id'] = i


    #for image stiching
    image_data['HtoReference'] = np.eye(3)
    image_data['HtoPrev'] = np.eye(3)
    image_data['HtoNext'] = np.eye(3)


    image_data_dicts.append(image_data)


#Computing Features with ORB per image


temp = []
for image_data in image_data_dicts:

    orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
    keypoints, descriptors = orb.detectAndCompute(image_data['img'], None)
    print ("ORB features on image " + str(image_data['id']) + " : " + str(len(keypoints)))

    image_data['keypoints'] = keypoints
    image_data['descriptors'] = descriptors
    temp.append(image_data)

image_data_dicts = temp




for i in range(1, len(image_data_dicts)):


    matches = computeMatches(image_data_dicts[i-1], image_data_dicts[i])

    H = computeHomographyRansac(image_data_dicts[i-1], image_data_dicts[i], matches, 1000, 2.0)
    image_data_dicts[i]['HtoPrev'] = np.linalg.inv(H)
    image_data_dicts[i-1]['HtoNext'] = H

    ## =============== Stitching ==================
simg = createStichedImage(image_data_dicts)
cv2.imwrite("output.png", simg)