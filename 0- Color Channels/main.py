import numpy
import cv2






file = 'img.png'

image = cv2.imread(file)
cv2.imshow('image',image)
cv2.waitKey(0)


#Resizing an image
smallim = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
cv2.imwrite('small.png',smallim)
cv2.imshow('image',smallim)
cv2.waitKey(0)




#Show intensity of color channels
#OpenCV -> Blue, Green, Red
bChannel = image[:,:,0]
blue_im = numpy.zeros(image.shape)
blue_im[:,:,0] = bChannel

cv2.imwrite('blue.png',blue_im)

gChannel = image[:,:,1]
green_im = numpy.zeros(image.shape)
green_im[:,:,1] = gChannel

cv2.imwrite('green.png',green_im)

rChannel = image[:,:,2]
red_im = numpy.zeros(image.shape)
red_im[:,:,2] = rChannel

cv2.imwrite('red.png',red_im)

blue = cv2.imread('blue.png')
cv2.imshow('Blue Channel',blue)
cv2.waitKey(0)

green = cv2.imread('green.png')
cv2.imshow('Green Channel', green)
cv2.waitKey(0)

red = cv2.imread('red.png')
cv2.imshow('Red Channel', red)
cv2.waitKey(0)


