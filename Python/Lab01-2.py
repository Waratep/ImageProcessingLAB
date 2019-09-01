import cv2
import numpy as np
import math

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm 
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter 

img = cv2.imread('lab01.jpg',1)                                 #read imgae (RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                     #convert img rgb to gray
dimensions = img.shape                                          #get dimensions from lab01.jpg
height = dimensions[0]                                          #(img.shape return (height, width, channel) )
width = dimensions[1] 


lower = [0, 200, 130]
upper = [120, 255, 255]


lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

mask = cv2.inRange(img, lower, upper)

output = cv2.bitwise_and(img, img, mask = mask)

cv2.bitwise_not(mask, mask)

gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
output1 = cv2.bitwise_and(gray, gray , mask = mask)


plt.figure(1)
plt.subplot(222,title='2')
plt.imshow(cv2.cvtColor(output1 + output,cv2.COLOR_BGR2RGB))

plt.subplot(221,title='1')
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

plt.subplot(223,title='3')
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(cv2.cvtColor(imgRGB,cv2.COLOR_BGR2YCR_CB))

plt.subplot(224,title='4')
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(cv2.cvtColor(imgRGB,cv2.COLOR_BGR2HSV))
plt.savefig('Lab01-2.png')
plt.show()

