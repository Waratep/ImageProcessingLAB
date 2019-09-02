import cv2
import numpy as np
import math
import scipy
from scipy import signal 
from scipy import misc 
from cv2 import VideoWriter, VideoWriter_fourcc 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import time

image = cv2.imread('Lab02.jpg',1)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


Wx = np.array([[-1,-2,-1], 
               [ 0, 0, 0],
               [ 1, 2, 1]])   

Wy = np.array([[-1, 0, 1], 
               [-2, 0, 2],
               [-1, 0, 1]])   

Gradientx = signal.convolve2d(image_gray, Wx, boundary='symm', mode='same')
Gradienty = signal.convolve2d(image_gray, Wy, boundary='symm', mode='same')

magnitude_gradient = np.sqrt(np.power(Gradientx,2) + np.power(Gradienty,2))
magnitude_gradient = magnitude_gradient.astype('float32')

print('min =',np.amin(magnitude_gradient))
print('max =',np.amax(magnitude_gradient))
print('mean =',np.mean(magnitude_gradient))
print('std =',np.std(magnitude_gradient))

print(magnitude_gradient.dtype)

plt.figure(1)
threshold = [40,50,80,120,150,160,180,200,250,300]
count = 1

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('Lab02.avi',fourcc, 2, (magnitude_gradient.shape[1],magnitude_gradient.shape[0]),0)

for th in threshold:
      ret , img_threshold = cv2.threshold(magnitude_gradient,th,723,cv2.THRESH_BINARY)
      img_threshold = np.uint8(img_threshold)
      out.write(img_threshold)

for i in range(10):
      ret , img_threshold = cv2.threshold(magnitude_gradient,threshold[9-i],723,cv2.THRESH_BINARY)
      img_threshold = np.uint8(img_threshold)
      out.write(img_threshold)
out.release()
plt.show()

# cap = cv2.VideoCapture('Lab02.avi')
# while(cap.isOpened()):
#       ret, frame = cap.read() 
#       cv2.imshow('frame',frame)
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#       time.sleep(0.5)
 
# cap.release()
# cv2.destroyAllWindows()

# plt.figure(1)
# plt.subplot(231,title='1')
# plt.imshow(image)
# plt.subplot(232,title='2')
# plt.imshow(image_gray,cmap='gray')
# plt.subplot(233,title='Gradientx')
# plt.imshow(Gradientx,cmap='gray')
# plt.subplot(234,title='Gradienty')
# plt.imshow(Gradienty,cmap='gray')
# plt.subplot(235,title='magnitude_gradient')
# plt.imshow(magnitude_gradient,cmap='gray')

# plt.subplot(236,title='calHist')
# calHist = cv2.calcHist([magnitude_gradient],[0],None,[int(np.amax(magnitude_gradient))],[int(np.amin(magnitude_gradient)),int(np.amax(magnitude_gradient))])
# plt.plot(calHist)
# plt.xlim([0,int(np.amax(magnitude_gradient))])
# plt.show()
