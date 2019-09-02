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

print('min =',np.amin(magnitude_gradient))
print('max =',np.amax(magnitude_gradient))
print('mean =',np.mean(magnitude_gradient))
print('std =',np.std(magnitude_gradient))


# calHist = cv2.calcHist([magnitude_gradient],[0],None,[256],[0,256])
# plt.plot(histogram ,color = 'red')
# plt.xlim([0,256])

plt.figure(1)
plt.subplot(231,title='1')
plt.imshow(image)
plt.subplot(232,title='2')
plt.imshow(image_gray,cmap='gray')
plt.subplot(233,title='Gradientx')
plt.imshow(Gradientx,cmap='gray')
plt.subplot(234,title='Gradienty')
plt.imshow(Gradienty,cmap='gray')
plt.subplot(235,title='magnitude_gradient')
plt.imshow(magnitude_gradient,cmap='gray')
plt.show()
