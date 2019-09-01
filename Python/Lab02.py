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

plt.figure(1)
plt.subplot(121,title='1')
plt.imshow(image)
plt.subplot(122,title='2')
plt.imshow(image_gray,cmap='gray')
plt.show()


Wx = [[-1,-2,-1], 
      [ 0, 0, 0],
      [ 1, 2, 1]]   

Wy = [[-1, 0, 1], 
      [-2, 0, 2],
      [-1, 0, 1]]   