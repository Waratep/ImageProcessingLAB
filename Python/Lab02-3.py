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
from PIL import Image


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

image = cv2.imread('Lab2-3.jpg',1)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.subplot(121,title='origin')
plt.imshow(image)
# plt.subplot(122,title='origin')
# plt.imshow(rescale_frame(image,20))

ax = plt.figure(1).add_subplot(122, projection='3d')
for i in rescale_frame(image,20):
    for j in i:
        print('ruinning',j[0], j[1], j[2])
        ax.scatter(j[0], j[1], j[2], marker='o')
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()