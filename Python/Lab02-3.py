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

image = cv2.imread('Lab2-3.jpg',1)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# plt.figure(1)
# plt.subplot(121,title='origin')
# plt.imshow(image)

# color = [[246,246,246] , [24,164,252] , [241,241,241] , [203,203,203] , [50,50,50] , [173,222,254] , [101,101,101]]
# # n = 100
# # color = [('o', -50, -25), ('^', -30, -5)]
# ax = plt.figure(1).add_subplot(122, projection='3d')
for i in image[0:1]:
    print(i[0:])
#     for j in i:
#         ax.scatter(j[0], j[1], j[2], marker='o')
    
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()