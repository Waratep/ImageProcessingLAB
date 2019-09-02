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

image1 = cv2.imread('Lab02.jpg',1)
# image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
# image2 = cv2.imread('Lab02.jpg',1)
# image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
image2 = cv2.flip(image1,0)

# img_res = (image1 * 0.7) + (image2 * 0.3)
# img_res = np.uint8(img_res)
# plt.imshow(img_res)
# plt.show()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('Lab02-2.avi',fourcc, 5, (image1.shape[1],image1.shape[0]),1)
counter = 0
while counter < 1:
    img_res = (image1 * (1-counter)) + (image2 * (counter)) 
    img_res = np.uint8(img_res)
    counter += 0.05
    out.write(img_res)
counter = 0
while counter < 1:
    img_res = (image1 * (counter)) + (image2 * (1-counter)) 
    img_res = np.uint8(img_res)
    counter += 0.05
    out.write(img_res)


out.release()
