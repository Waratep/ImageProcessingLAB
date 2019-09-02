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

image1 = cv2.imread('Lab02.jpg',1)
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image2 = cv2.imread('Lab02-2.jpg',1)
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

img_res = (image1 * 2) + (image1 * 0.5)

plt.imshow(img_res)
plt.show()