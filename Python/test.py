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

pixel_values = np.array(image).reshape((image.shape[0], image.shape[1], 3))
# print(pixel_values)
plt.figure(1)
plt.subplot(121,title='origin')
plt.imshow(image)
plt.show()