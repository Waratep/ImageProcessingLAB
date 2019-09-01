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

# print(img)
print('height',height,'width',width)

datatype = np.array(img)                                        #get data type of lab01.jpg
print('Data Type is',datatype.dtype)

color = ('b','g','r')                                           #show histogram of img (RGB)
for i,col in enumerate(color):
    histogram = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histogram ,color = col)
    plt.xlim([0,256])
plt.show()

Qlevel = 4
info = np.iinfo(gray.dtype)
Qstep = (info.max - info.min) / 2**Qlevel

img4bit = np.floor_divide(gray,Qstep) * Qstep
img4bit = np.uint8(img4bit)

datatype = np.array(img4bit)


cv2.imshow('img',img4bit) 
cv2.waitKey(0)   
cv2.destroyAllWindows()


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.subplot(121,title='1')
plt.imshow(img)
plt.colorbar(orientation='horizontal')

img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.subplot(122,title='2')
plt.imshow(img,cmap='gray')
plt.colorbar(orientation='horizontal')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')

x,y = np.mgrid[0:height,0:width]
surf = ax.plot_mesh(x, y, gray, cmap=cm.coolwarm,linewidth=0, antialiased=False)

ax.set_zlim(0, 256)
ax.zaxis.set_major_locator(LinearLocator(30))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()

