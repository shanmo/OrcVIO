# -*- coding: utf-8 -*-

import numpy as np 
from collections import OrderedDict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

meanShape = np.array([[-0.30718615, 0.19259487, 0.13099417],
        [0.30625051, 0.19245705, 0.13087167],
        [0.30588192, -0.19297513, 0.13099417],
        [-0.30671038, -0.19352385, 0.13099417],
        [-0.28230901, 0.15100807, -0.1288375],
        [0.28118408, 0.14851215, -0.1284475],
        [0.28159246, -0.16839029, -0.12771],
        [-0.28178158, -0.16882001, -0.12771],
        [0. , 0.05205833, 0.03904333],
        [-0.05205833, 0. , 0.03904333],
        [0. , -0.05205833, 0.03904333],
        [0.05205833, 0. , 0.03904333]
       ])

# Average table dimensions (in meters)
AVG_HEIGHT = 1
AVG_WIDTH = 1
AVG_LENGTH = 1

# Compute (rough) dimensions (length, width, height) of the car shape prior
w = np.abs(np.max(meanShape[:, 0]) - np.min(meanShape[:, 0]))
l = np.abs(np.max(meanShape[:, 1]) - np.min(meanShape[:, 1]))
h = np.abs(np.max(meanShape[:, 2]) - np.min(meanShape[:, 2]))  

sx = AVG_WIDTH / w
sy = AVG_LENGTH / l
sz = AVG_HEIGHT / h

# Scale the mean shape and the basis vectors
meanShape_scaled = np.copy(meanShape)
meanShape_scaled[:, 0] = sx * meanShape[:, 0]
meanShape_scaled[:, 1] = sy * meanShape[:, 1]
meanShape_scaled[:, 2] = sz * meanShape[:, 2]

print("scaled shape kps")
print(repr(meanShape_scaled.T))

fig = figure()
ax = Axes3D(fig)

m = meanShape_scaled.T

for i in range(np.shape(meanShape)[0]): #plot each point + it's index as text above
  ax.scatter(m[0, i],m[1, i],m[2, i], color='b') 
  ax.text(m[0, i],m[1, i],m[2, i],  '%s' % (str(i)), size=20, zorder=1,  
  color='k') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pyplot.show()