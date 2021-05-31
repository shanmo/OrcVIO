# -*- coding: utf-8 -*-

import numpy as np 
from collections import OrderedDict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

meanShape = np.array([[-0.15692863, 0.16241859, -0.28825333],
       [-0.1798645 , 0.12591167, -0.005276],
       [0.1585531 , 0.16203016, -0.28810933],
       [0.1746565 , 0.12572383, -0.005172],
       [-0.16835921, -0.18672756, -0.28799594],
       [-0.18498084, -0.20056212, -0.00706541],
       [0.17062087, -0.18674056, -0.28799594],
       [0.18131248, -0.19883625, -0.00708729],
       [-0.19126813, 0.18548547, 0.3550319],
       [0.18972683, 0.1843335 , 0.355234]])

# Average chair dimensions (in meters)
AVG_HEIGHT = 1
AVG_WIDTH = 0.4
AVG_LENGTH = 0.4

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