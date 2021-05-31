# -*- coding: utf-8 -*-

import numpy as np 
from collections import OrderedDict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

meanShape = np.array([[-1.32916667e-04, -1.60928750e-01, 2.01763333e-01],
       [-0.11672605, -0.16468605, 0.19614239],
       [0.11672303, -0.16524933, 0.19602163],
       [-0.02192129, -0.26586988, -0.08524955],
       [0.02273579, -0.26586488, -0.08522026],
       [-1.34333333e-04, 2.03842000e-01, 1.72801000e-01],
       [-1.20833333e-05, 3.85917857e-02, 1.77299643e-01],
       [-0.02799391, 0.26613146, -0.08263057],
       [0.02527268, 0.26451735, -0.08285319],
       [-0.03516337, 0.02347924, -0.09574672],
       [0.03071458, 0.0243553 , -0.094824]
       ])

# Average bike dimensions (in meters)
AVG_HEIGHT = 1.0
AVG_WIDTH = 0.3
AVG_LENGTH = 1.8

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