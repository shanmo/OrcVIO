# -*- coding: utf-8 -*-

import numpy as np 
from collections import OrderedDict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

meanShape = np.array([[-0.3585725 , -0.14129875, -0.21807875],
        [0.359125 , -0.14131625, -0.21808875],
        [-0.358585 , -0.1412875, 0.276735],
        [0.35913625, -0.1413175 , 0.27672],
        [-0.1128375, 0.125925 , -0.0737725],
        [0.1128375, 0.125925 , -0.0737725],
        [-0.1128375, 0.125925 , 0.075065],
        [0.1128375, 0.125925 , 0.075065]
        ])

# Average monitor dimensions (in meters)
AVG_HEIGHT = 0.5
AVG_WIDTH = 0.5
AVG_LENGTH = 0.05

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