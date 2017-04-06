# -*- coding: utf-8 -*-
'''
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
figure = plt.figure()
ax = Axes3D(figure)
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
#网格化数据
X, Y = np.meshgrid(X, Y)
Z = -2*X*X-3*Y*Y
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.contour(X, Y, Z)
plt.show()
'''

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

delta = 0.2
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(x, y)
Z = -0.5*(X-1)**2 - Y**2

x=X.flatten()
y=Y.flatten()
z=Z.flatten()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.01)

plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()