# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:50:08 2015

@author: Doug
"""
from scipy import ndimage
import numpy as np
import visvis as vv

vol_units = 1e-6
#save the name
name = "Test"
#set the diffusion coefficient
D = 10. #all distance units in um
#set the dimensions in each direction
x = 750.
y = 750.
z = 750.
#set the spatial resolution in each direction
dx = 10.
dy = 10.
dz = 10.
#solve for the index of the array
x_dim = int(x/dx)
y_dim = int(y/dy)
z_dim = int(z/dz)
#solve some of the more specific constraints
dx2 = dx**2
dy2 = dy**2
dz2 = dz**2
#time step solution
dt = .5 / ((D/dx2) + (D/dy2) + (D/dz2))
print(dt)

#set the boundary conditions
c_out = 0 #u-molar
#set the iniatil conditions
Ci = np.zeros((x_dim, y_dim, z_dim))*c_out
C = np.zeros((x_dim, y_dim, z_dim))

#compute the unit volume
grid_vol = (vol_units **3)* dx * dy * dz
#define the discrete laplacian operator
##        laplacian = np.array([[[0,0,0],[0,1/dz2,0],[0,0,0]],
##                    [[0,1/dy2,0],[1/dx2,-6/(dx2+dy2+dz2),1/dx2],[0,1/dy2,0]],
##                    [[0,0,0],[0,1/dz2,0],[0,0,0]]])
lz = np.array([[[0,0,0],[0,1./dz2,0],[0,0,0]],
            [[0,0,0],[0,-2./(dz2),0],[0,0,0]],
            [[0,0,0],[0,1./dz2,0],[0,0,0]]])
ly = np.array([[[0,0,0],[0,0,0],[0,0,0]],
            [[0,1./dy2,0],[0,-2./(dy2),0],[0,1./dy2,0]],
            [[0,0,0],[0,0,0],[0,0,0]]])
lx = np.array([[[0,0,0],[0,0,0],[0,0,0]],
            [[0,0,0],[1./dx2,-2./(dx2),1./dx2],[0,0,0]],
            [[0,0,0],[0,0,0],[0,0,0]]])

l = lx + ly + lz

#define theb sink matrix
sink = np.zeros(Ci.shape)
#define the source matrix
source = np.zeros(Ci.shape)
source[30:32,30:32,30:32] = 2E-15

#apply this to the inital condition
t = 0
t_end = 3600
epsilon = 1E-10
diff = epsilon  * 2
zeros = np.zeros(Ci.shape)
while(t <= t_end and diff >= epsilon):
    #solve for the gradients in each direction
    l_xyz = ndimage.convolve(Ci, l, mode = "constant",
                           cval = c_out)
#    l_y = ndimage.convolve(Ci, ly, mode = "constant",
#                           cval = c_out)
#    l_z = ndimage.convolve(Ci, lz, mode = "constant",
#                           cval = c_out)
    #first diffusion
    C = Ci + (l_xyz)*D*dt
    #MUST BE normalized by unit VOLUME
    temp_sink = (-sink*dt) / grid_vol
    temp_source = source*dt / grid_vol
    C += temp_sink + temp_source
    #get the summed difference
    diff = np.sum(np.abs(Ci - C))
    #make sure its positive
    C = C * (C > 0.0)
    #update the old
    Ci = C
    #update the time step
    t += dt

vv.use('qt4')    
vv.volshow3(C)
app = vv.use()
app.Run()