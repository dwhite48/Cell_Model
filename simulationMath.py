import math as math
import random as rand
import numpy as np
from scipy import ndimage

def CalculateLocalGradient(array):
    """ Calcualtes the local gradient of the specified array
    """
    pass


def FindHighestVector(array):
    """ Finds the highest point in the array and returns the vector
        representing the direction of movement. If multiple maximum
        are detected, this fucntion returns all arrays
    """
    x,y,z = array.shape
    pos = ndimage.measurements.maximum_position(array)
    #get the difference of this form the middle of the array
    mx = int(x/2)
    my = int(y/2)
    mz = int(z/2)
    #Now compute the vector
    vec = SubtractVec(pos, (mx, my, mz))
    return vec
    
def FindLowestVector(array):
    """ Finds the lowest point in the array and returns the vector
        representing the direction of movement. If multiple minima
        are detected, this fucntion returns all arrays
    """
    x,y,z = array.shape
    pos = ndimage.measurements.minimum_position(array)
    #get the difference of this form the middle of the array
    mx = int(x/2)
    my = int(y/2)
    mz = int(z/2)
    #Now compute the vector
    vec = SubtractVec(pos, (mx, my, mz))
    return vec    

def RandomPointOnSphere():
    """ Computes a random point on a sphere
        Returns - a point on a unit sphere [x,y,z] at the origin
    """
    u = rand.random()*pow(-1., rand.randint(0,1))
    theta = rand.random()*2*math.pi
    x = math.sqrt(1-(u*u))*math.cos(theta)
    y = math.sqrt(1 - (u*u))*math.sin(theta)
    z = u
    return np.array((x,y,z))

def AddVec(v1, v2):
    """ Adds two vectors that are in the form [x,y,z]
        Returns - a new vector [x,y,z] as a numpy array
    """
    return np.array([v1[0] + v2[0],
                     v1[1] + v2[1],
                     v1[2] + v2[2]])

def SubtractVec(v1, v2):
    """ Subtracts vector [x,y,z] v2 from vector v1
        Returns - a new vector [x,y,z] as a numpy array
    """
    return np.array([v1[0] - v2[0],
                     v1[1] - v2[1],
                     v1[2] - v2[2]])

def ScaleVec(v1, s):
    """ Scales a vector f*[x,y,z] = [fx, fy, fz]
        Returns - a new scaled vector [x,y,z] as a numpy array
    """
    return np.array([v1[0]*s, v1[1]*s, v1[2]*s])

def Mag(v1):
    """ Computes the magnitude of a vector
        Returns - a float representing the vector magnitude
    """
    return math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])

def NormVec(v1):
    """ Computes a normalized version of the vector v1
        Returns - a normalizerd vector [x,y,z] as a numpy array
    """
    mag = Mag(v1)
    if(mag == 0):
        return np.array([0,0,0])
    return np.array([v1[0]/mag, v1[1]/mag, v1[2]/mag])

def Distance(p1, p2):
    """ Computes the distance between two points [x,y,z]
        Returns - a float representing the distance
    """
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) +
                    (p1[1]-p2[1])*(p1[1]-p2[1]) +
                    (p1[2]-p2[2])*(p1[2]-p2[2]))
