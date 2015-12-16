import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
import math
import pickle

#-----------------------------------------------------------------------
#class Gradient - creates a gradient for a certain object
class Gradient(object):
    """ Class which defines a gradient.
        name - the name of the species
        D - the diffusion coeffiecnt
        x, y, z - the spaital size of the grid
        dx, dy, dz - the spatial resolution of each grid step
        outside_c - the value fo the gradient outside the box
        unit_vol = the uinit vole in L (ul) m or um
    """

    def __init__(self, name, D, x, y, z, dx, dy, dz,
                 outside_c = 0.0, vol_units = 1e-6):
        #save the unit volume
        self._vol_units = 1e-6
        #save the name
        self.name = name
        #set the diffusion coefficient
        self._D = float(D)
        #set the dimensions in each direction
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)
        #set the spatial resolution in each direction
        self._dx = float(dx)
        self._dy = float(dy)
        self._dz = float(dz)
        #solve for the index of the array
        self.x_dim = int(x/dx)
        self.y_dim = int(y/dy)
        self.z_dim = int(z/dz)
        #solve some of the more specific constraints
        dx2 = self._dx**2
        dy2 = self._dy**2
        dz2 = self._dz**2
        #time step solution
        self.dt = .5 / ((D/dx2) + (D/dy2) + (D/dz2))
        print(self.dt)
        #set the iniatil conditions
        self.Ci = np.zeros((self.x_dim, self.y_dim, self.z_dim))*outside_c
        self.C = np.zeros((self.x_dim, self.y_dim, self.z_dim))
        #set the boundary conditions
        self._c_out = outside_c
        #compute the unit volume
        self._grid_vol = (self._vol_units **3)* self._dx * self._dy * self._dz
        #define the discrete laplacian operator
##        self._laplacian = np.array([[[0,0,0],[0,1/dz2,0],[0,0,0]],
##                    [[0,1/dy2,0],[1/dx2,-6/(dx2+dy2+dz2),1/dx2],[0,1/dy2,0]],
##                    [[0,0,0],[0,1/dz2,0],[0,0,0]]])
        self._lz = np.array([[[0,0,0],[0,1./dz2,0],[0,0,0]],
                    [[0,0,0],[0,-2./(dz2),0],[0,0,0]],
                    [[0,0,0],[0,1./dz2,0],[0,0,0]]])
        self._ly = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                    [[0,1./dy2,0],[0,-2./(dy2),0],[0,1./dy2,0]],
                    [[0,0,0],[0,0,0],[0,0,0]]])
        self._lx = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                    [[0,0,0],[1./dx2,-2./(dx2),1./dx2],[0,0,0]],
                    [[0,0,0],[0,0,0],[0,0,0]]])
        #make the 3D kernel
        self._l = self._lx + self._ly + self._lz

    def set_initial_conditions(self, mask):
        """ Will set the inital conditions everywhere a zero exsists
            in the mask, to the c_out parameter specified
            during the gradient creation
        """
        self.Ci = (mask == 0)*self._c_out
        
    def save(self, base_path, time_stamp):
        """ Saves the gradients to a binary numpy file
            Also outputs a string which represents
            import header information about the gradient
        """
##        path = base_path + self.name + "_" + time_stamp
##        np.save(path, self.C)
##        return path + ".npy"
        #delete the list
        path = base_path + self.name + "_" + time_stamp
        f = open(path, "w")
        pickle.dump(self, f)
        f.close()
        return path

    def shape(self):
        """ Returns the shape of the array
        """
        return (self.x_dim, self.y_dim, self.z_dim)

    def cubic_interpolate_at_object_locations(self, agents):
        """ Performs an interpolation of the whole data set
            based on a cubic intepolation, at all the specified points
        """
        #get all the agent locations
        points = []
        for i in range(0, len(agents)):
            #get the indicies of the gradient
            pos = agents[i].get_gradient_location(self.name)
            x = int(pos[0])
            y = int(pos[1])
            z = int(pos[2])
            points.append([x,y,z])
        #Now convert the points to an array
        points = np.array(points)
        #transpose it to the correct format
        cords = points.T
        #get the interpolated values
        zi = ndimage.map_coordinates(self.C, cords, order=5, mode='reflect')
        #set all the agents values
        for i in range(0, len(agents)):
            #now set the gradient value at this point
            agents[i].set_gradient_value(self.name, zi[i])
        
    def get_object_position_on_grid(self, distance):
        """ Assumes the distance vector is assigned as a vector pointing from
            the center out towards the point to index. Will return an error if
            point is outside the range of the grid
        """
        #now we know the spatial resolution per grid spacing
        dx = distance[0] / self._dx
        dy = distance[1] / self._dy
        dz = distance[2] / self._dz
        #get the center loction
        cx = int(self.x_dim / 2)
        cy = int(self.y_dim / 2)
        cz = int(self.z_dim / 2)
        #return the x, y, z indicies to add at
        return cx + dx, cy + dy, cz + dz

    def get_gradient_value_at_point(self, point):
        """ Will get the gradient value at a specific point. Useful for
            interpolation when the grid size is much bigger than a single cell.
            Assume the point is a distance vector representing the radius
            of the cell within the strutural aggregate
        """
        x, y, z = self.get_object_position_on_grid(point)
        #get the indices
        ix = int(x)
        iy = int(y)
        iz = int(z)
        #perform a trilinear interpolation from wikipedia.org
        #x
        if(x - ix > .5):
            x0 = ix
            x1 = ix + 1
        else:
            x0 = ix - 1
            x1 = ix
        #y
        if(y - iy > .5):
            y0 = iy
            y1 = iy + 1
        else:
            y0 = iy - 1
            y1 = iy
        #z
        if(z - iz > .5):
            z0 = iz
            z1 = iz + 1
        else:
            z0 = iz - 1
            z1 = iz                       
            
        #solve for xd, yd, zd
        xd = (abs(x - (ix + .5))) / (x1 - x0)
        yd = (abs(y - (iy + .5))) / (y1 - y0)
        zd = (abs(z - (iz + .5))) / (z1 - z0)

        #now the first set of linear interp
        c00 = self.C[x0, y0, z0]*(1 - xd) + self.C[x1, y0, z0]*xd
        c10 = self.C[x0, y1, z0]*(1 - xd) + self.C[x1, y1, z0]*xd
        c01 = self.C[x0, y0, z1]*(1 - xd) + self.C[x1, y0, z1]*xd
        c11 = self.C[x0, y1, z1]*(1 - xd) + self.C[x1, y1, z1]*xd

        #now the second set
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
        #finally the last set
        c = c0*(1-zd) + c1*zd
        #return the predicted value
        return c
    
    def update(self, t_end, sink, source):
        """ Solves the system over using the predetermined time step dt
            until the end time of the simulation is reached.
            t_end - the end time to solve the system towards
        """
        t = 0
        epsilon = 1E-10
        diff = epsilon  * 2
        while(t <= t_end and diff >= epsilon):
            #solve for the gradients in each direction
            l_xyz = ndimage.convolve(self.Ci, self._l, mode = "constant",
                                   cval = self._c_out)
            #first diffusion
            self.C = self.Ci + (l_xyz)*self._D*self.dt
            #MUST BE normalized by unit VOLUME
            temp_sink = (-sink*self.dt) / self._grid_vol
            temp_source = source*self.dt / self._grid_vol
            self.C += temp_sink + temp_source
            #get the summed difference
            diff = np.sum(np.abs(self.Ci - self.C))
            #make sure its positive
            self.C = self.C * (self.C > 0.0)
            #update the old
            self.Ci = self.C
            #update the time step
            t += self.dt

    def __repr__(self):
        """ Returns a string representation of the gradient
        """
        return self.name + ": " + repr(self.Ci.shape) + " " + repr(np.min(self.Ci)) + " " + repr(np.max(self.Ci))
        
                         
    #OPERATORS - equals and not equals
    #eq - eqauls
    def __eq__(self, other):
        if(other is None):
            #return not equal
            return False
        if(isinstance(other, Gradient)):
            if(other.name == self.name):
                return True
            else:
                return False
        #if nothing esle then return false
        return False
    
    #ne - not equals
    def __ne__(self, other):
        return not(self.__eq__(other))
    
    #also make it hashable
    def __hash__(self):
        return hash(self.name)
        

