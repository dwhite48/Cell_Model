################################################################################
# Name:   SimulationObjects
# Author: Douglas E. White
# Date:   10/17/2013
################################################################################
from simulationMath import *
import random as rand
import math as math

class SimulationObject(object):
    """ Base class from which all simulation obejcts must inherit
    """

    def __init__(self, location, radius, ID, owner_ID, sim_type):
        """ Base class which defines properties all sim objects MUST have
            location - the location of the sphere
            radius - the radius of the sphere
            ID - the ID of the object #WARNING# This ID is used ot hash objects
                  it MUST BE UNIQUE
            owner_ID - usually the same as the ID, also must be unique between
                       agents, unless all agents are part of a larger group
                       i.e. this is the mechanism for multi-agent agents
            sim_type - the type of object the simulation object is
        """
        self.location = location
        self.radius = radius
        self.sim_type = sim_type
        self.ID = ID
        self.owner_ID = owner_ID
        #keep track of the opt and col vecs
        self._disp_vec = [0,0,0]
        self._fixed_contraint_vec = [0,0,0]
        self._v = [0,0,0]
        #keep track of production consumptions values
        self.gradient_source_sink_coeff = dict()
        #keep track of the relative indices in the gradient array
        self.gradient_position = dict()
        #keep track of the value of the gradient associated with these agents
        self.gradient_value = dict()
         
    def update(self, sim, dt):
        """ Updates the simulation object
        """
        pass


    def get_max_interaction_length(self):
        """ Get the max interaction length of the object
        """
        return self.radius*2.0 #in um

    def get_interaction_length(self):
        return self.radius #in um

    def get_spring_constant(self, other):
        """ Gets the spring constant of the object
            Returns: 1.0 by default
            NOTE: Meant to be overwritten by a base class if more
                  functionality is required
        """
        return 0.25

    def add_displacement_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._disp_vec = AddVec(self._disp_vec, vec)

    def add_fixed_constraint_vec(self, vec):
        """ Adds a vector to the optimization vector
        """
        self._fixed_contraint_vec = AddVec(self._fixed_contraint_vec, vec)

    def set_gradient_source_sink_coeff(self, name, source, sink):
        """ Adds a production/consumption terms to the dicationary based on
            the gradient name
        """
        #overwrite exsisting data
        self.gradient_source_sink_coeff[name] = (source, sink)

    def get_gradient_source_sink_coeff(self, name):
        """ Gets the graident terms (source, sink) for the given gradient name
            returns - a tuple of the (source, sink) values. If these are not
                      in the dictionary, returns (0,0)
        """
        #returns the gradient values for the source and sink
        if(name in self.gradient_source_sink_coeff.keys()):
            #name is in the dictionary
            return self.gradient_source_sink_coeff[name]
        else:
            #the name is not in the dictionary
            return (0,0)

    def set_gradient_location(self, name, location):
        """ Adds the location of the agent on the grid for the gradient
            whose name is specified by name
        """
        self.gradient_position[name] = location

    def get_gradient_location(self, name):
        """ Return the location of the agent on the grid for the gradient
            specified by the name, name
        """
        if(name in self.gradient_position.keys()):
            #name is in the dictionary
            return self.gradient_position[name]
        else:
            #the name is not in the dictionary
            return None

    def set_gradient_value(self, name, value):
        """ Adds the value of the gradient at this agent
        """
        self.gradient_value[name] = value

    def get_gradient_value(self, name):
        """ Return the value fo the gradient at this agent
        """
        if(name in self.gradient_value.keys()):
            #name is in the dictionary
            return self.gradient_value[name]
        else:
            #the name is not in the dictionary
            return -1
        
    def update_constraints(self, dt):
        """ Updates all of the contraints on the object
        """
        #first update the posiiton by the col and opt vectors
        #make sure neither of these vectors is greater than error
        mag = Mag(self._disp_vec)
        if(mag > 5):
            n = NormVec(self._disp_vec)
            self._disp_vec = ScaleVec(n, 5.0)
        self.location = AddVec(self.location, self._disp_vec)
        #then clear it
        self._disp_vec = [0,0,0]
        
        #then update the the pos using the fixed vectors
        mag = Mag(self._fixed_contraint_vec)
        if(mag > 5):
            n = NormVec(self._disp_vec)
            self._fixed_contraint_vec = ScaleVec(n, 5.0)
        self.location = AddVec(self.location, self._fixed_contraint_vec)
        #htne clear it
        self._fixed_contraint_vec = [0,0,0]
        
    def __repr__(self):
        """ Returns a string representation of the object
        """
        return self.sim_type+": "+repr(self.ID)+" "+repr(self.location)

    def __eq__(self, other):
        """ Handles the equal operator for the object
        """
        if(isinstance(other, SimulationObject)):
            return self.ID == other.ID
        #otherwise
        return False

    def __hash__(self):
        """ Handles the hashing operator for the object
        """
        return hash(self.ID)
