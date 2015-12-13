from simulationMath import *
import random as rand
import math as math
from simulationObjects import *

class DividingCell(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID,
                 division_set = 0.0,
                 division_time = 14.0,
                 owner_ID = None):
        """ Constructor for a stem cell
            location - the location fo the stem cell
            radius - the size of the stem cell
            ID - the unique ID for the agent
            state - the state of the stem cell
            division_set - the initial division set for the cell
            division_time - the time it takes the cell to divide
            owner_ID - the ID associated with the owner of this agent
        """
        #define some variables
        if(owner_ID == None):
            owner_ID = ID
        self.division_timer = division_set
        self.division_time = division_time
        #call the parent constructor
        super(DividingCell, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "dividingcell")

    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        #growth kinetics
        self.division_timer += dt
        if(self.division_timer >= self.division_time):
            #now you can divide
            #get the location
            #pick a random point on a sphere
            location = RandomPointOnSphere()*self.radius/2.0 + self.location
            #get the radius
            radius = self.radius
            #get the ID
            ID = sim.get_ID()
            #make the object
            sc = DividingCell(location, radius, ID)
            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0


    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um
