from simulationMath import *
import random as rand
import math as math
from simulationObjects import *

class StemCell(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID, state,
                 params = None,
                 division_set = 0.0,
                 division_time = 19.0,
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
        #set thet state
        self.state = state
        self.division_timer = division_set
        self.division_time = division_time
        #call the parent constructor
        super(StemCell, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "stemcell")
        #make sure the soluble count is set to zero
        self.sol_count = 0
        self.sol_count_TNF = 0
        #set its max to determine when the cell will differentiate
        self.sol_count_max = 30
        self.sol_count_TNF_max = 15
        #save the parameters
        self._params = params

        #set up all of the params
        self._params = params
        
    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        #growth kinetics
        self.division_timer += dt
        #you can grow unless you are in the A state meaning apoptosis
        if(self.division_timer >= self.division_time and self.state != "A"):
            #now you can divide
            if(self.state == "T" or self.state == "T1"):
                #change the current sytate to D
                self.state = "D"
                self.division_time = 51 #in hours
                #also set the current consumption rate
                source, consump_rate = self.get_gradient_source_sink_coeff("LIF")
##                c1 = self._params[3]
                c2 = self._params[4]
                self.set_gradient_source_sink_coeff("LIF", 0.1*source,
                                                    consump_rate)
                source, consump_rate = self.get_gradient_source_sink_coeff("TNF")
                self.set_gradient_source_sink_coeff("TNF", 25.0*consump_rate,
                                                    consump_rate)
            #get the location
            #pick a random point on a sphere
            location = RandomPointOnSphere()*self.radius/2.0 + self.location
            #get the radius
            radius = self.radius
            #get the ID
            ID = sim.get_ID()
            #make the object
            sc = StemCell(location, radius, ID, self.state,
                          params = self._params,
                          division_time = self.division_time)
            #set its soluble count
            sc.sol_count = self.sol_count / 2.
            self.sol_count = self.sol_count / 2.
            sc.sol_count_TNF = self.sol_count_TNF / 2.
            self.sol_count_TNF = self.sol_count_TNF / 2.
            
            #copy over all of the coefficients to the new cells
            prod_cons = self.get_gradient_source_sink_coeff("LIF")
            sc.set_gradient_source_sink_coeff("LIF", prod_cons[0], prod_cons[1])
            prod_cons = self.get_gradient_source_sink_coeff("TNF")
            sc.set_gradient_source_sink_coeff("TNF", prod_cons[0], prod_cons[1])           

            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0
        

        if(self.state == "U"):
            #then the stem cell is still a stem cell
            #HANDLE DIFFERENTIATION

            #RANDOM RULE
            x = rand.random()
##            prob = 0.00185 #used to be 0.0025 but we need it to take slightly
##            prob = 0.0075
##            prob = 0.005
##            prob = 0.0025
##            prob = 0.00125
            prob = self._params[0]
            #longer before the differentiation starts
            if(x < prob):
                #differentiation occurs
                self.state = "T"

            #Diff based on concentration level
            val = self.get_gradient_value("LIF")
            #compute a probability based on differentiation
            n = self._params[2]
            #old ka = 0.0091
            ka = self._params[1]
            prob1 = 1 - (1.*val**n)/(ka**n + val**n)
            x1 = rand.random()
            
            val = self.get_gradient_value("TNF")
            #compute a probability based on differentiation
            n = self._params[2]
            #old ka = 0.0091
##            ka = self._params[1] / 5.
            ka = self._params[3]
            prob2 = (1.*val**n)/(ka**n + val**n)
            x2 = rand.random()
            if(x1 <= prob1):
                self.sol_count += 1.
##                print(self.sol_count)
                if(self.sol_count >= self.sol_count_max):
                    self.state = "T"
##                    print('LIF diff')
            if(x2 <= prob2):
                self.sol_count_TNF += 1
                if(self.sol_count_TNF >= self.sol_count_TNF_max):
                    self.state = 'T1'
                    print('TNF diff')     
                

    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um
