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

class FishCell(SimulationObject):
    """ A class for simulating fish embryo evolution
    """
    def __init__(self, location, radius, ID, state,
                 division_set = 0.0,
                 division_time = 14.0*60.0,
                 owner_ID = None,
                 params = [0,0,0,0]):
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
        self.state = state
        self._params = params
        self.delay_time = params[2]
        self.delay_count = 0
        #call the parent constructor
        super(FishCell, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "fishcell")
    
    def update(self, sim, dt):
        """ Updates the cell to decide wether they differentiate
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
            sc = FishCell(location, radius, ID, self.state,
                          division_time = self.division_time,
                          params = self._params)
            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0

        #state behaviors
        if(self.state == "n"):
            #secrete decide to differentiate
            x = rand.random()
            #express red based on the concentration of BMP in the system
            val = self.get_gradient_value("BMP")
            #compute a probability based on differentiation
            n = self._params[1]
            #old ka = 0.0091
            ka = self._params[0]
            prob = (1.*val**n)/(ka**n + val**n)
            if(x < prob):
                self.state = "ne"
            elif(x > prob):
                self.state = "np"
                self.set_gradient_source_sink_coeff("BMP_I", 100.0e-20,
                                                    0.0e-20)
        elif(self.state == "ne"):
            #you can differentiate into dlx3b
            #get the value of BMP4 and BMPI
            bmp = self.get_gradient_value("BMP")
            bmp_i = self.get_gradient_value("BMP_I")
            #if the BMP is low enough
            n = self._params[3]
            #old ka = 0.0091
            ka_bmp = self._params[2]
            p_bmp = 1 - (1.*bmp**n)/(ka_bmp**n + bmp**n)
            #and BMP_I is high enough then differentiate
            n = self._params[5]
            #old ka = 0.0091
            ka_bmp_i = self._params[4]
            p_bmp_i = (1.*bmp_i**n)/(ka_bmp_i**n + bmp_i**n)
            x1 = rand.random()
            x2 = rand.random()
            if(x1 < p_bmp and x2 < p_bmp_i):
                #move to dlx3b positive
                print('dlx3b')
                self.state = 'dlx3b'
        elif(self.state == 'np'):
            #do nothing
            pass
            


    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um

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
##        #make sure the soluble count is set to zero
##        self.sol_count = 0
##        self.sol_count_TNF = 0
##        #set its max to determine when the cell will differentiate
##        self.sol_count_max = 30
##        self.sol_count_TNF_max = 15
##        #save the parameters
##        self._params = params

        self.Nanog = 50
        self.Oct4_Sox2 = 50
        self.Gata6 = 0.1
        self.N = 0

        #set up all of the params
        self._params = params

    def model1(self, y, t, ks = []):
        #basic interaction
        #define the parameters
        #association/dissociation
        Oct4_Sox2 = y[0]
        Nanog = y[1]
        FGF = y[2]
        LIF = y[3]
        Gata6 = y[4]

        #Regulation constants
        #activation
        a_OS_OS = ks[0]
        a_LIF_OS = ks[1]
        a_N_O = ks[2]
        a_OS_N = ks[3]
        a_N_N = ks[4]
        a_G_G = ks[5]
        #inhibition
        i_N_G = ks[6]
        i_G_N = ks[7]
        i_LIF_G = ks[8]
        i_OS_G = ks[9]
        #FGF regulation
        i_FGF_N = ks[10]
        i_FGF_LIF = ks[11]
        k = ks[12]
        #degredation constants
        d_OS = ks[13]
        d_N = ks[14]
        d_FGF = ks[15]
        d_LIF = ks[16]
        d_Gata6 = ks[17]
        #Noise parameters
        sigma_N = ks[18]
        sigma_OS = ks[19]
        sigma_Gata6 = ks[20]

        #production LIF, FGF
        prod_FGF = ks[21]
        prod_LIF = ks[22]
        
        #Oct4 sox2 regulation
        d_Oct4_Sox2 = (a_OS_OS*Oct4_Sox2) / ((1./k) + Oct4_Sox2 + i_OS_G*Gata6)
        d_Oct4_Sox2 += (a_LIF_OS*LIF) / ((((1./k) + (LIF) + i_FGF_LIF*FGF**2)))
        d_Oct4_Sox2 += (a_N_O*Nanog) / (((1./k) + Nanog))
        d_Oct4_Sox2 -= d_OS*Oct4_Sox2
        d_Oct4_Sox2 += np.random.normal(0, sigma_OS)*Oct4_Sox2

        #Nanog regulation
        d_Nanog = (a_N_N*Nanog*Oct4_Sox2)  / ((1./k) + (Nanog*Oct4_Sox2) + i_FGF_N*FGF**2)
        d_Nanog -= d_N*Nanog
        d_Nanog -= i_G_N*Gata6 / ((1./k) + Gata6)
        d_Nanog += np.random.normal(0, sigma_N)*Nanog

        #FGF
        d_dt_FGF = prod_FGF*(Oct4_Sox2/ ((1/k) + Oct4_Sox2)) 
        d_dt_FGF += ks[23]*prod_FGF*(Gata6/ ((1/k) + Gata6))
        
        
        #LIF
        d_dt_LIF = prod_LIF - d_LIF*LIF

##        #solve for Gata6
##        if(Nanog < 1):
##            print(a_G_G*(Gata6) / ((1./k) + Gata6 + i_N_G*Nanog + i_LIF_G*LIF))
##            print(i_LIF_G*LIF)
##            print(i_N_G, Nanog)
##            print(i_N_G*Nanog)
        d_Gata6_dt = a_G_G*(Gata6) / ((1./k) + Gata6 + i_N_G*Nanog + i_LIF_G*LIF)
        d_Gata6_dt += np.random.normal(0, sigma_Gata6)*Gata6
        d_Gata6_dt -= d_Gata6*Gata6
        
        #return the values
        return [d_Oct4_Sox2, d_Nanog, d_dt_FGF, d_dt_LIF, d_Gata6_dt]

    
        
    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        #growth kinetics
        self.division_timer += dt
        #you can grow unless you are in the A state meaning apoptosis
        if(self.division_timer >= self.division_time and self.state != "A"):
            #check state
##                #then the stem cell is transitioning
##                #change it to a neuron
##                #keep the same ID, position, and size
##                print("Neuron formed...")
##                nueron = Nueron(self.location, self.radius, self.ID)
##                #add the neuron to the sim in the next time step
##                sim.objects_to_add.append(nueron)
##                #remove the stem cell from the sime in the next time step
##                sim.objects_to_remove.append(self)
##            #now you can divide
##            if(self.state == "T" or self.state == "T1"):
##                #change the current sytate to D
##                self.state = "D"
##                self.division_time = 51 #in hours
##                #also set the current consumption rate
##                source, consump_rate = self.get_gradient_source_sink_coeff("LIF")
####                c1 = self._params[3]
##                c2 = self._params[4]
####                c1 = 1.
####                c2 = 1.
##                self.set_gradient_source_sink_coeff("LIF", 0.1*source,
##                                                    consump_rate)
##                source, consump_rate = self.get_gradient_source_sink_coeff("TNF")
####                c1 = 1.
####                c2 = 1.
##                self.set_gradient_source_sink_coeff("TNF", 25.0*consump_rate,
##                                                    consump_rate)
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
##            #set its soluble count
##            sc.sol_count = self.sol_count / 2.
##            self.sol_count = self.sol_count / 2.
##            sc.sol_count_TNF = self.sol_count_TNF / 2.
##            self.sol_count_TNF = self.sol_count_TNF / 2.
            #copy over all of the coefficients to the new cells
            prod_cons = self.get_gradient_source_sink_coeff("LIF")
            sc.set_gradient_source_sink_coeff("LIF", prod_cons[0], prod_cons[1])
            prod_cons = self.get_gradient_source_sink_coeff("TNF")
            sc.set_gradient_source_sink_coeff("TNF", prod_cons[0], prod_cons[1])           

            #update all of its soluble properties
            sc.Nanog = self.Nanog
            sc.Oct4_Sox2 = self.Oct4_Sox2
            sc.Gata6 = self.Gata6


            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0

        #PERFORM INTEGRATION

        #perform integration of the species
        OS = self.Oct4_Sox2
        N = self.Nanog
        G = self.Gata6
        F = 10.*self.get_gradient_value("TNF")
        L = self.get_gradient_value("LIF")

        d_OS, d_N, d_FGF, d_LIF, d_G = self.model1([OS, N, F, L, G], dt, self._params)

        #Now at the end add these back in
        self.Oct4_Sox2 += d_OS*dt
        self.Nanog += d_N*dt
        self.Gata6 += d_G*dt

        #reset the FGF prodcution value
        d_FGF = 2.0e-20 * d_FGF
        prod_cons = self.get_gradient_source_sink_coeff("TNF")
        self.set_gradient_source_sink_coeff("TNF", d_FGF, prod_cons[1])

        #compla these values at .1
        self.Oct4_Sox2 = max(self.Oct4_Sox2, .1)
        self.Nanog = max(self.Nanog, .1)
        self.Gata6 = max(self.Gata6, .1)

        #determine your Oct4 and Nanog State
        if(self.Oct4_Sox2 > 1.0):
            self.state = "O+"
        else:
            self.state = "O-"
            print("Oct4 Negative " + self.state)
        #Nanog
        if(self.Nanog > .5):
            self.state += "N+"
            self.N = 0
        else:   
            self.state += "N-"
            self.N += 1
##            print("Nanog negative " + self.state)
##            print(self.Oct4_Sox2, self.Nanog, self.Gata6, d_G, d_N)

        #Increase division time
        if(self.state == "O-N-"):
            self.division_time = 51 # in hours
        else:
            self.division_time = 19 # in hours
##        print(self.Oct4_Sox2, self.Nanog, self.Gata6)
##        print(self.state)
        
##        #APOPOTOSIS
##        if(self.state != "A"):
##        #first look at the oxygen gradient
##            val = self.get_gradient_value("O2")
##            #min value aty which grow and apopotosis begins
##            min_c = (8.2e-6)*10**6 #umol/L or uM
####            min_c = 96
##            if(val < min_c):
##                #check the apopototic probability
##                x = rand.random()
##                prob = (min_c - val) / min_c
##                if(x >= prob):
##                    #The cell is now is an apopotosis state
##                    #we assume this timer takes ~ 15 hours
##                    self.state = "A"
##                    self._apotosis_time = 15 #hours
##                    self._apop_timer = 0.0
        
            
##        #STATE DEPENDANT BEHAVIOR    
##        if(self.state == "A"):
##            #check to see if this cell should die
##            if(self._apop_timer >= self._apotosis_time):
##                #then the cell is removed
##                sim.add_object_to_remnoval_queue(self)
##            else:
##                #shrink accordingly
##                self.radius = self.radius - 0.1 #in um
##                self._apop_timer += dt

##        if(self.state == "U"):
            #then the stem cell is still a stem cell
            #HANDLE DIFFERENTIATION

            #unpack the constraints
                #Regulation constants

            #Update the FGF and secreation profile

            #Check what 

##            #RANDOM RULE
##            x = rand.random()
####            prob = 0.00185 #used to be 0.0025 but we need it to take slightly
####            prob = 0.0075
####            prob = 0.005
####            prob = 0.0025
####            prob = 0.00125
##            prob = self._params[0]
##            #longer before the differentiation starts
##            if(x < prob):
##                #differentiation occurs
##                self.state = "T"
##
##            #Diff based on concentration level
##            val = self.get_gradient_value("LIF")
##            #compute a probability based on differentiation
##            n = self._params[2]
##            #old ka = 0.0091
##            ka = self._params[1]
##            prob1 = 1 - (1.*val**n)/(ka**n + val**n)
##            x1 = rand.random()
##            
##            val = self.get_gradient_value("TNF")
##            #compute a probability based on differentiation
##            n = self._params[2]
##            #old ka = 0.0091
####            ka = self._params[1] / 5.
##            ka = self._params[3]
##            prob2 = (1.*val**n)/(ka**n + val**n)
##            x2 = rand.random()
##            if(x1 <= prob1):
##                self.sol_count += 1.
####                print(self.sol_count)
##                if(self.sol_count >= self.sol_count_max):
##                    self.state = "T"
####                    print('LIF diff')
##            if(x2 <= prob2):
##                self.sol_count_TNF += 1
##                if(self.sol_count_TNF >= self.sol_count_TNF_max):
##                    self.state = 'T1'
##                    print('TNF diff')     
####            #get the neighboring states
####            nbs = sim.network.neighbors(self)
####            tot = len(nbs)
####            if(tot > 0):
####                u = 0
####                d = 0
####                for i in range(0, tot):
####                    if(nbs[i].state == "U" or nbs[i].state == "T"):
####                        u += 1
####                    if(nbs[i].state == "D"):
####                        d += 1
####                #set the hill coefficients
####                knf = self._params[1]
####                kpf = self._params[2]
####                n_p = self._params[3]
####                n_n = self._params[4]
####                #now calcualte the values
####                norm_u = float(u) / float(tot)
####                norm_d = float(d) / float(tot)
####                #use as a ratio of u to d
####                #compute the activator porbability
######                activate = (1*u**n) / (ka**n + u**n)
######                negative_feedback = 1. / (1. + (norm_u/knf)**n_n)
####                negative_feedback = 0
####                #compute the inhibitor probability
######                inhibit = 1 / (1 + (d/kd)**n)
######                postive_feedback = (1*norm_d**n_p)/(kpf**n_p + norm_d**n_p)
####                postive_feedback = 0
####                #Now compute them seperately
####                x1 = rand.random()
####                x2 = rand.random()
####                #IMPORTANT: normally this is an OR not AND
####                if(x1 < negative_feedback or x2 < postive_feedback):
####                    #put yourself into a differentiating state
####                    self.state = "T"
                

    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um

class StemCellMes(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID, state, params,
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
        self.params = params
        #call the parent constructor
        super(StemCellMes, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "stemcellmes")
        #make sure the soluble count is set to zero
        self.sol_count = 0
        #set its max to determine when the cell will differentiate
        self.sol_count_max = self.params[2]
        
    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        #growth kinetics
        self.division_timer += dt
        #you can grow unless you are in the A state meaning apoptosis
        if(self.division_timer >= self.division_time):
            #now you can divide
            if(self.state == "T"):
                #change the current sytate to D
                self.state = "D"
                self.division_time = 51 #in hours
                #also set the current consumption rate
                source, consump_rate = self.get_gradient_source_sink_coeff("EGF")
                self.set_gradient_source_sink_coeff("EGF", self.params[3]*source,
                                                    consump_rate)
            #get the location
            #pick a random point on a sphere
            location = RandomPointOnSphere()*self.radius/2.0 + self.location
            #get the radius
            radius = self.radius
            #get the ID
            ID = sim.get_ID()
            #make the object
            sc = StemCellMes(location, radius, ID, self.state, self.params,
                          division_time = self.division_time)
            #set its soluble count
            sc.sol_count = self.sol_count / 2.
            self.sol_count = self.sol_count / 2.
            #copy over all of the coefficients to the new cells
            prod_cons = self.get_gradient_source_sink_coeff("O2")
            sc.set_gradient_source_sink_coeff("O2", prod_cons[0], prod_cons[1])
            prod_cons = self.get_gradient_source_sink_coeff("EGF")
            sc.set_gradient_source_sink_coeff("EGF", prod_cons[0], prod_cons[1])           
            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0

        if(self.state == "D"):
            #then add some movement
            nbs = sim.network.neighbors(self)
            #keep track fo the displacement vec
            dists = [0,0,0]
            for n in nbs:
                if(n.state == "D"):
                    #Move toward this
                    dist = SubtractVec(n.location, self.location)
                    dist = NormVec(dist)
                    dists = AddVec(dist, dists)
            #Now we have the normalized movement
            self._disp_vec = AddVec(ScaleVec(dists, self.params[4]), self._disp_vec)
            
        if(self.state == "U"):
            #then the stem cell is still a stem cell
            #HANDLE DIFFERENTIATION

            #RANDOM RULE
            x = rand.random()
##            prob = 0.00185 #used to be 0.0025 but we need it to take slightly
##            prob = 0.0075
##            prob = 0.005
##            prob = 0.0025
            prob = 0.00125
            #longer before the differentiation starts
            if(x < prob):
                #differentiation occurs
                self.state = "T"

            #Diff based on concentration level
            val = self.get_gradient_value("EGF")
            #compute a probability based on differentiation
            n = self.params[1]
            #old ka = 0.0091
            ka = self.params[0]
            prob = 1 - (1.*val**n)/(ka**n + val**n)
            x = rand.random()
            #COMPETEING FEEDBACK RULE
            if(x < prob):
                self.sol_count += 1.
                if(self.sol_count >= self.sol_count_max):
                    self.state = "T"
    

    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        if(self.state == "U" or self.state == "T"):
            return self.radius + 2.0
        elif(self.state == "D"):
            return self.radius #in um

    def get_max_interaction_length(self):
        """ Get the max interaction length of the object
        """
        if(self.state == "U" or self.state == "T"):
            return self.radius*2.0 #in um
        elif(self.state == "D"):
            return self.radius*3.0 #in um


    def get_spring_constant(self, other):
        """ Gets the spring constant of the object
        """
        if(other.state == self.state):
            return 0.25
        else:
            #spring conastant is much less stiff as these regions don't
            #adhere to each other as much
            return .125
    
class NueronalStemCell(SimulationObject):
    """ A stem cell class
    """
    def __init__(self, location, radius, ID, state,
                 division_set = 0.0,
                 division_time = 25.0,
                 params = [.01, .1, .1, 1],
                 owner_ID = None):
        """ Constructor for a NueronalStemCell
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
        self._p1 = params[0]
        self._p2 = params[1]
        self._p3 = params[2]
        self._p4 = params[3]
        self._p5 = params[4]
        self._p6 = params[5]
        self._p7 = params[6]
        #call the parent constructor
        super(NueronalStemCell, self).__init__(location,
                                       radius,
                                       ID,
                                       owner_ID,
                                       "NueronalStemCell")
        #make sure the soluble count is set to zero
        self.sol_count = 0
        #set its max to determine when the cell will differentiate
        self.sol_count_max = 10
        #keeping track fo NSC division state
        self._division = True
        
    def update(self, sim, dt):
        """ Updates the stem cell to decide wether they differentiate
            or divide
        """
        #growth kinetics
        self.division_timer += dt
        #you can grow unless you are in the A state meaning apoptosis
        if(self.division_timer >= self.division_time and self._division):
            #now you can divide
            if(self.state == "T1"):
                #change the current sytate to D
                self.state = "NSC"
                self._division = False
                self.division_time = 36
                #progenitor time is faster with concentration factor

                #add the concentration
                source, consump_rate = self.get_gradient_source_sink_coeff("TNF")
                self.set_gradient_source_sink_coeff("TNF", 50.0*source, 1.0*consump_rate)
##                #get neighbors
##                nbs = sim.network.neighbors(self)
##                #get the total
##                tot = len(nbs)
##                mn_count = 0
##                for i in range(0, tot):
##                    if(nbs[i].state == "MN" or nbs[i].state == "T2"):  
##                        mn_count += 1
##                norm_mn = float(mn_count) / float(tot)
##                if(norm_mn < self._p2):
##                    self.division_time = 36*(norm_mn) # in hours
##                    self.division_time = max(self.division_time, 1) 
##                else:
##                    
##                print(norm_mn, self.division_time)
                #also set the current consumption rate
##                source, consump_rate = self.get_gradient_source_sink_coeff("EGF")
##                self.set_gradient_source_sink_coeff("EGF", source, 1.0*consump_rate)
            if(self.state == "T2"):
                #change the current sytate to D
                self.state = "MN"
                self.division_time = 56 #in hours
                #also set the current consumption rate
                source, consump_rate = self.get_gradient_source_sink_coeff("EGF")
                self.set_gradient_source_sink_coeff("EGF", 50.0*source, 1.0*consump_rate)
            if(self.state == "T3"):
                #change the current sytate to D
                self.state = "G"
                self.division_time = 56 #in hours
                #also set the current consumption rate
##                source, consump_rate = self.get_gradient_source_sink_coeff("EGF")
##                self.set_gradient_source_sink_coeff("EGF", source, 1.0*consump_rate)
            #get the location
            #pick a random point on a sphere
            location = RandomPointOnSphere()*self.radius/2.0 + self.location
            #get the radius
            radius = self.radius
            #get the ID
            ID = sim.get_ID()
            #make the object
            sc = NueronalStemCell(location, radius, ID, self.state,
                                  division_time = self.division_time,
                                  params = [self._p1, self._p2,
                                            self._p3, self._p4, self._p5,
                                            self._p6, self.p7])
            #copy secretion to NSC progeny
            if(self.state == "NSC"):
                source, consump_rate = self.get_gradient_source_sink_coeff("TNF")
                sc.set_gradient_source_sink_coeff("TNF", 50.0*source, 1.0*consump_rate)
                sc._division = False
            #set its soluble count
##            sc.sol_count = self.sol_count / 2.
##            self.sol_count = self.sol_count / 2.
            #copy over all of the coefficients to the new cells
##            prod_cons = self.get_gradient_source_sink_coeff("O2")
##            sc.set_gradient_source_sink_coeff("O2", prod_cons[0], prod_cons[1])
            prod_cons = self.get_gradient_source_sink_coeff("EGF")
            sc.set_gradient_source_sink_coeff("EGF", prod_cons[0], prod_cons[1])           
            #add it to the imsulation
            sim.add_object_to_addition_queue(sc)
            #reset the division time
            self.division_timer = 0
        
        if(self.state == "U"):
            #HANDLE DIFFERENTIATION
            #RANDOM RULE
            x = rand.random()
            prob = self._p1 #probability of turning into a NSC
            #longer before the differentiation starts
            if(x < prob):
                #differentiation occurs
                self.state = "T1"
             #also add a proabability to differentiate directly to a mn
            n1 = self._p4
##            #get neighbors
##            nbs = sim.network.neighbors(self)
##            #get the total
##            tot = len(nbs)
##            mn_count = 0
##            if(tot > 0):
##                #count up the states fo all fo these
##                for i in range(0, tot):
##                    if(nbs[i].state == "MN" or nbs[i].state == "T2"):  
##                        mn_count += 1
            #get the value fo the gradient and make differntiation inversly
            #inversly correlated with the proportion present
            norm_mn = self.get_gradient_value("EGF")
            #probability of turning into a motor nueron
            n1 = self._p4
##            #normalize the result
##            if(tot != 0):
##                norm_mn = float(mn_count) / float(tot)
##            else:
##                norm_mn = 0
            #calculate the probability
            prob_MN = 1 - (1.*norm_mn**n1)/(self._p2**n1 + norm_mn**n1)
            x1 = rand.random()
            if(x1 <= self._p1*prob_MN):
                #differentiation occurs towards a motor nueron
                self.state = "T2"
                
        if(self.state == "NSC"):
            #HANDLE DIFFERENTIATION
            #RANDOM RULE
            x1 = rand.random()
            x2 = rand.random()
            #Find all the motor nuerons
##            #get neighbors
##            nbs = sim.network.neighbors(self)
##            #get the total
##            tot = len(nbs)
##            mn_count = 0
##            if(tot > 0):
##                #count up the states fo all fo these
##                for i in range(0, tot):
##                    if(nbs[i].state == "MN" or nbs[i].state == "T2"):  
##                        mn_count += 1
##            #normalize the result
##            norm_mn = float(mn_count) / float(tot)
            #Make differerntiationd ependant on the gradient value
            norm_mn = self.get_gradient_value("EGF")
            #set the paramaters
            n1 = self._p4
            #update the division time
##            self.division_time = norm_mn * 38 #in hours takes care of the feedback
            #depends on other motor nuerons
            prob_MN = 1 - (1.*norm_mn**n1)/(self._p3**n1 + norm_mn**n1) #probability of turning into a motor nueron
##            prob_G = (1.*norm_mn**n2)/(self._p3**n1 + norm_mn**n2) #of turning into a glial cell
            prob_G = self._p5
            #longer before the differentiation starts
            if(x1 <= prob_MN and x2 > prob_G):
                #differentiation occurs towards a motor nueron
                self.state = "T2"
            if(x1 > prob_MN and x2 <= prob_G):
                #differentiation occurs towards a glial cell
                self.state = "T3"
            #check to see if division enabled
            if(self._division == False):
                #check for mitotic speed up
                a = self._p6
                b = self._p7
                norm_nsc = self.get_gradient_value("TNF")
                prob_divide = (1.*norm_nsc**b)/(a**b + norm_nsc**b)
                r = rand.random()
                if(r <= x):
                    self._division = True
                
    def get_interaction_length(self):
        """ Gets the interaciton elngth for the cell. Overiides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um
        

class Nueron(SimulationObject):
    """ A neruon class
    """
    def __init__(self, location, radius, ID,
                 owner_ID = None):
        """ Constructor for the nueron
            location - the location of the microparticle
            radius - the size of the microparticle
            ID - the unique ID for the agent
            strength - the strength of the microparticle
            decay_rate - the decay rate fo the microparticle
            owenr_ID - the ID associated with the owner of this agent
        """
        #call parent constructor
        super(Nueron, self).__init__(location,
                                     radius,
                                     ID,
                                     owner_ID,
                                     "nueron")
        #define some base class vairables
        #for the neurite growth
        self.number_of_neurites = 0
        #keep track of child neurites
        self.nuerites = []
        if(owner_ID == None):
            owner_ID = ID
        #set the gradient value production
        self.add_gradient_value("X", 0.01, 0.0)

    def update(self, sim, dt):
        #decide wether you grow or not
        #look for other nuerons around you
        #and use them to deicde whether to sprout or not
        nbs = sim.network.neighbors(self)
        #now sum up numer of mps and nuerons around you
        queues = 0
        for i in range(0, len(nbs)):
            if(nbs[i].sim_type == "nueron"):
                #add it to the sum
                queues += 1
        #now check if sprouting occurs
        queues_prob = 1.0/(1.0+math.exp(-(queues - self.number_of_neurites)))
        print(queues, self.number_of_neurites, queues_prob)
        if(queues_prob > .5):
            #grow a nuerite
            print("Nuerites!!!!")
            #get the location
            #pick a random point on a sphere
            location = RandomPointOnSphere()*self.radius + self.location
            #get the radius
            radius = self.radius * 0.4
            #get the ID
            ID = sim.get_ID()
            #make the object
            nuerite = Nuerite(location, radius, ID, self, owner_ID = self.ID)
            #now add this to the nuerite list
            self.nuerites.append(nuerite)
            #also increment the nuerite count
            self.number_of_neurites += 1
            #also add a fixed constraint between these objects
            sim.add_fixed_constraint(self, nuerite)
            #finally add it to the sim
            sim.add_object_to_addition_queue(nuerite)

    def get_interaction_length(self):
        """ Gets the interaciton length for the cell. Overides parent
            Returns - the length of any interactions with this cell (float)
        """
        return self.radius + 2.0 #in um

class Nuerite(SimulationObject):
    """ A nuerite class
    """
    
    def __init__(self, location, radius, ID, parent,
                 owner_ID = None):
        """ Constructor for the neurite
            location - the location of the microparticle
            radius - the size of the microparticle
            ID - the unique ID for the agent
            parent - the neurite must have a parent object associated with it
            owenr_ID - the ID associated with the owner of this agent
        """
        #define internal variables
        self.parent = parent
        self.children = []
        self._division_timer = 0.0
        self._divition_time = 2.0
        if(owner_ID == None):
            owner_ID = ID
        #call the parent constructor
        super(Nuerite, self).__init__(location,
                                      radius,
                                      ID,
                                      owner_ID,
                                      "nuerite")

    def update(self, sim, dt):
        """ Updates the neurite 
        """
        #decide wether this neurite should grow
        if(len(self.children) == 0):
            self._division_timer += dt
            #if you have no children and division timer is reached
            #grow a nuerite
            if(self._division_timer >= self._divition_time):
##                #get the gradient we care about
##                g = sim.get_gradient_by_name("X")
##                local_env = g.Ci[self.gx-1:self.gx+1,
##                                 self.gy-1:self.gy+1,
##                                 self.gz-1:self.gz+1]
##                #figure out which direction to grow in
##                vec = NormVec(FindHighestVector(local_env))
##                location = vec*self.radius + self.location
                #get the location
                l1 = self.parent.location
                vec = SubtractVec(self.location, l1)
                #normalize it
                norm = NormVec(vec)
                #divide in the opposite direction
                location = norm*self.radius + self.location
                #get the radius
                radius = self.radius * .75
                radius = max(radius, 0.5)
                #get the ID
                ID = sim.get_ID()
                #make the object
                nuerite = Nuerite(location, radius, ID, self, owner_ID = self.owner_ID)
                #set the child
                self.children.append(nuerite)
                #add a fixed constraint
                sim.add_fixed_constraint(self, nuerite)
                #finally add it to the sim
                sim.add_object_to_addition_queue(nuerite)
                #set the end to zero
                self._division_timer = 0
        
