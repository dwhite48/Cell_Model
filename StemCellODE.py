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
