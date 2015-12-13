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
