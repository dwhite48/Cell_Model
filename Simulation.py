################################################################################
# Name:   Simulation v 3.0
# Author: Douglas E. White
# Date:   10/16/2013
################################################################################
import os, sys
import platform
import networkx as nx
import numpy as np
from simulationMath import *
from simulationObjects import *
from Gradient import Gradient
from scipy.spatial import *
import pickle

#create the time series class
class TimeSeries(object):
    """ A TimeSeries class meant to hold all of the information about
        a given simulation data set
    """

    def __init__(self, path):
        """ Initilization function for the time series setup
            path - the directory holding the simulation information
            ERRORS - TypeError if path is not a string
        """
        #set some base params
        self._networks = dict()
        self._gradients = dict()
        
        #check types
        if(type(path) is not str):
            raise TypeError("The path argument must be of type string")
        self.path = path
        #make sure path exsists
        if(not(os.path.exists(path))):
            raise IOError("""The path specified either does not exist, or you
                            don not have the privledges to acess it.""")       

        #keep track of the fuile sepeartor to use
        if(platform.system() == "Windows"):
            #windows
            self.__sep = "\\"
        else:
            #linux/unix
            self.__sep = "/"
        #load the actual file
        self._load()

    def _load(self):
        """ Takes care of laoding the simulation header file
        """
        #try to load
        try:
            f = open(self.path + self.__sep + "info.sim", "r")
        except IOError:
            raise IOError("""Simulation header file not found""")
        #ok now the file has been loaded
        #get the sim name
        line = f.readline()
        data = line.split(":")
        self.name = data[1]
        #get the base path
        line = f.readline()
        data = line.split(":")
        base_path = data[1]
        if(base_path != self.path):
            base_path = self.path
        #get the simulation seperator
        line = f.readline()
        data = line.split(":")
        data = data[1].strip("\n")
        if(data == "Windows"):
            self._sep = "\\"
        else:
            self._sep = "/"
        #get the time range
        line = f.readline()
        data = line.split(":")
        data = data[1].split(",")
        start_time = float(data[0])
        end_time = float(data[1])
        time_step = float(data[2])
        #get the gradient names
        line = f.readline()
        data = line.split(":")
        data = data[1].strip("\n")
        data = data.split(",")
        if(data[0] == "" or data[0] == " "):
            self._grad_names = []
        else:
            self._grad_names = data
        #now for all the ranges in time loop over the list
        num_times = int((end_time - start_time) / time_step) + 1 #inclusive
        for i in range(0, num_times):
            #read the line
            line = f.readline()
            if(line != ""):
                #get the time
                data = line.split(",")
                time = float(data[0])
                #get the network path
                n_path = data[1]
                #make sure no new line characters on this'
                n_path = n_path.strip("\n")
                #rebuild the path
                paths = n_path.split(self._sep)
                #just take the last part
                path = paths[-1]
                #then add it to the path
                n_path = self.path + self.__sep + path
                #associate it with the time
                self._networks[time] = n_path
                #get all the gradient paths
                grad_paths = dict()
                for j in range(0, len(self._grad_names)):
                    #add this path to the grad path list
                    g_path = data[j+2].strip("\n")
                    #rebuild the path
                    paths = g_path.split(self._sep)
                    #just take the last part
                    path = paths[-1]
                    #then add it to the path
                    g_path = self.path + self.__sep + path
                    grad_paths[self._grad_names[j]] = g_path
                self._gradients[time] = grad_paths
        #once this is done close this file
        f.close()

    def get_raw_agent_data(self):
        """ Return all fo the networks X objects at once in an ordered list
        """
        time_series = []
        for i in range(0, len(self._networks.keys())):
            time = self._networks.keys()[i]
            print(time)
            tp  = self.get_time_point(time)
            time_series.append(tp)
        return time_series

    def get_raw_gradient_data(self):
        """ Returns all of the gradient data at once in an ordered list
        """

    def get_times(self):
        return self._networks.keys()

    def get_gradients(self):
        """ Returns a list of all the gradient names
        """
        return self._grad_names
        
    def get_time_point(self, time):
        """ Loads the specific network from the time point and returns it
            returns - network file
        """
        return nx.read_gpickle(self._networks[time])

    def get_gradient_at_time(self, time, name):
        """ Loads the gradient data specified by the specific time
            and name
        """
        print(self._gradients[time][name])
        f = open(self._gradients[time][name], "r")
        g =  pickle.load(f)
        f.close()
        return g

    def convert_to_data_base(self):
        """ 
        """
        #use can use the dict method here

    def __eq__(self, other):
        """ Handles the equal operator for the object
        """
        if(isinstance(other, TimeSeries)):
            return self.name == other.name
        #otherwise
        return False

    def __hash__(self):
        """ Handles the hashing operator for the object
        """
        return hash(self.name)
        
        

#Create a new simulation class
class Simulation(object):
    """ A class for running simulations
    """

    def __init__(self, name, path, start_time, end_time, time_step):
        """ Initialization function for the simulation setup.
            name - the simulation name (string)
            path - the path to save the simulation information to (string)
            start_time - the start time for the simulation (float)
            end_time - the end time for the simulation (float)
            time_step - the time step to increment the simulation by (float)
        """
        #set the base parameters
        #do some basic type checking
        if(type(name) is str):
            self.name = name
        else:
            self.name = repr(name)
        if(type(path) is str):
            self.path = path
        else:
            self.path = repr(path)
        #now convert all fo these to float
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.time_step = float(time_step)

        #make a list to keep track of the sim objects
        self.objects = []
        #also the gradients
        self.gradients = []
        self._gradients_by_name = dict()
        #keep track of the fixed constraints
        self._fixed_constraints = nx.Graph()
        self.network = nx.Graph()
        #add the add/remove buffers
        self._objects_to_remove = []
        self._objects_to_add = []

        #also keep track of the current sim object ID
        self._current_ID = 0

        #keep track of the fuile sepeartor to use
        if(platform.system() == "Windows"):
            #windows
            self._sep = "\\"
        else:
            #linux/unix
            self._sep = "/"

    def add(self, sim_object):
        """ Adds the specified object to the list
        """
        if(isinstance(sim_object, SimulationObject)):
            self.objects.append(sim_object)
            #also add it to the network
            self.network.add_node(sim_object)
            #increment the current ID
            self._current_ID += 1
        
    def remove(self, sim_object):
        """ Removes the specified object from the list
        """
        self.objects.remove(sim_object)
        #remove it from the network
        self.network.remove_node(sim_object)
        #also remove it fomr the fixed network
        try:
            self._fixed_constraints.remove_node(sim_object)
        except nx.NetworkXError:
            pass

    def add_object_to_addition_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be added to the simulation at the end of
            the update phase.
        """
        self._objects_to_add.append(sim_object)
        #increment the sim ID
        self._current_ID += 1

    def add_object_to_remnoval_queue(self, sim_object):
        """ Will add an object to the simulation object queue
            which will be removed from the simulation at the end of
            the update phase.
        """
        #place the obejct in the removal queue
        self._objects_to_remove.append(sim_object)

    def add_gradient(self, gradient):
        """ Adds gradients to the simulation
        """
        if(isinstance(gradient, Gradient)):
            self.gradients.append(gradient)
            self._gradients_by_name[gradient.name] = gradient

    def remove_gradient(self, gradient):
        """ Removes a gradient form the simulation
        """
        self.gradients.remove(gradient)
        del self._gradients_by_name[gradient.name]

    def add_fixed_constraint(self, obj1, obj2):
        """ Adds a fixed immutable constraint between two objects which is
            processed with the other optimization constraints
        """
        self._fixed_constraints.add_edge(obj1, obj2)

    def remove_fixed_contraint(self, obj1, obj2):
        """ Removes a fixed immutable constraint between two objects
        """
        self._fixed_constraints.remove_edge(obj1, obj2)

    def get_ID(self):
        """ Returns the current unique ID the simulation is on
        """
        #return the next avilable ID
        return self._current_ID

    def get_gradient_by_name(self, name):
        """ Returns the specific gradient by it's name
        """
        return self._gradients_by_name[name]
        
    def run(self):
        """ Runs the simulation until either the stopping criterio is reached
            or the simulation time runs out.
        """
        self.time = self.start_time
        #try to make a new directory for the simulation
        try:
            os.mkdir(self.path + self._sep + self.name)
        except OSError:
            #direcotry already exsists... overwrite it
            print("Direcotry already exsists... overwriting directory")
        #create the simulation header file
        self._create_header_file()
##        #save the inital state configuration
##        self.save()
        #set up the gradients
        self._set_gradient_inital_conditions()
        #also get the number of nodes
        #now run the loop
        while(self.time <= self.end_time and len(self.objects) < 60000):
            print("Time: " + repr(self.time))
            print("Number of objects: " + repr(len(self.objects)))
            
            #Update the objects and gradients
            self.update()
            #remove/add any objects
            self.update_object_queue()
            #perform physcis
            self.collide()
            #now optimize the resultant constraints
            self.optimize()
            #increment the time
            self.time += self.time_step
            #save 
            self.save()
        #once the sim is done close the header file
        self._header.flush()
        self._header.close()

    def update_object_queue(self):
        """ Updates the object add and remove queue
        """
        print("Adding " + repr(len(self._objects_to_add)) + " objects...")
        print("Removing " + repr(len(self._objects_to_remove)) + " objects...")
        for i in range(0, len(self._objects_to_remove)):
            self.remove(self._objects_to_remove[i])
        for i in range(0, len(self._objects_to_add)):
            self.add(self._objects_to_add[i])
        #then clear these lists
        self._objects_to_remove = []
        self._objects_to_add= []
        

    def update(self):
        """ Updates all of the objects in the simulation
        """
        #define the sinks and sources
        print("Updating gradients...")
        if(len(self.gradients) > 0):
            sources, sinks = self._define_sinks_and_sources()
            #these are defined four an entire 1 hour time step
        #update the gradients
        for i in range(0, len(self.gradients)):
            self.gradients[i].update(60*60, sinks[i], sources[i])   
            self.gradients[i].cubic_interpolate_at_object_locations(self.objects)
            print(self.gradients[i])
            print("Sources: " + repr(np.max(sources[i])) + " , " +repr(np.min(sources[i])))
            print("Sinks: " + repr(np.max(sinks[i])) + " , " + repr(np.min(sinks[i])))
        #finally interpolate over all the object positions to solve for the gradient
        #at those speicif cloations
        
        #update the sim objects 
        print("Updating objects...")
        #RUN A FASTER UPDATE LOOP
        split = 1
        for j in range(0, split):
            dt = self.time_step / float(split)
            for i in range(0, len(self.objects)):
                self.objects[i].update(self, dt)
                
    def collide(self):
        """ Handles the collision map generation 
        """
        #first make a list of points for the delaunay to use
        points = np.zeros((len(self.objects), 3))
        index_to_object = dict()
        for i in range(0, len(self.objects)):
            #Now we can add these points to the list
            points[i] = self.objects[i].location
        #now perform the nearest neghbor assessment by building a delauny triangulation
        tree = KDTree(points)
        #keep track of this as a network
        self.network = nx.Graph()
        #add all the simobjects
        self.network.add_nodes_from(self.objects)
        #add the edge if it is less than our interaction length
        for i in range(0, len(self.objects)):
            obj1 = self.objects[i]
            dist = obj1.get_max_interaction_length()
            indicies = tree.query_ball_point(points[i], dist)
            for j in range(0, len(indicies)):
                ind = indicies[j]
                obj2 = self.objects[ind]
                #now add these to the edges
                self.network.add_edge(obj1, obj2)

    def optimize(self):                
        #apply constraints from each object and update the positions
        #keep track of the global col and opt vectors
        opt = 2
        col = 2
        fixed = 2
        itrs = 0
        max_itrs = 10
        avg_error = 1.0 # um
        while ((opt + col + fixed) >= avg_error and itrs < max_itrs):
            #reset these values
            opt = 0
            col = 0
            fixed = 0
            #handle the spring constraints
            opt, col = self._handle_spring_constraints()
            #handle to fixed constraints
            fixed = self._handle_fixed_constraints()
            #now loop over and update all of the constraints
            for i in range(0, len(self.objects)):
                #update these collision and optimization vectors
                #as well as any other constraints
                self.objects[i].update_constraints(self.time_step)
            #increment the itrations
            itrs += 1
            #print the results
            print(itrs)
            print(opt, col, fixed)

    def _set_gradient_inital_conditions(self):
        """ Defines the masks to specify the gradient intal conditions
        """
        cent = self.get_center()
        for i in range(0, len(self.gradients)):
            #make a mask the same size as the gradient
            shp = self.gradients[i].shape()
            print(shp)
            mask = np.zeros(shp)
            for j in range(0, len(self.objects)):
                agent = self.objects[j]
                #get the object position
                pos = agent.location
                #get the difference from the center
                dist_vec = SubtractVec(pos, cent)
                #convert this to a index value
                #for all of the gradients get the coeffients
                x, y, z = self.gradients[i].get_object_position_on_grid(dist_vec)
                mask[x,y,z] = 1
            #now pass this in to the set inital values function
            self.gradients[i].set_initial_conditions(mask)

    def _handle_spring_constraints(self):
        """
        """
        col = 0
        opt = 0
        edges = self.network.edges()
        for i in range(0, len(self.network.edges())):
            #for each edge optimize the spring interaction
            edge = edges[i]
            #get the objects
            obj1 = edge[0]
            obj2 = edge[1]
            if(obj1.owner_ID != obj2.owner_ID):
                #that is to say these are NOT internal constraints pciked up
                #by the nearest neighbor approach
                v12 = SubtractVec(obj2.location, obj1.location)
                dist = Mag(v12)
                #also compute the normal
                norm = NormVec(v12)
                #get the object radii
                r_sum = obj1.radius + obj2.radius
                #check for a collision
                if(r_sum >= dist):
                    #then apply the collision
                    d = -norm*((r_sum-dist)/2.0)
                    obj1.add_fixed_constraint_vec(d)
                    obj2.add_fixed_constraint_vec(-d)
                    #add this to the collision vec
                    col += Mag(d)*2
                #apply the spring
                #also get the normal interaction length
                l1 = obj1.get_interaction_length()
                l2 = obj2.get_interaction_length()
                #now figure out how far off the connection length  is
                #from that distance
                dist = dist - (l1 + l2)
                #now get the spring constant strength
                k1 = obj1.get_spring_constant(obj2)
                k2 = obj2.get_spring_constant(obj1)
                k = min(k1, k2)
                #now we can apply the spring constraint to this
                dist = (dist/2.0) * k
                #make sure it has the correct direction
                temp = ScaleVec(norm, dist)
                #add these vectors to the object vectors
                obj1.add_displacement_vec(temp)
                obj2.add_displacement_vec(-temp)
                #add to the global opt vec
                opt += Mag(temp)
        #return the average opt and col values
        opt = opt / (len(self.network.edges())*2.0)
        col = col / (len(self.network.edges())*2.0)
        return opt, col

    def _handle_fixed_constraints(self):
        """
        """
        error = 0
        edges = self._fixed_constraints.edges()
        for i in range(0, len(edges)):
            #for each edge optimize the spring interaction
            edge = edges[i]
            #get the objects
            obj1 = edge[0]
            obj2 = edge[1]
            dist_vec = SubtractVec(obj2.location, obj1.location)
            dist = Mag(dist_vec)
            #also compute the normal
            norm = NormVec(dist_vec)
            #get the object radii
            r_sum = obj1.radius + obj2.radius
            #then apply the collision
            d = -norm*((r_sum-dist)/2.0)
            obj1.add_fixed_constraint_vec(d)
            obj2.add_fixed_constraint_vec(-d)
            #add this to the collision vec
            error += Mag(d)*2
        #calculate the average error
        try:
            error = error / len(self._fixed_constraints.edges())
        except ZeroDivisionError:
            error = 0 
        return error

    def _define_sinks_and_sources(self):
        """ Defines the sinks and soruce terms from the simulation
            network by figuring out where each agent in the network lies
            and defining the subsequent diffusion procedures.
            returns - a list of (sink, source) as a tuple for each gradeint
        """
        #make a set of sink and source arrays which match
        #the size of each gradient
        sources = []
        sinks = []
        for i in range(0, len(self.gradients)):
            shp = self.gradients[i].shape()
            source = np.zeros(shp)
            sink = np.zeros(shp)
            sources.append(source)
            sinks.append(sink)
        #get the center of the network
        cent = self.get_center()
        #now loop over all of the agents in the simulation
        for j in range(0, len(self.objects)):
            agent = self.objects[j]
            #get the object position
            pos = agent.location
            #get the difference from the center
            dist_vec = SubtractVec(pos, cent)
            #convert this to a index value
            #for all of the gradients get the coeffients
            for i in range(0, len(self.gradients)):
                x, y, z = self.gradients[i].get_object_position_on_grid(dist_vec)
                agent.set_gradient_location(self.gradients[i].name, (x,y,z))
                #see if the object conatins a value for this gradient
                src, snk = agent.get_gradient_source_sink_coeff(self.gradients[i].name)
                sources[i][int(x),int(y),int(z)] += src
                sinks[i][int(x),int(y),int(z)] += snk
                
        #and return the list
        return sources, sinks
                    
    def get_center(self):
        """ Returns the center of the simulation
            return - point in the form of (x,y,z)
        """
        cent = (0,0,0)
        for i in range(0, len(self.objects)):            
            cent = AddVec(self.objects[i].location, cent)
        #then scale the vector
        cent = ScaleVec(cent, 1.0/ len(self.objects))
        #and return it
        return cent
    
    def _create_header_file(self):
        """ Creates a simulation header file which will save all the relevant
            information in text format to the simulation file.
        """
        #first create the file
        temp_path = self.path + self._sep + self.name + self._sep + "info.sim"
        self._header = open(temp_path, "w")
        #now save the simulation name
        self._header.write("Name:" + self.name + "\n")
        #write the path
        self._header.write("Base path:" + self.path + "\n")
        #write the sepeartor info
        self._header.write("Platform:" + platform.system() + "\n")
        #write the time info
        self._header.write("Time range:" + repr(self.start_time) + ",")
        self._header.write(repr(self.end_time) + "," + repr(self.time_step) + "\n")
        #Write the gradient names
        self._header.write("Gradient Species:")
        for i in range(0, len(self.gradients)):
            self._header.write(self.gradients[i].name)
            if(i != len(self.gradients)-1):
                self._header.write(",")  
        #Now the header file is open and defined
        self._header.write("\n")  
        
    def save(self):
        """ Saves the simulation snapshot.
        """
        #Write the current itme into the header file
        self._header.write(repr(self.time))
        #get the base path
        base_path = self.path + self.name + self._sep
        #First save the network files
        n_path = base_path + "network" + repr(self.time) + ".gpickle"
        nx.write_gpickle(self.network, n_path)
        #now write that path to the file
        self._header.write("," + n_path)
        #Then save the gradient files
        for i in range(0, len(self.gradients)):
            grad_path = self.gradients[i].save(base_path, repr(self.time))
            self._header.write("," + grad_path)
        #put the final new line character
        self._header.write("\n")
        #fluish the file
        self._header.flush()

