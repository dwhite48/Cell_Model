from Simulation import *
import sys
from scipy.spatial.distance import euclidean
import random as r
from StemCellNN import *
from Gradient import Gradient
import networkx as nx
import simAnaylsisScriptOct4Nanog as sa
import platform

#runs from the command line and takes arguments in the follwing format
#-0
#-1 structure sim save file path
#-2 simulation base save file path
#-3 simulation number
#-4 start time
#-5 end time
#-6 time step

if(__name__ == '__main__'):
    args = sys.argv
    #if these arguments are not set throw an error
    if(len(args) < 6):
        raise TypeError('6 arguments required')
    #else set them correctly
    struct_path = str(args[1])
    sim_base_path = str(args[2])
    sim_id = int(args[3])
    ts = float(args[4])
    te = float(args[5])
    dt = float(args[6])
    p1 = float(args[7])
    p2 = float(args[8])
    p3 = float(args[9])

##    ts = 0
##    te = 100
##    dt = 1
##    sim_id = 9
##    sim_base_path = "C:\\users\\doug\\desktop\\"
##    struct_path = sim_base_path + "struct 100\\"
##    struct_path = sim_base_path + "structs\\EB\\1000\\25\\"
##    p1 = .001
##    p2 = .06
##    p3 = 10
##    p4 = .1
##    p5 = 1000.
####    p1 = .001
##    p1 = 0.000
####    p2 = .007
####    p2 = 0.007
##    p2 = 0.0075
##    p3 = 10
##    p4 = 5E-4
##    p5 = 10.0

    #Now make a new simulation
    sim = Simulation(sim_id,
                     sim_base_path,
                     ts,
                     te,
                     dt)
    #add some gradients
##    #from Van Winkle et al 2012 for 02
##    D = 1.7e-9 / (1e-12) #um^2/sec
##    consump_rate = (4.0e-17)#*(10**6) mols/cell/sec #umols/cell/sec
##    #This value later gets normalized using the 
##    #set the outside concentration
##     outside_c = 1.04e-4*10**6 #(umol/L) or uM
    #from War-tenberg et al., 2001

    #find the only file in the dir
    f = os.listdir(struct_path)

    #then load this file
    #open the input structure file
    net = nx.read_gpickle(struct_path + f[0])
    cent = (0,0,0)
    placed = False
    nodes = net.nodes()
    np.random.shuffle(nodes)
    id_map = dict()
    for i in range(0, len(nodes)):
        node = nodes[i]
        #randmoize the division set
        #so that cells divide at different times
        div_set = r.random()*19
        #now make a new stem cell
        sim_obj = StemCell(node.location, node.radius, node.ID,
                           "U", division_set = div_set,
                           params = [p1, p2, p3, p4, p5])
        #keep track fo the ID dict for the connection mapping
        id_map[node.ID] = sim_obj
        #add it to the sim
        sim.add(sim_obj)

    #also import the connections to use as well
    cons = net.edges()
    for i in range(0, len(cons)):
        con = cons[i]
        n1, n2 = con
        s1 = id_map[n1.ID]
        s2 = id_map[n2.ID]
        sim.network.add_edge(s1, s2)
    #Run the simulation
    sim.run()
    #set the seperator
    if(platform.system() == "Windows"):
        #windows
        sep = "\\"
    else:
        #linux/unix
        sep = "/"   
    
    #analyze the data
    print(sim_base_path)
    sa.get_simulation_metric_data(sim_base_path + sep + repr(sim_id) + sep)

