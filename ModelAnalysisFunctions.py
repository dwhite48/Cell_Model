import Data
import os, sys

def create_culled_data_set(d, count = 10):
    """ Creats a culled data set based on the count parameter
        count - keeps data every count
        returns a new culled data set 
    """
    #then delete every data point except the one marked by count
    ml = d.get_measurement_list()
    sl = d.get_sample_list()
    #make a new data set
    d2 = data()
    #save all the samples
    for i in range(0, len(sl)):
        d2.add_sample(sl[i])
    #Now begin remving samples
    for i in range(0, len(ml)):
        if(i % count == 0):
            #copy this point over
            info = d.get_measurement(ml[i])
            #add this as a measurement
            d2.add_measurement(ml[i])
            for j in range(0, len(sl)):
                d2.set_data(sl[j], ml[i], info[j])
    #Now print these and save as well for visualization
    #save the heat maps
    d2.save_heat_map(path + "culled_")
    d2.cluster_data()
    d2.save_clustered_heat_map(path + "culled_")
    #return the new data set
    return d2

def get_model_trace_by_sim_id(d, sim_id):
    """ Gets all timepoints associated with the sim_ID and returns
        a trace with the timepoints correctly ordered
    """
    #get the
    ms = d.get_measurement_list()
    sl = d.get_sample_list()
    new_data = np.zeros((25, len(ms)))
    for i in range(0, len(sl)):
        #split the name to get the sample ID
        sim_index = int(sl[i].split("_")[0])
        time = float(sl[i].split("_")[1])
        #now figure out the time indes
        time_index = (time - 1.0)/6.0
        if(sim_index == sim_id):
            #this data can be kept. Also get the second value
            #which tells the time
            #now get all data associated with this row
            temp = d.get_sample(sl[i])
            new_data[time_index, :] = temp
    #return the array
    return new_data

def graph_pca_traces(d, n_dims, n_traces, save_path):
    """ Graphs traces of the PCA data
    """
    labels = calculate_true_labels(d)
    #first get the PCA transform on the whole data
    dat = d.get_all_data()
##    x_min, x_max = dat[:, 0].min()-.1, dat[:, 0].max()+.1
##    y_min, y_max = dat[:, 1].min()-.1, dat[:, 1].max()+.1
    pca = d.PCA(save_path, n_dims, labels, graphing = False)
    #Now create a culled version of the data set by sample ID
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,1,0), (1,0,1), (0,1,1),
              (.5,0,0), (0,.5,0), (0,0,.5),
              (.5,.5,0), (.5,0,.5), (0,.5,.5)]

    markers = ['o', 's', 'D', '^', 'v',
               '*', '>', '.', 'p', '<']
    plt.cla()
    plt.clf()
    for i in range(1, n_traces+1):
        new_data = get_model_trace_by_sim_id(d, i)
        #scale the new data
        new_data = scale(new_data)
        #apply the PCA transform to this
        trans = pca.transform(new_data)
        #get the color
        col = colors[(i-1) % len(colors)]
        mark = markers[(i-1) % len(markers)]
##        for j in range(0, 24):
##            mark = markers[(j) % len(markers)]
##            #Now graph this data
        plt.plot(trans[:, 0], trans[:, 1],
                 markerfacecolor = col,
                 color = col,
                 markeredgecolor = 'k',
                 marker = mark)
##    plt.xlim(-4, 8)
##    plt.ylim(-3, 10)
    plt.savefig(save_path)
    plt.show()

def graph_avg_pca_traces(d, n_dims, save_path, n_traces = 50):
    """ Graphs traces of the PCA data
    """
    labels = calculate_true_labels(d)
    #first get the PCA transform on the whole data
    pca = d.PCA(save_path, n_dims, labels, graphing = False)
##    dat = d.get_all_data()
##    x_min, x_max = dat[:, 0].min()-.1, dat[:, 0].max()+.1
##    y_min, y_max = dat[:, 1].min()-.1, dat[:, 1].max()+.1
    #Now create a culled version of the data set by sample ID
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,1,0), (1,0,1), (0,1,1),
              (.5,0,0), (0,.5,0), (0,0,.5),
              (.5,.5,0), (.5,0,.5), (0,.5,.5)]
    plt.cla()
    plt.clf()
    xs = []
    ys = []
    for i in range(1, n_traces+1):
        new_data = get_model_trace_by_sim_id(d, i)
        new_data = scale(new_data)
        #apply the PCA transform to this
        trans = pca.transform(new_data)
##        for j in range(0, 24):
##            mark = markers[(j) % len(markers)]
##            #Now graph this data
        plt.plot(trans[:, 0], trans[:, 1],
                 color = [.5,.5,.5, .25],
                 marker = 'o')
        #append these to the x and y list
        xs.append(trans[:, 0])
        ys.append(trans[:, 1])
    #Now at the end compute the average of these
    avg_x = np.average(xs, axis = 0)
    avg_y = np.average(ys, axis = 0)
    plt.plot(avg_x, avg_y,
             color = [1,0,0],
             linewidth = 2,
             marker = 's')
    #save it
##    plt.xlim(-4, 8)
##    plt.ylim(-3, 10)
    plt.savefig(save_path)
    plt.show()

def compile_and_resample(rs_factor, path):
    """ Will compile all of the data for a given simulation run and save
        all of it including individual metrics trends. Will also return
        a saved version of the compiled data
        return - compiled data object
    """
    #keep track of all the data
    all_data = []
    #first list all of the dirs
    dirs = os.listdir(path)
    num_sims = 0#keep track of the names
    names = []
    for i in range(0, len(dirs)):
        d = dirs[i]
        if(os.path.isdir(path + d)):
            #find the data.txt and load this form a file
            info = Data.data()
            loaded = True
            try:
                info.load(path + d + "\\data.txt")
            except IOError:
                loaded = False
            print(path + d, loaded)
            if(loaded):
                #append this to the data list
                all_data.append(info)
                names.append(d)
                #increment this count
                num_sims += 1

    #Now for each of the data sets only take every r_s timepoint
    d = Data.data()
    #add all fo the measurements to this
    print('Copying measurements...')
    ml = all_data[0].get_measurement_list()
    for i in range(0, len(ml)):
        d.add_measurement(ml[i])
        print(ml[i])
    for i in range(0, num_sims):
        #get the data object
        info = all_data[i]
        #get all its samples (ie.e time-points)
        sl = info.get_sample_list()
        #add these measurements with the sample ID
        for j in range(0, len(sl)):
            #gte it measurement list
            ml = info.get_measurement_list()
            #see if it matches the resampleing factor
            if(j % rs_factor == 0):
                #get the measurement associated with this data
                ms = info.get_sample(sl[j])
                #get the name
                name = names[i] + "_" + sl[j]
##                print("Processing...")
##                print(name)
                #add this as a sample
                d.add_sample(name)
                #Now for all of these measurements,
                #add them to the new data set
                for k in range(0, len(ml)):
                    d.set_data(name, ml[k], ms[k])
    #save the heat map
    d.save(path + "compiled_data.txt")
    #normalize
    d.normalize_measurements()
    #tranpose the measurements
    d.transpose_measurments_and_samples()
    d.save_heat_map(path)
    #cluster
    d.save_clustered_heat_map(path)
    #return it
    return d

def jagged_compute_avg(jagged):
    """ Compute the average values along a jagged array
    """
    avgs = []
    #first find the max length
    mx = 0
    for i in range(0, len(jagged)):
        mx = max(mx, len(jagged[i]))
    #now compute the average along this length
    for i in range(0, mx):
        tsum = 0
        count = 0
        #check to see if a value is needed
        for j in range(0, len(jagged)):
            #if a vlaue still exsists
            if(len(jagged[j]) > i):
                count += 1
                tsum += jagged[j][i]
        avg = tsum / count
        #add it to the list
        avgs.append(avg)
    #return this value
    return avgs, mx

##def process(path):
##    """ Searches the given path to find all valid simulations assuming
##        that all direcotries in the path are simulations
##    """
##
##    for p in os.listdir(path):
        
