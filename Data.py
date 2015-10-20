import sys, os
import numpy as np
import networkx as nx
#saving objects to pickled objects
import pickle
#visualzation and drawing imports
from PIL import Image, ImageDraw, ImageFont, ImageGrab, ImageOps
import visual as vis
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#base modules
import math as math
import time as t_m
import operator
#custom classes
from Point import vector
from Colormaps import *
#skleanr - machine learning and clustering imports
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, Ward, DBSCAN
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import metrics
from scipy.cluster.hierarchy import *
from sklearn.grid_search import GridSearchCV
#linear models
from sklearn.linear_model import LinearRegression, RidgeCV, LassoLars
from sklearn.linear_model import ElasticNetCV, OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge, ARDRegression, SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, LogisticRegression
from sklearn.linear_model import Lars, Lasso
#svms
from sklearn.svm import SVC, NuSVC, LinearSVC
#nearest neighbor
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
#misc classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
#cross validation tools
from sklearn.grid_search import GridSearchCV
#use feature selection
from sklearn.feature_selection import SelectPercentile, f_classif
#outlier removal
from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope

#classes for assigning data
class data(object):
    """ Class for keeping track of data
    """

    def __init__(self, sl = None, ml = None, values = None):
        """ Constructor for the data class
        """
        self.__sample_list = dict()
        self.__measurement_list = dict()
        if(sl == None and ml == None and values == None):
            self.__sample_count = 0
            self.__measurement_count = 0
            self.__data = np.zeros((0,0), dtype = np.float64)
        else:
            #create the measurement and sample dicts
            for i in range(0, len(sl)):
                if(type(sl[i]) == type("")):
                    self.__sample_list[sl[i]] = i
                else:
                    self.__sample_list[repr(sl[i])] = i
            for i in range(0, len(ml)):
                if(type(ml[i]) == type("")):
                    self.__measurement_list[ml[i]] = i
                else:
                    self.__measurement_list[repr(ml[i])] = i
            #Update the counts
            self.__sample_count = len(sl)
            self.__measurement_count = len(ml)
            #Now add all the data in
            self.__data = np.zeros((self.__sample_count,
                                    self.__measurement_count),
                                   dtype = np.float64)
            for i in range(0, len(sl)):
                for j in range(0, len(ml)):
                    self.__data[i,j] = values[i][j]
                    
        #set the color map values up
        vals = {0:[0,0,0,1], 1:[1,0,0,1]}
        self._default_color_map = colormap(vals)

    def save(self, file_name):
        """ save the data to the file
        """
        f = open(file_name, "w")
        #save all the variable names IN ORDER
        #sort the dicationary by the index values
        sorted_samples = sorted(self.__sample_list.iteritems(),
                                key = operator.itemgetter(1))
        for i in range(0, self.__sample_count):
            keys = sorted_samples
            f.write(keys[i][0])
            if(i < self.__sample_count - 1):
                f.write(",")
        f.write("\n")
        #save all of the measurement names
        #sort the dicationary by the index values
        sorted_measur = sorted(self.__measurement_list.iteritems(),
                                key = operator.itemgetter(1))
        for i in range(0, self.__measurement_count):
            keys = sorted_measur
            f.write(keys[i][0])
            if(i < self.__measurement_count - 1):
                f.write(",")
        f.write("\n")
        #now save the actual data
        for i in range(0, self.__sample_count):
            for j in range(0, self.__measurement_count):
                f.write(repr(self.__data[i,j]))
                if(j < self.__measurement_count - 1):
                    f.write(",")
            f.write("\n")
        f.flush()
        f.close()
        
    def load(self, file_name):
        """ load the data to the file
        """
        f = open(file_name, "r")
        #get the first line
        line = f.readline()
        data = line.split(",")
        #corresonpnds to the names
        for i in range(0, len(data)):
            #strip the \n
            data[i] = data[i].strip("\n")
            self.__sample_list[data[i]] = i
        self.__sample_count = len(data)
        #next line
        line = f.readline()
        data = line.split(",")
        #corresonpnds to the names
        for i in range(0, len(data)):
            #strip the \n
            data[i] = data[i].strip("\n")
            self.__measurement_list[data[i]] = i
        self.__measurement_count = len(data)
        #Now read the data
        self.__data = np.zeros((self.__sample_count, self.__measurement_count))
        #now read the actual data
        for i in range(0, self.__sample_count):
            line = f.readline()
            data = line.split(",")
            for j in range(0, len(data)):
                val = float(data[j])
                self.__data[i,j] = val
        #all data is read close the file
        f.close()

    def get_all_data(self):
        """ Returns a copy of the current data as an array
        """
        return np.copy(self.__data)
        
    def add_sample(self, name):
        """ Adds a new sample catagory to the data matrix
        """
        #check if this is in the list already
        if(name in self.__sample_list.keys()):
            #do not add it
            pass
        #other wise add it
        else:
            self.__sample_list[name] = self.__sample_count
            #increment the sample count
            self.__sample_count += 1
            #also increase the array size in terms of the rows
            temp = np.resize(self.__data,
                            (self.__sample_count,
                            self.__measurement_count))
            temp[0:self.__sample_count-1, 0:self.__measurement_count] = self.__data
            self.__data = temp
            #set the last line to zeros
            self.__data[self.__sample_count-1, :] = 0

    def rename_sample(self, old_name, new_name):
        """ Renames a sample
        """
        self.__sample_list[new_name] = self.__sample_list[old_name]
        del self.__sample_list[old_name]

    def rename_measurement(self, old_name, new_name):
        """ Renames a measurement
        """
        self._measurement_list[new_name] = self._measurement_list[old_name]
        del self._measurement_list[old_name]

            
    def add_measurement(self, name):
        """ Adds a new measurement catagory to the data matrix
        """
        if(not isinstance(name, str)):
            name = repr(name)
        #check if this is in the list already
        if(name in self.__measurement_list.keys()):
            #do not add it
            pass
        #other wise add it
        else:
            self.__measurement_list[name] = self.__measurement_count
            #increment the sample count
            self.__measurement_count += 1
            #also increase the array size in terms of the rows
            temp = np.resize(self.__data,
                            (self.__sample_count,
                            self.__measurement_count))
            temp[0:self.__sample_count, 0:self.__measurement_count-1] = self.__data
            self.__data = temp
            self.__data[:, self.__measurement_count-1] = 0

    def remove_measurement(self, name):
        """ Adds a new measurement catagory to the data matrix
        """
        #check if this is in the list already
        #get the list
        lm = self.get_measurement_list()
        ind = lm.index(name)
        if(ind != -1):
            #remove this column of the array
            self.__data = np.delete(self.__data, ind, 1)
            #remove it from the measurement list
            del self.__measurement_list[name]
            #decrement the count
            self.__measurement_count -= 1
            #remove it from the return list
            lm.remove(name)
            #now update the list
            for i in range(ind, len(lm)):
                #update the respective dictionary by moving the indicies
                #down by one
                self.__measurement_list[lm[i]] -= 1

    def remove_sample(self, name):
        """ Adds a new measurement catagory to the data matrix
        """
        #check if this is in the list already
        #get the list
        ls = self.get_sample_list()
        try:
            ind = ls.index(name)
        except ValueError:
            ind = -1
        if(ind != -1):
            #remove this column of the array
            self.__data = np.delete(self.__data, ind, 0)
            #remove it from the measurement list
            del self.__sample_list[name]
            #decrement the count
            self.__sample_count -= 1
            #remove it from the return list
            ls.remove(name)
            #now update the list
            for i in range(ind, len(ls)):
                #update the respective dictionary by moving the indicies
                #down by one
                self.__sample_list[ls[i]] -= 1
                
    def get_sample_list(self):
        """ returns the list of all the samples
        """
        #sort the dicationary by the index values
        sorted_samples = sorted(self.__sample_list.iteritems(),
                                key = operator.itemgetter(1))
        temp = []
        for i in range(0, self.__sample_count):
            keys = sorted_samples
            try:
                temp.append(keys[i][0])
            except IndexError:
                print(i)
        return temp

    def get_measurement_list(self):
        """ returns the list of all the samples
        """
        sorted_measur = sorted(self.__measurement_list.iteritems(),
                        key = operator.itemgetter(1))
        temp = []
        for i in range(0, self.__measurement_count):
            keys = sorted_measur
            temp.append(keys[i][0])
        return temp    
            
    def get_sample(self, name):
        """ Returns all the variables associated with that sample
        """
        #get the index from the list
        index = self.__sample_list[name]
        #now return the data associated with that list
        return self.__data[index, :]

    def get_measurement(self, name):
        """ Returns all the sample values associated with this variable
        """
        #get the index from the list
        index = self.__measurement_list[name]
        #now return the data associated with that list
        return self.__data[:, index]

    def get_data(self, sample_name, variable_name):
        """ Returns the data at the sample and variable specified
        """
        i1 = self.__sample_list[sample_name]
        i2 = self.__measurement_list[variable_name]
        return self.__data[i1, i2]

    def set_data(self, sample_name, variable_name, value):
        """ Sets the data at the sample and variable specified to the value
        """
        i1 = self.__sample_list[sample_name]
        i2 = self.__measurement_list[variable_name]
        self.__data[i1, i2] = value

    def unit_variance_mean_center_scaling(self):
        """ Scales the data to be centered around 0 with unit variance
        """
        self.__data = preporocessing.scale(self.__data)

    def normalize_measurements(self):
        """ Normalize all of the measurements
        """
        for i in range(0, self.__measurement_count):
            self.__data[:, i] = self.__normalize_data(self.__data[:, i])

    def normalize_samples(self):
        """ Normalize all of the samples
        """
        for i in range(0, self.__sample_count):
            self.__data[i, :] = self.__normalize_data(self.__data[i, :])
            
    def transpose_measurments_and_samples(self):
        """ Transposes the meausrements and samples
        """
        sl = self.__sample_list
        ml = self.__measurement_list
        sc = self.__sample_count
        mc = self.__measurement_count
        #Now switch these
        self.__sample_list = ml
        self.__measurement_list = sl
        self.__sample_count = mc
        self.__measurement_count = sc
        #Now transpose the data
        self.__data = np.transpose(self.__data)


    ###########################################################################
    # UNIVARIATE FEATURE SELECTION

    def select_features(self, percent, true_values):
        """ Perform feature selection keeping the percentage of features
            pased in to the function. Returns a reduced copy of the data set
            which has these features removed
        """
        selector = SelectPercentile(f_classif, percentile = percent)
        #fit this to the data
        selector.fit(self.__data, true_values)
        #now we know which data are necessary
        keep = selector.get_support()
        #Now we know which of thes eindicies to actually insert into the
        #new data set
        temp = data()
        #Only add the keeper measurements
        ml = self.get_measurement_list()
        for i in range(0, len(ml)):
            if(keep[i]):
                print("Keeping feature " + ml[i])
                temp.add_measurement(ml[i])
            else:
                print("Removing feature " + ml[i])
        sl = self.get_sample_list()
        #loop over all the samples and such
        for i in range(0, len(sl)):
            #add thi sample
            temp.add_sample(sl[i])
            for j in range(0, len(ml)):
                #only add it to the new data set if the
                #keep array is true
                if(keep[j]):
                    val = self.get_data(sl[i], ml[j])
                    temp.set_data(sl[i], ml[j], val)
        #finally return the culled feature selected data set
        return temp

    def AnnotateOutliersInDataBySample(self, value = 2.0):
        """ Calcualutes the outliers in the data and anotates them
        """
        #perofrom PCA to two deminsions
        ms = scale(self.__data)
        pca = PCA(n_components = 2)
        pca.fit(ms)
        dat = pca.transform(ms)
        #Now calculate the mean and the subsequent t-test values
        #for each to see if it falls outside the distribution
        robust_cov = EmpiricalCovariance().fit(dat)
        dists = robust_cov.mahalanobis(dat - np.mean(dat, 0)) ** (0.33)
        print(dists)
        sl = self.get_sample_list()
        for i in range(0, len(sl)):
            #if the value of dists here is greater remove this sample
            if(dists[i] > value):
                self.remove_sample(sl[i])
                print("Removing Sample: " + sl[i])
        

    ###########################################################################
    # MACHINE LEARNING - CLASSIFICATION

    # LINEAR REGRESSION
    def LinearRegression(self, save_path, true_data, regr = None):
        """ Performs linear regression on the data and will save a heat map
            representing the coeffiecnt matrix, and the goodness of fit. Takes
            an optional parameter allowing the data set to be fit with a
            pre-defined regression model...
        """
        if(regr == None):
            #create the regression model
            regr = LinearRegression()
        #otherwise use the one passed in
        print("Running Linear regression...")
        return self.__perform_regression(save_path + "linear_regression_",
                                  true_data, regr)

    def Ridge(self, save_path, true_data, regr = None):
        """ Use the ridge linear classifier to pick a model to fit the data
        """
        if(regr == None):
            #create the regression model which validates itself
            regr = RidgeCV(alphas = np.logspace(-5,5,num=22))
        #otherwise use the one passed in
        print("Running Ridge regression...")
        return self.__perform_regression(save_path + "ridge_regression_",
                                  true_data, regr)

    def Lars(self, save_path, true_data, regr = None):
        """ Use the lars linear classifier to pick a model to fit the data
        """
        if(regr == None):
            #create the regression model which validates itself
            regr = Lars()
            params = {'n_nonzero_coefs':[np.inf]}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running Lars regression...")
        return self.__perform_regression(save_path + "lars_regression_",
                                  true_data, regr)
    
    def Lasso(self, save_path, true_data, regr = None):
        """ Use the lars linear classifier to pick a model to fit the data
        """
        if(regr == None):
            #set up the base regressor
            regr = Lasso()
            params = {'alpha':np.logspace(-5,5,num=22)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running Lasso regression...")
        return self.__perform_regression(save_path + "lasso_regression_",
                                  true_data, regr)
    

    def LarsLasso(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = LassoLars()
            #set the params
            params = {'alpha':np.logspace(-5,5,num=22)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running LarsLasso regression...")
        return self.__perform_regression(save_path + "lars_lasso_regression_",
                                  true_data, regr)
        
    def ElasticNet(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = ElasticNetCV(alphas = np.logspace(-5,5,num=22))
        #otherwise use the one passed in
        print("Running ElasticNet regression...")
        return self.__perform_regression(save_path + "elastic_net_regression_",
                                  true_data, regr)
        
    def OMP(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = OrthogonalMatchingPursuitCV()
        #otherwise use the one passed in
        print("Running OMP regression...")
        return self.__perform_regression(save_path + "OMP_regression_",
                                  true_data, regr)

    def Logistic(self, save_path, true_data, regr = None):
        if(regr == None):
            #set up the base regressor
            regr = LogisticRegression(penalty = 'l1')
            params = {'C':np.logspace(-5,5,num=22)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running Logistic regression...")
        return self.__perform_regression(save_path + "Logistic_regression_",
                                  true_data, regr)

    def BayesianRidge(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = BayesianRidge(compute_score = True)
            #set up the params
            params = {'alpha_1':np.logspace(-5,5,num=11),
                      'alpha_2':np.logspace(-5,5,num=11),
                      'lambda_1':np.logspace(-5,5,num=11),
                      'lambda_2':np.logspace(-5,5,num=11)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running BayesianRidge regression...")
        return self.__perform_regression(save_path + "Bayesian_Ridge_regression_",
                                  true_data, regr)

    def ARDRegression(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = ARDRegression(compute_score = True)
                        #set up the params
            params = {'alpha_1':np.logspace(-5,5,num=11),
                      'alpha_2':np.logspace(-5,5,num=11),
                      'lambda_1':np.logspace(-5,5,num=11),
                      'lambda_2':np.logspace(-5,5,num=11)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running ARDRegression regression...")
        return self.__perform_regression(save_path + "Bayesian_ARD_regression_",
                                  true_data, regr)

    def SGD(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regression model which validates itself
            regr = SGDRegressor()
            #set up the params
            params = {'alpha':np.logspace(-10,3,num=22)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running SGD regression...")
        return self.__perform_regression(save_path + "SGD_regression_",
                                  true_data, regr)  

    def PassiveAggressive(self, save_path, true_data, regr = None):
        if(regr == None):
            #create the regressor
            regr = PassiveAggressiveRegressor(n_iter = 20)
            #set up the params
            params = {'epsilon':np.logspace(-5,5,num=22)}
            #run the optimization
            gs = GridSearchCV(regr, params)
            regr = self.__run_grid_search(true_data, gs)
        #otherwise use the one passed in
        print("Running PassiveAggressive regression...")
        return self.__perform_regression(save_path + "Passive_Agressive_regression_",
                                  true_data, regr)


    def __run_grid_search(self, true_data, grid_seacrh):
        """ Helper method to split the data and perform the grid search
        """
        #split the data to training and test sets
        #samples are the rows, measurements the columns
        x, y = self.__data.shape
        train_data = self.__data[0:x:2, :]
        test_data = self.__data[1:x:2, :]
        #split up the true values as well
        x = len(true_data)
        train_target = true_data[0:x:2].T
        test_target = true_data[1:x:2].T
        #fit it to the data
        grid_seacrh.fit(train_data, train_target)
        return grid_seacrh.best_estimator_

    def __perform_regression(self, save_path, true_data, regr):
        """ Helper method for eprforming the linear regression steps
        """
        #split the data to training and test sets
        #samples are the rows, measurements the columns
        x, y = self.__data.shape
        train_data = self.__data[0:x:2, :]
        test_data = self.__data[1:x:2, :]
        #split up the true values as well
        x = len(true_data)
        train_target = true_data[0:x:2].T
        test_target = true_data[1:x:2].T
        #fit it to the data
        regr.fit(train_data, train_target)
        #print the coefficient matrix
        print("Coefficient Matrix:")
        print(regr.coef_)
        #predict new values and compute the mean squre error
        msq = metrics.mean_squared_error(test_target, regr.predict(test_data))
        print("Mean square error: " + repr(msq))
        #get the explained variance ratio
        r_sqr = regr.score(test_data, test_target)
        print("R^2 value: " + repr(r_sqr))
        #save the regressor to a g-pickle file
        f = open(save_path + "object.gpickle", "w")
        pickle.dump(regr, f)
        f.close()
        #now return these values as well as the regression model
        return msq, r_sqr, regr.coef_

    #CLASSIFICATION

    #state vector machines
    def SVM_NuSVC(self, save_path, true_data, classifier = None):
        """ Construct a NuSVC model and fit it to the data
        """
        if(classifier == None):
            #setup the param grid list
            nus = [1E-7]
            params = [{'nu':nus, 'kernel': ['linear']},
                      {'nu':nus, 'gamma':np.logspace(-3,0,num=4), 'kernel':['rbf']},
                      {'nu':nus, 'gamma':np.logspace(-3,0,num=4), 'degree':np.arange(2,5), 'kernel':['poly']},
                      {'nu':nus, 'gamma':np.logspace(-3,0,num=4), 'kernel':['sigmoid']}]
            #make a new classifier
            classifier = NuSVC(probability = True)
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "NuSVC_", classifier,
                                      true_data)

    
    def SVM_SVC(self, save_path, true_data, classifier = None):
        """ Construct a SVC model and fit it to the data
        """
        if(classifier == None):
            #setup the param grid list
            params = [
                      {'C':np.logspace(0,5,num=11), 'kernel': ['linear']},
                      {'C':np.logspace(0,5,num=11), 'gamma':np.logspace(-3,3,num=7), 'kernel':['rbf']},
                      {'C':np.logspace(0,5,num=11), 'gamma':np.logspace(-3,3,num=7), 'degree':np.arange(2,5), 'kernel':['poly']},
                      {'C':np.logspace(0,5,num=11), 'gamma':np.logspace(-3,3,num=7), 'kernel':['sigmoid']}
                      ]
            #make a new classifier
            classifier = SVC(probability = False)
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "SVC_", classifier, true_data)

    def SVM_LinearSVC(self, save_path, true_data, classifier = None):
        """ Construct a Linear SVC model and fit it to the data
        """
        if(classifier == None):
            #setup the param grid list
            params = [{'C':np.logspace(-2,5,num=22)}]       
            #make a new classifier
            classifier = LinearSVC()
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "LinearSVC_", classifier,
                                      true_data)
        

    #Stochastic gradient descent
    def SGD_classify(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        if(classifier == None):
            #setup the param grid list
            params = [{'loss':['hinge']},
                      {'loss':['log']},
                      {'loss':['modified_huber'],},
                      {'loss':['perceptron']},
                      {'loss':['squared_hinge']}]       
            #make a new classifier
            classifier = SGDClassifier(shuffle = True)
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "SGD_", classifier,
                                      true_data)

    #Nearest Neghbor Classification
    def K_NearestNeighbor_classify(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        if(classifier == None):
            #setup the param grid list
            params = [{'n_neighbors':np.arange(1,100)}]   
            #make a new classifier
            classifier = KNeighborsClassifier()
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "NNK_", classifier,
                                      true_data)
    
    def Radius_NearestNeighbor_classify(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        if(classifier == None):
            #setup the param grid list
            params = [{'radius':np.arange(1,20)}] 
            #make a new classifier
            classifier = RadiusNeighborsClassifier()
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "NNR_", classifier,
                                      true_data)
    #decision trees
    def DecisionTree_classify(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        ml = self.get_measurement_list()
        if(classifier == None):
            #setup the param grid list
            params = [{'criterion':['entropy'], 'max_features':np.arange(1, len(ml))},
                      {'criterion':['gini'], 'max_features':np.arange(1, len(ml))}] 
            #make a new classifier
            classifier = DecisionTreeClassifier()
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "DecisionTree_", classifier,
                                      true_data)    

    #decision trees
    def NaiveBayes(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        ml = self.get_measurement_list()
        if(classifier == None):
            #make a new classifier
            classifier = GuaussianNB()      
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "DecisionTree_", classifier,
                                      true_data)
    
    #ensemble methods
    def GradientBoosting_classify(self, save_path, true_data, classifier = None):
        """ Use the SGD model to calculate a classification system
        """
        ml = self.get_measurement_list()
        if(classifier == None):
            #setup the param grid list
            params = [{'learning_rate':np.logspace(-5,5, num = 22), 'n_estimators':np.logspace(0,5, num = 6)}]
            #make a new classifier
            classifier = GradientBoostingClassifier()
            #Now the grid is set up so run it         
            gs = GridSearchCV(classifier, params)
            classifier = self.__run_grid_search(true_data, gs)        
        #else, use the one passed in to fit the data
        return self.__perform_classification(save_path + "GradientBoosting_", classifier,
                                      true_data)     
        
    def __perform_classification(self, save_path, clf, true_data):
        """ Perform the classification and save the results. Returns metrics
            for comparisons between methods and models in the following format:
            tuple(
            area under curve, recall score, percision score, f1_score,
            average percision score, accuracy score
            )endtuple
        """
        true_data = np.array(true_data)
        #Split the data into a test and training set
        x, y = self.__data.shape
        train_data = self.__data[0:x:2, :]
        test_data = self.__data[1:x:2, :]
        #split up the true values as well
        x = len(true_data)
        train_target = true_data[0:x:2].T
        test_target = true_data[1:x:2].T
        #fit
        result = clf.fit(train_data, train_target)
        #save the classifier object
        f = open(save_path +".clf", "w")
        pickle.dump(clf, f)
        f.close()

        #Make a prediction with the data
        predict = clf.predict(test_data)
        #Now compute the metrcis to return
        #reuires binary input so convert via > 0 logic
        ras = metrics.roc_auc_score(test_target > 0, predict)
        rs = metrics.recall_score(test_target, predict)
        ps = metrics.precision_score(test_target, predict)
        f1 = metrics.f1_score(test_target, predict)
        #reuires binary input so convert via > 0 logic
        aps = metrics.average_precision_score(test_target > 0, predict)
        aS = metrics.accuracy_score(test_target, predict)
        #and return them
        return rs, ps, f1, ras, aps, aS

    def __run_grid_search(self, true_data, grid_search):
        """ Helper method to split the data and perform the grid search
        """
        #split the data to training and test sets
        #samples are the rows, measurements the columns
        x, y = self.__data.shape
        d = scale(self.__data)
        train_data = d[0:x:2, :]
        test_data = d[1:x:2, :]
        #split up the true values as well
        x = len(true_data)
        train_target = true_data[0:x:2].T
        test_target = true_data[1:x:2].T
        #fit it to the data
        grid_search.fit(train_data, train_target)
        print(grid_search.best_score_)
        print(grid_search.best_params_)
        return grid_search.best_estimator_
        
        
    ############################################################################
    # HEAT MAP DRAWING METHODS

    def save_heat_map(self, file_path,
                      color_map = None,
                      grid_size = 20,
                      font_size = 16):
        """ Draw a heat map which is unclustered and not normalized
        """
        if(color_map == None):
            color_map = self._default_color_map
        #Draw a unclsutered heat map
        #load in the font settings
        font = ImageFont.truetype("arial.ttf", font_size)
        #make a new rgb image
        im = Image.new("RGB", (10,10))
        draw = ImageDraw.Draw(im)
        #actually figure out what the grid_size should be
        names = self.__sample_list.keys()
        w, h = draw.textsize(names[0], font = font)
        grid_size = max(grid_size, h*1.1)
        #calculate difference for font scaling
        text_shift = int((grid_size - h*1.1)/2)
        #compute the size of the square image to create
        ho = self.__sample_count * grid_size
        wo = self.__measurement_count * grid_size

        #draw the heat map
        heat_map = self.__draw_heat_map(wo, ho, grid_size, color_map)
        #draw the sample IDS
        samples = self.__draw_samples(ho, grid_size, font)
        #draw the measurement IDs
        measurements = self.__draw_measurements(wo, grid_size, font)
        #draw the color bar
        color_bar, th = color_map.draw((int(grid_size*1.5), ho), font)

        #finally make a composite of all of these images
        w, h = heat_map.size
        w1, h1 = samples.size
        w2, h2 = measurements.size
        w3, h3 = color_bar.size
        #find the total w, h pof the image
        wt = w + w1 + w3 + grid_size #this is the space for the color bar
        ht = h3 + h2
        #make a new image
        im_final = Image.new("RGB", (wt,ht), "white")
        #paste the other images on to this
        im_final.paste(heat_map, (w1, h2))
        im_final.paste(samples, (0, h2 + text_shift))
        im_final.paste(measurements, (w1 + text_shift, 0))
        im_final.paste(color_bar, (w1 + w + grid_size, h2 - int(th/2)))
        #save the image
        im_final.save(file_path + "heatmap.tiff")

    def save_clustered_heat_map(self, file_path,
                                cluster_axis = 0,
                                color_map = None,
                                grid_size = 20,
                                font_size = 16,
                                annotation_func_s = None,
                                annotation_func_m = None):
        """ Plots the data and exports it to a save file as an image
        """
        if(color_map == None):
            color_map = self._default_color_map
        #load in the font settings
        font = ImageFont.truetype("arial.ttf", font_size)
        #make a new rgb image
        im = Image.new("RGB", (10,10))
        draw = ImageDraw.Draw(im)
        #actually figure out what the grid_size should be
        names = self.__sample_list.keys()
        w, h = draw.textsize(names[0], font = font)
        grid_size = max(grid_size, h*1.1)
        #calculate difference for font scaling
        text_shift = int((grid_size - h*1.1)/2)
        #compute the size of the square image to create
        ho = self.__sample_count * grid_size
        wo = self.__measurement_count * grid_size
        
        #draw the dendogram for samples
        if(cluster_axis == 0 or cluster_axis == 1):
            self.cluster_data_by_sample()
            Z = linkage(self.__data, method = 'average')
            tree = to_tree(Z)
            #with of this is the sampple
            d_s = self.__draw_dendogram(ho, 100, tree,
                                        self.__sample_count,
                                        grid_size)
            d_s = d_s.rotate(90)
            d_s = ImageOps.flip(d_s)
            #get the size
            w4, h4 = d_s.size
            #draw annotations if necessary
##            print(annotation_func_s('r'))
            if(annotation_func_s == None):
                w6 = 0
                h6 = 0
            else:
                #make a new image
                w6 = grid_size*2
                h6 = h4
                im_s_a = Image.new("RGBA", (w6, h6), "white")
                im_s_a_d = ImageDraw.Draw(im_s_a)
                #use it to get a list of colors for each
                sl = self.get_sample_list()
                for i in range(0, self.__sample_count):
                    s = sl[i]
                    col = annotation_func_s(s)
                    bbox = (0, i*grid_size, grid_size, (i+1)*grid_size)
                    im_s_a_d.rectangle(bbox, fill = col)
        else:
            w4 = 0
            h4 = 0
            w6 = 0
            h6 = 0
                
        #now flip this
        #draw the dendogram for measurements
        if(cluster_axis == 0 or cluster_axis == 2):
            self.cluster_data_by_measurement()
            Z = linkage(np.transpose(self.__data), method = 'average')
            tree = to_tree(Z)
            d_m = self.__draw_dendogram(wo, 100, tree,
                                        self.__measurement_count,
                                        grid_size)
            #get the size
            w5, h5 = d_m.size
        else:
            w5 = 0
            h5 = 0
            #draw annotations if necessary

        #draw the heat map
        heat_map = self.__draw_heat_map(wo, ho, grid_size, color_map)
        #draw the sample IDS
        samples = self.__draw_samples(ho, grid_size, font)
        #draw the measurement IDs
        measurements = self.__draw_measurements(wo, grid_size, font)
        #draw the color bar
        color_bar, th = color_map.draw((int(grid_size*1.5), ho), font)

        #finally make a composite of all of these images
        w, h = heat_map.size
        w1, h1 = samples.size
        w2, h2 = measurements.size
        w3, h3 = color_bar.size
        
        #find the total w, h pof the image
        wt = w + w1 + w3 + grid_size + w4 + w6 #this is the space for the color bar
        ht = h3 + h2 + h5
        #make a new image
        im_final = Image.new("RGB", (wt,ht), "white")
        #paste the other images on to this
        im_final.paste(heat_map, (w6 + w1, h2))
        im_final.paste(samples, (0, h2 + text_shift))
        im_final.paste(measurements, (w6 + w1 + text_shift, 0))
        #depending on the cluster type draw the dendrogram
        if(cluster_axis == 0 or cluster_axis == 1):
            im_final.paste(d_s, (w6 + w1 + w, h2 + int(grid_size/2)))
        if(cluster_axis == 0 or cluster_axis == 2):
            im_final.paste(d_m, (w6 + w1 + int(grid_size/2), h2+h))
        #pase the annotations
        if(annotation_func_s == None):
            pass
        else:
            im_final.paste(im_s_a, (w1, h2 + text_shift))
        #past the color bar
        im_final.paste(color_bar, (w6 + w1 + w + w4 + grid_size, h2))
        #save the image
        im_final.save(file_path + "clustered_heatmap.tiff")

    def __draw_samples(self, ho, grid_size, font):
        #Now draw all of the sample IDs on the image
        #get the list of names to draw
        names = self.__sample_list.keys()
        #get all the w and h's necessary for drawing the sample IDs
        max_w = 0
        #make a dummy drawing object and image
        im = Image.new("RGB", (0,0), "white")
        draw = ImageDraw.Draw(im)
        for i in range(0, self.__sample_count):
            w, hh = draw.textsize(names[i], font = font)
            #find the max w
            max_w = max(max_w, w)
        #make the new image
        wi = 10 + max_w
        hi = ho #defined to be the height of the heat map
        im_samples = Image.new("RGB", (wi,hi), "white")
        #make a PIL drawing map for the image
        draw_samples = ImageDraw.Draw(im_samples)
        #loop over the names
        for i in range(0, self.__sample_count):
            index = self.__sample_list[names[i]]
            pos = (5, grid_size*index)
            #render the text at this position
            draw_samples.text(pos, names[i], font = font, fill = (0,0,0))
        #return the image
        return im_samples

    def __draw_measurements(self, wo, grid_size, font):
        #get the list of measurmenets to draw
        measurements = self.__measurement_list.keys()
        max_w = 0
        #make a dummy drawing object and image
        im = Image.new("RGB", (0,0), "white")
        draw = ImageDraw.Draw(im)
        for i in range(0, self.__measurement_count):
            w, hh = draw.textsize(measurements[i], font = font)
            #find the max w
            max_w = max(max_w, w)
        #now we have the h
        #make the new image
        wi = 15 + max_w
        hi = wo
        #make a new image and drawing obect
        im_measurements = Image.new("RGB", (wi,hi), "white")
        draw_samples = ImageDraw.Draw(im_measurements)
        #loop over all the names and draw them
        for i in range(0, self.__measurement_count):
            index = self.__measurement_list[measurements[i]]
            pos = (10, grid_size*index)
            draw_samples.text(pos, measurements[i], font = font, fill = (0,0,0))
        #rotate the image 90 degrees
        #and return it
        return im_measurements.rotate(90)
        
    def __draw_heat_map(self, wo, ho, grid_size, color_map):
        #data is normalized
        im = Image.new("RGB", (wo,ho), "white")
        draw = ImageDraw.Draw(im)
        #now for each data point draw a square correpsonding to the point
        for i in range(0, self.__sample_count):
            for j in range(0, self.__measurement_count):
                #figure out the value to draw
                data = self.__data[i, j]
                if(np.isnan(data)):
                    data = 0
                col = color_map.getColor(data)
                col = (int(col[0]*256), int(col[1]*256), int(col[2]*256))
                #now draw the rectangle
                bbox = (j*grid_size,
                        i*grid_size,
                        (j+1)*grid_size,
                        (i+1)*grid_size)
                draw.rectangle(bbox,
                               fill = col)
        return im

    def __draw_color_bar(self, ho, grid_size, font, color_map):
        #make a dummy image form drawing and sizing
        im = Image.new("RGB", (0,0), "white")
        draw = ImageDraw.Draw(im)
        #giure ou the range over which things will be drawn
        mx = round(np.max(self.__data), 4)
        mn = round(np.min(self.__data), 4)
        mid = round(((mn + mx) / 2.0), 4)
        #figure out the sizing of the color bar
        tw, th = draw.textsize(repr(mx), font = font)
        w = int(grid_size*1.5) + tw + 10
        h = ho + th + 10 #same height as the heat map, plus the height of the text
        #new image and drawing object
        
        return self._
        
    def __draw_dendogram(self, w, h, tree, count, grid_size):
        #get the height of the top node which is the height of the graph
        top = tree.dist
        #now get the y scale ot use for drawing based on the input size
        scale_y = (h - 5) / top
        #figure out what the image dimesnsiosn should be
        im = Image.new("RGB", (w,h), "white")
        draw = ImageDraw.Draw(im)
        #first label the terminal ndoes
        self.__label_terminal_nodes(tree, count, 0, grid_size)
        #then repeat the whole tree labeling until the root id
        #is in the id list
        ids = []
        while(tree.id not in ids):
            self.__label_tree(tree, count, grid_size, ids)
        #Now the tree is completely labeled, so recursively draw
        #it on the image
        fill = (0,0,0)
        self.__draw_tree(tree, draw, fill, scale_y)
        #the return the image
        return im
        
    def cluster_data_by_measurement(self):
        """ Cluster the data by measurement
        """
        #also reorder the clusters based on this
        #then repeat for the data transposed
        Z = linkage(np.transpose(self.__data), method = 'average')
        tree = to_tree(Z)
        id_list = []
        self.__get_l2r_base_node_order(tree, id_list, self.__measurement_count)
        #now reorder the data by this list
        temp = np.zeros(self.__data.shape)
        temp_dict = dict()
        for i in range(0, len(id_list)):
            old_index = id_list[i]
            new_index = i
            #first change the data
            temp[:, new_index] = self.__data[:, old_index]
            #now shuffle the sample dictionary
            #find the name in the dict with the old index, change it to
            #the new index
            for key in self.__measurement_list.keys():
                #check if the value is the index we are alooking for
                if(self.__measurement_list[key] == old_index):
                    temp_dict[key] = new_index
        #now resave the data
        self.__data = temp
        #and the smaple list
        self.__measurement_list = temp_dict

    def cluster_data_by_sample(self):
        """ Cluster the data by sample
        """
        #Now cluster this
        Z = linkage(self.__data, method = 'average')
        tree = to_tree(Z)
        id_list = []
        self.__get_l2r_base_node_order(tree, id_list, self.__sample_count)
        #now reorder the data by this list
        temp = np.zeros(self.__data.shape)
        temp_dict = dict()
        for i in range(0, len(id_list)):
            old_index = id_list[i]
            new_index = i
            #first change the data
            temp[new_index, :] = self.__data[old_index, :]
            #now shuffle the sample dictionary
            #find the name in the dict with the old index, change it to
            #the new index
            for key in self.__sample_list.keys():
                #check if the value is the index we are alooking for
                if(self.__sample_list[key] == old_index):
                    temp_dict[key] = new_index
        #now resave the data
        self.__data = temp
        #and the smaple list
        self.__sample_list = temp_dict

    def __repr__(self):
        """ Returns a string representation of the data set
        """
        s = ""
        #now save the actual data
        for i in range(0, self.__sample_count):
            for j in range(0, self.__measurement_count):
                s += repr(self.__data[i,j])
                if(j < self.__measurement_count - 1):
                    s += ","
            s+= "\n"
        return s
  
    def __get_l2r_base_node_order(self, node, node_list, count):
        """ Returns the base nodes in a left to right order
            for cluster ordering pruposes
        """
        #get the right and the left nodes as well as the count
        right = node.right
        left = node.left
        #determine wether to add your ID to the list
        if(node.id < count):
            node_list.append(node.id)
        if(left != None):
            self.__get_l2r_base_node_order(left, node_list, count)  
        if(right != None):
            self.__get_l2r_base_node_order(right, node_list, count)

    def __draw_tree(self, node, draw, fill, scale_y):
        """ Draws the tree once it has been correctly annotated
        """
        if(node.right != None and node.left != None):
            #then you are a node which should draw its children
            lx, ly = node.left.point
            rx, ry = node.right.point
            x, y = node.point
            #scale the y -axis
            ly = int(ly*scale_y)
            ry = int(ry*scale_y)
            y = int(y*scale_y)
            #draw the left extension
            draw.line((lx, ly, lx, y), fill = fill)
            #draw the right extension
            draw.line((rx, ry, rx, y), fill = fill)
            #draw the top
            draw.line((lx, y , rx, y), fill = fill)
            #now repeat with the two child nodes
            self.__draw_tree(node.left, draw, fill, scale_y)
            self.__draw_tree(node.right, draw, fill, scale_y)
            
    def __label_tree(self, node, count, grid_size, ids):
        """ Labels the tree with a point, which defines where to draw
            the dendogram
        """
        #first check that you are not a terminal node
        if(node.id >= count and node.id not in ids):
            #check to see if your children are in the ID list
            idr = node.right.id
            idl = node.left.id
            check1 = (idr in ids or idr < count)
            check2 = (idl in ids or idl < count)
            #if both are in the id list
            #or both are less then count
            if(check1 and check2):
                #it has been labeled with a point, make your point
                #the average of the x distances
                x1, y1 = node.right.point
                x2, y2 = node.left.point
                x = int((x1 + x2)/2.0)
                y = node.dist
                node.point = [x,y]
                #append the id to the ids
                ids.append(int(node.id))
            else:
                #now recurse over the child nodes
                self.__label_tree(node.left, count, grid_size, ids)
                self.__label_tree(node.right, count, grid_size, ids)
            

    def __label_terminal_nodes(self, node, count, tn_count, grid_size):
        """ Labels all the terminal nodes with a point defining their draw
            posiiton
        """
        #first check if you are a terminal node
        #by checking if the id is less then count
        if(node.id < count):
            #figure out where to place it
            x = tn_count*grid_size
            y = 0
            node.point = [x,y]
            #increment the terminal node
            tn_count += 1
            return tn_count 
        else:
            #You must not be a terminal node so recurse
            #get the left and right nodes
            left = node.left
            right = node.right
            tn_count = self.__label_terminal_nodes(left, count,
                                                    tn_count,
                                                    grid_size)  
            tn_count = self.__label_terminal_nodes(right, count,
                                                    tn_count,
                                                    grid_size)
            return tn_count

    def __normalize_data(self, data):
        """ Normalizes the data making sure to preserve the sign if
            it the data is both negative and positive
        """
        #compute the min
        mn = float(np.min(data))
        if(mn < 0):
            #then we have negative data
            mx = float(np.max(data))
            rng_pos = mx
            rng_neg = mn
            result = []
            for i in range(0, len(data)):
                if(data[i] > 0):
                    value = (data[i]) / rng_pos
                else:
                    value = -(data[i]) / rng_neg
                if(np.isnan(value)):
                    value = 0
                result.append(value)
        else:
            mx = float(np.max(data))
            rg = mx - mn    
            result = []
            for i in range(0, len(data)):
                value = (data[i] - mn) / rg
                if(np.isnan(value)):
                    value = 0
                result.append(value)
        return result
################################################################################
# DATA UTILITY METHODS
################################################################################
def copy(d):
    """ Makes a copy of the data set
    """
    vals = d.get_all_data()
    sl = list(d.get_sample_list())
    ml = list(d.get_measurement_list())
    return data(sl = sl, ml = ml, values = vals)
    
def make_new_data_set(base_data, sl, ml):
    """ Makes a new data set using the values stored in the base data parameter
        base_data - the vlaues to pupulate the data object with
        ml - the list of measurement values
        sl - the list of sample values
        *IMPORTANT* - data must be a len(sl) by len(ml) array
        RETURNS - a new data object
    """
    #add the data
    new_data = data()
    #add the measurements
    for i in range(0, len(ml)):
        new_data.add_measurement(ml[i])
    #Now add the samples
    for i in range(0, len(sl)):
        #add the sample
        new_data.add_sample(sl[i])
        #now copy the data over
        for j in range(0, len(ml)):
            new_data.set_data(sl[i], ml[j], base_data[i][j])
    #return the new data
    return new_data

def merge_data_sets(data_set_list):
    """ Merges all data sets in the data set list into one final data set
    """
    sls = []
    mls = []
    vals = []
    for i in range(0, len(data_set_list)):
        current_data = data_set_list[i]
        #get the measuremenet list
        mls = current_data.get_measurement_list()
        #Now add all the samples
        sl = current_data.get_sample_list()
        sls.extend(sl)
        #Now get the data
        for j in range(0, len(sl)):
            vals.append(current_data.get_sample(sl[j]))
    #Make a new data object using these arguments
    merged_data = data(sl = sls, ml = mls, values = vals)
    #return the new merged data set
    return merged_data

################################################################################
# CLUSTERING METHODS
################################################################################
def k_means(d, n_clusters, n_init = 15):
    #make the k_means clusterer
    k_means = KMeans(n_clusters = n_clusters, n_init = n_init)
    #get all the data form the external data set
    dat = d.get_all_data()
    #then fit it to the data
    k_means.fit(dat)
    #return the fitted cluster
    return k_means
    

def affinity_propogation(d, damping = 0.5, preference = None):
    #make the model
    af = AffinityPropagation(damping = damping,
                             preference = preference)
    #fit the data
    dat = d.get_all_data()
    af.fit(dat)
    #return the fitted cluster metric
    return af


def mean_shift(d):
    #make the model
    ms = MeanShift()
    #get the data matrix
    dat = d.get_all_data()
    #fit it
    ms.fit(dat)
    #return the classifier
    return ms 
    

def hierarchical(d, n_clusters):
    #make the model
    ward = Ward(n_clusters = n_clusters)
    #get the data
    dat = d.get_all_data()
    #fit it
    ward.fit(dat)
    #return the clustering method
    return ward

def BDSCAN(d, eps = 0.5, min_samples = 10):
    #make the model
    db = DBSCAN(eps = eps, min_samples = min_samples)
    #get the data
    dat = d.get_all_data()
    #fit it
    db.fit(dat)
    #return the clustering methods
    return db



def calculate_cluster_metrics(self, estimator, true_labels):
    #Now calculate all of the
    hs = metrics.homogeneity_score(true_labels, estimator.labels_)
    cs = metrics.completeness_score(true_labels, estimator.labels_)
    vms = metrics.v_measure_score(true_labels, estimator.labels_)
    ars = metrics.adjusted_rand_score(true_labels, estimator.labels_)
    amif = metrics.adjusted_mutual_info_score(true_labels, estimator.labels_)
    sf = metrics.silhouette_score(self.__data, estimator.labels_,
                                  metric='euclidean',
                                  sample_size=300)
    return (hs, cs, vms, ars, amif, sf)


def plot_cluster(d, estimator, save_path = None):
    #Now compute the values for the true clustering for this data set
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,1,0), (1,0,1), (0,1,1),
              (.5,0,0), (0,.5,0), (0,0,.5),
              (.5,.5,0), (.5,0,.5), (0,.5,.5)]
    
    #Then graph it by first compressing the data using PCA 
    #now compress the data set down to two dimensions using PCA
    #mean unit scale data
    ms = scale(d.get_all_data())
    pca = PCA(n_components=2)
    pca.fit(ms)
    reduced_data = pca.fit_transform(ms)
    estimator.fit(reduced_data)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min()-.1, reduced_data[:, 0].max()+.1
    y_min, y_max = reduced_data[:, 1].min()-.1, reduced_data[:, 1].max()+.1

    #clear all current figures and axis
    plt.clf()
    plt.cla()

    n_clusters = len(np.unique(estimator.labels_))
    #Now plot the results
    #DBSCAN
    if(isinstance(estimator, DBSCAN)):
        title = "DBSCAN"
        core_samples = estimator.core_sample_indices_
        labels = estimator.labels_
        for i in range(0, n_clusters):
            col = colors[i%len(colors)]
            marker = 'o'
            #assume not a core indices
            if i == -1:
                # Black used for noise.
                col = 'k'
                markersize = 6
            class_members = [index[0] for index in np.argwhere(labels == i)]
            cluster_core_samples = [index for index in core_samples
                                    if labels[index] == i]
            for index in class_members:
                x = reduced_data[index]
                #now plt
                plt.plot(x[0], x[1], 'o', markerfacecolor=col,
                        markeredgecolor='k')
                
    #Ward
    elif(isinstance(estimator, Ward)):
        title = "WARD"
        for i in range(0,n_clusters):
            my_members = estimator.labels_ == i
            col = colors[i%len(colors)]
            plt.plot(reduced_data[my_members, 0],
                     reduced_data[my_members, 1],
                     'o', markerfacecolor=col,
                     markeredgecolor = 'k',
                     marker = "o")
            
    #AFP
    elif(isinstance(estimator, AffinityPropagation)):
        title = "AffinityPropagation"
        for i in range(0, n_clusters):
            cluster_centers_indices = estimator.cluster_centers_indices_
            labels = estimator.labels_
            class_members = labels == i
            cluster_center = reduced_data[cluster_centers_indices[i]]
            #plot the reduced data
            col = colors[i%len(colors)]
            plt.plot(reduced_data[class_members, 0],
                     reduced_data[class_members, 1],
                     'o', markerfacecolor=col,
                     markeredgecolor = 'k',
                     marker = "o")
            #plot the cluster centers
##                plt.plot(cluster_center[0],
##                         cluster_center[1],
##                         '^',
##                         markerfacecolor = col,
##                         markeredgecolor='k',
##                         markersize=15)
##                #draw a line from the center to the outside point
##                for x in reduced_data[class_members]:
##                     plt.plot([cluster_center[0], x[0]],
##                              [cluster_center[1], x[1]],
##                              color= col)           

    #Mean Shift/K-means
    else:
        title = "K-means"
        for i in range(0, n_clusters):
            my_members = estimator.labels_ == i
            cluster_center = estimator.cluster_centers_[i]
            col = colors[i%len(colors)]
            #plot all of the objects in the same color 
            plt.plot(reduced_data[my_members, 0],
                     reduced_data[my_members, 1],
                     'o', markerfacecolor=col,
                     markeredgecolor = 'k',
                     marker = "o")
            
##                #make the centroid a larger version of the same color to mark the middle
##                plt.plot(cluster_center[0],
##                         cluster_center[1],
##                         'o',markerfacecolor = col,
##                         markeredgecolor = 'k',
##                         markersize=15,
##                         marker = "^") 
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.show()
    if(save_path != None):
        plt.savefig(save_path + ".tiff")


################################################################################
# DIMENSIONAL REDUCTION
################################################################################
def ICA(d, n_components):
    ms = scale(self.__data)
    #Apply the ICA transform
    ica = FastICA(n_components = n_components)
    ica.fit(ms)
    #now apply the transform
    d = ica.fit_transform(ms)
    #return the fit data
    return d
  
def perform_PCA(d, n_dims, info = False, pca = None):
    """ Perform a PCA transform on the data set.
        n_dims = the number of dimensions to use
        info - boolean vlaue to print out information or not
        Return - the PCA object, and the transformed data saved to a new
                 data set
    """
##    #normalize the data
##    d.normalize_measurements()
    #get the data
    dat = d.get_all_data()
    #first mean center and shift the data
    ms = scale(dat)
    #Create a PCA model
    if(pca == None):
        pca = PCA(n_components = n_dims)
        pca.fit(ms)
    #apply it to the data
    transformed_data = pca.transform(ms)
    if(info):
        #calculate the information about the PCA
        print("PCA explained variables")
        print(pca.explained_variance_ratio_)
        print("PCA component information")
        print(pca.components_)
        print("Number of components Selected: " + repr(len(pca.components_)))

    #make the new measurement list
    ml = []
    for i in range(0, len(pca.components_)):
        ml.append("PCA-" + repr(i))
    #Copy over the data to the new data set
    new_data = data(sl = d.get_sample_list(), ml = ml, values = transformed_data)
    #now return the new data and the pca object
    return pca, new_data

def plot_component_heat_maps(pca, variable_names, save_path):
    """ Make a heatmap depicting the strength of the variables defined
        in the variable named function for the given PCA transform
        pca - the fitted pca transform to graph the ocmponents of
        variable_names - the variable names
        save_path - the path to save the resulting components to
    """
    #get the number of dimnesions
    n_dims = len(pca.components_)
    #set the labels
    x_labels = variable_names
    y_labels = []
    for i in range(0,  n_dims):
        p1 = pca.explained_variance_ratio_[i]*100
        #draw labeled heat maps
        y_labels.append("PCA-" + repr(i+1) + "(" + repr(round(p1, 2)) +"%)")
    #set the data vlaues
    values = pca.components_
    #define a new data set
    d = data()
    #find the max and min of the data
    mx = np.max(values)
    mn = np.min(values)
    #define a complex linear color map
    set_points = {mx:[1,0,0,1], 0:[0,0,0,1], mn:[0,0,1,1]}
    cm = colormap(set_points)
    for i in range(0, len(x_labels)):
        d.add_sample(x_labels[i])
        for j in range(0, len(y_labels)):
            d.add_measurement(y_labels[j])
            d.set_data(x_labels[i], y_labels[j], values[j,i])
    #finally draw the saved heat map
    d.transpose_measurments_and_samples()
    d.save_heat_map(save_path, cm)

def plot_loading_plots(pca, n_dims, measurement_list, save_path):
    """ Plots loading plots for 2D and 3D PCA instances
    """
    #clear all current figures and axis
    plt.cla()
    plt.clf()
    plt.close()
    #set the colors and markers to be used
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,1,0), (1,0,1), (0,1,1),
              (.5,0,0), (0,.5,0), (0,0,.5),
              (.5,.5,0), (.5,0,.5), (0,.5,.5)]
    markers = ['o', 's', 'D', '^', 'v',
               '*', '>', '.', 'p', '<']
    if(n_dims == 2):
        #Plot the 2D principal components
        #get the tranpose of the component weights
        pts = pca.components_.T
        title = 'PCA variable loadings PCA-1 vrs PCA-2'
        xlabel = "PCA-1"
        ylabel = "PCA-2"
        x = pts[:,0]
        y = pts[:,1]
        sp = save_path + "2D_loadings_2D_PCA1vPCA2"
        self.__plot_2D_loading_plots(measurement_list, colors, markers, x, y,
                                     title, ylabel, xlabel, save_path)      
    if(n_dims == 3):
        #get the loading values for the componenets
        pts = pca.components_.T
        #also plot both the PCA-1 vr.s PCA-2 and PCA-1 vrs PCA-3
        #PCA-1 vrs PCA-2
        title = 'PCA variable loadings PCA-1 vrs PCA-2'
        xlabel = "PCA-1"
        ylabel = "PCA-2"
        x = pts[:,0]
        y = pts[:,1]
        sp = save_path + "3D_loadings_2D_PCA1vPCA2"
        __plot_2D_loading_plots(measurement_list, colors, markers, x, y,
                                title, ylabel, xlabel, save_path)  
        #PCA-1 vrs PCA - 3
        title = 'PCA variable loadings PCA-1 vrs PCA-3'
        xlabel = "PCA-1"
        ylabel = "PCA-3"
        x = pts[:,0]
        y = pts[:,2]
        sp = save_path + "3D_loadings_2D_PCA1vPCA3"
        __plot_2D_loading_plots(measurement_list, colors, markers, x, y,
                                title, ylabel, xlabel, sp)
        #PCA-2 vrs PCA - 3
        title = 'PCA variable loadings PCA-2 vrs PCA-3'
        xlabel = "PCA-2"
        ylabel = "PCA-3"
        x = pts[:,1]
        y = pts[:,2]
        sp = save_path + "3D_loadings_2D_PCA2vPCA3"
        __plot_2D_loading_plots(measurement_list, colors, markers, x, y,
                                title, ylabel, xlabel, save_path)

def plot_PCA_transformed_data(pca_data, pca, n_dims, labels,
                              save_path = None,
                              marker_labels = None,
                              annotations = None,
                              axis_boundaries = None,
                              show_fig = True):
    """ Plot the transformed data on the given PCA axis
    """
    d = pca_data.get_all_data()
    #clear all current figures and axis
    colors = [(1,0,0), (0,1,0), (0,0,1),
              (1,1,0), (1,0,1), (0,1,1),
              (.5,0,0), (0,.5,0), (0,0,.5),
              (.5,.5,0), (.5,0,.5), (0,.5,.5)]

    markers = ['o', 's', 'D', '^', 'v',
               '*', '>', '.', 'p', '<']
    #get the number of clusters
    unique_labels = np.unique(labels)
    if(marker_labels != None):
        unique_marker_labels = np.unique(marker_labels)
    #Now for each point, plot it on the PCA axis
    if(n_dims == 2):
        # Plot the decision boundary.
        if(axis_boundaries == None):
            x_min, x_max = d[:, 0].min()-.1, d[:, 0].max()+.1
            y_min, y_max = d[:, 1].min()-.1, d[:, 1].max()+.1
        else:
            x_min, x_max, y_min, y_max = axis_boundaries
        plt.cla()
        plt.clf()
        #Keep track fo the plots for the legend
        plots = []
        i = 0
        j = 0
        for l in unique_labels:
            if(marker_labels == None):
                if(l != -1):
                    print(l)
                    my_members = labels == l
                    m_i = (l) % len(markers)
                    c_i = (l) % len(colors)
                    col = colors[c_i]
    ##                if(i >= 6):
    ##                    col = colors[0]
    ##                if(i < 6):
    ##                    col = colors[1]
                    #now plot it
                    mark = markers[m_i]
                    plt.plot(d[my_members, 0],
                                 d[my_members, 1],
                                 'o',
                                 markersize = 10,
                                 markerfacecolor=col,
                                 markeredgecolor = 'k',
                                 marker = mark,
                                 label = i)
                    #increment the count
                    i += 1
            else:
                if(l != -1):
                    j = 0
                    for ml in unique_marker_labels:
                        if(ml != -1):
                            my_members = labels == l
                            my_members2 = marker_labels == ml
                            my_members = np.logical_and(my_members, my_members2)
                            m_i = (j) % len(markers)
                            c_i = (i) % len(colors)
                            col = colors[c_i]
                            mark = markers[m_i]
                            #now plot it
                            plt.plot(d[my_members, 0],
                                    d[my_members, 1],
                                     'o',
                                    markersize = 10,
                                    markerfacecolor=col,
                                    markeredgecolor = 'k',
                                    marker = mark,
                                    label = i)
                            #increment the count
                            j += 1
                    #increment the count
                    i += 1
        #Annotate every point
        if(annotations != None):
            for i in range(0, len(annotations)):
                if(labels[i] != -1):
                    plt.annotate(annotations[i], d[i])
        plt.title('PCA')
        plt.xlabel('PCA-1')
        plt.ylabel('PCA-2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if(save_path != None):
            plt.savefig(save_path + "2D" + ".tiff")
        if(show_fig):
            plt.show()
    
    if(n_dims == 3):
        if(axis_boundaries == None):
            x_min, x_max = d[:, 0].min()-.1, d[:, 0].max()+.1
            y_min, y_max = d[:, 1].min()-.1, d[:, 1].max()+.1
            z_min, z_max = d[:, 2].min()-.1, d[:, 2].max()+.1
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = axis_boundaries
        #set the labels
        p1 = pca.explained_variance_ratio_[0]*100
        p2 = pca.explained_variance_ratio_[1]*100
        p3 = pca.explained_variance_ratio_[2]*100
        #draw labeled heat maps
        y_labels = ["PCA-1 (" + repr(round(p1, 2)) +"%)",
                    "PCA-2 (" + repr(round(p2, 2)) +"%)",
                    "PCA-3 (" + repr(round(p3, 2)) +"%)"]
        
        #plot the scores plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #get the number of clusters
        unique_labels = np.unique(labels)
        i = 0
        j = 0
        for l in unique_labels:
            if(marker_labels == None):
                if(l != -1):
                    my_members = (labels == l)
                    m_i = (i) % len(markers)
                    c_i = (i) % len(colors)
                    col = colors[c_i]
                    mark = markers[m_i]
                    #now plot it
                    ax.plot(d[my_members, 0],
                            d[my_members, 1],
                            d[my_members, 2],
                            'o',
                            markerfacecolor=col,
                            markeredgecolor = 'k',
                            marker = mark,
                            label = i)
                    #increment the count
                    i += 1
            else:
                if(l != -1):
                    j = 0
                    for ml in unique_marker_labels:
                        if(ml != -1):
                            my_members = labels == l
                            my_members2 = marker_labels == ml
                            my_members = np.logical_and(my_members, my_members2)
                            m_i = (j) % len(markers)
                            c_i = (i) % len(colors)
                            col = colors[c_i]
                            mark = markers[m_i]
                            #now plot it
                            plt.plot(d[my_members, 0],
                                    d[my_members, 1],
                                    d[my_members, 2],
                                     'o',
                                    markerfacecolor=col,
                                    markeredgecolor = 'k',
                                    marker = mark,
                                    label = i)
                            #increment the count
                            j += 1
                    #increment the count
                    i += 1
        plt.title('PCA')
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_zlabel("PCA-3")
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.view_init(elev = 25, azim = -77)
        if(save_path != None):
            plt.savefig(save_path + "3D" + ".tiff")
        if(show_fig):
            plt.show()

def __plot_2D_loading_plots(ml, colors, markers, x, y,
                            title, ylabel, xlabel, save_path):
        plt.cla()
        plt.clf()
        #keep track of all the plots and titles
        pls = []
        mls = []
        plt.figure(num = 1, figsize = (10,10))
        for i in range(0, len(ml)):
            c_ml = ml[i]
            c_i = i % len(colors)
            m_i = i % len(markers)
            plt.plot(x[i], y[i],
                    marker = markers[m_i],
                    linestyle = '',
                    markerfacecolor=colors[c_i],
                    markeredgecolor = 'k',
                    markersize = 10,
                    label = c_ml)
            #make the legend
##                plt.legend(ncol = 4)
            
        #Draw the legend seperatly
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.savefig(save_path + ".tiff", DPI = 100)
        plt.show()
