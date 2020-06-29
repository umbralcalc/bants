'''
BANTS - BAyesian Networks for Time Series forecasting

This is the main 'bants' class to be used on generic N-dimensional datasets. The mathematics behind the computations used in the class are explained
in comments above each method to ensure maximum clarity.

'''
import numpy as np


# Initialize the 'bants' method class
class bants:


    def __init__(self,
                 network_type,                     # Must initialise with a network type declared - only type currently is 'GP1HL' (Gaussian Processes with 1 Hidden Layer) 
                 path_to_bants_directory           # Must initialise with a directory declared
                 ):

        self.network_type = network_type
        self.path_to_bants_directory = path_to_bants_directory

        if self.network_type == 'GP1HL':
            ######


    def fit(self,t_train,x_train):
    '''
    INPUTS:


    OUTPUTS:


    '''
