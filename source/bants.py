
'''
BANTS - BAyesian Networks for Time Series forecasting

This is the main 'bants' class to be used on generic N-dimensional datasets. The structure is intended to be as simple
as possible for rapid use in a commerical data science context. For more details on the mathematics behind 'bants'
please refer to the notes/how_bants_works.ipynb Jupyter Notebook in the Git repository.

'''

import numpy as np
import pandas as pd


# Initialize the 'bants' method class
class bants:


    # Initialisation needs only the network type as input. Only type so far is 'AR-GP'.
    def __init__(self,net_type):       

        # Methods and functions    
        self.fit
        self.predict
        self.optimise_ARGP_hyperp
        
        # Set network type
        self.net_type = net_type
        
        # Initialise empty dictionaries of network parameters to learn, fitting information and prediction results
        self.params = {}
        self.info = {}
        self.results = {}

        # If network type is 'AR-GP' then set kernel scales
        if self.net_type == 'AR-GP':
            # Default type of convolution kernel for each column is always 'SquareExp'. The other option, for oscillatory 
            # columns in the data over time, is 'Periodic'. 
            self.column_kernel_types = ['SquareExp']


    # Function for optimising the hyperparameters of the 'AR-GP' network 
    def optimise_ARGP_hyperp(self,data):
    '''
    Method of second-order gradient descent to optimise the 'AG-GP' network hyperparameters as defined in 
    notes/how_bants_works.ipynb. Optimisation outputs are written to bants.params and bants.info accordingly. 
    
    INPUT:

    data         -      This is an input dataframe representing the data of the vector time series process that
                        one wishes to model. Simply set this to be a pandas dataframe with time as the index.
                       
    '''


    # Method of 'fit' is analogous to the scikit-learn pattern
    def fit(self,train_df):
    '''
    Method to first infer the maximum likelihood for the scale parameter of each convolution kernel, followed by a
    computation of the log Bayesian evidence through optimisation of the prior hyperparameters of the network. The 
    latter optimisation uses a second-order gradient descent algorithm. Learned network parameters can be found in 
    the bants.params dictionary, where the keys are: ['h','nu','Psi']. Fitting information can be found in the 
    bants.info dictionary, where the keys are: [XXXXXXXXXXXXXXX].

    INPUT:

    train_df     -     This is an input dataframe representing the training data of the vector time series process 
                       that one wishes to model. Simply set this to be a pandas dataframe with time as the index.
                       
    '''

        # Make training data available to class object
        self.train_df = train_df

        # Find dimensions of dataset
        self.N = len(train_df.columns)

        # If number of columns is more than 1 then check to see if kernel types have been specified. If not then
        # simply set each column to the 'SquareExp' default option. 
        if (self.N > 1) and (len(self.column_kernel_types) > 1): 
            self.column_kernel_types = self.column_kernel_types*self.N

        


    # Method of 'predict' is analogous to the scikit-learn pattern
    def predict(self,times,validate=None):
    '''
    Method to predict future timepoints of the N-dimensional time series using the fitted 'bants' network. All of 
    the results from the prediction, including a future point sampler and potentially the results of the 
    cross-validation against the testing data (if this was provided) can be found in the bants.results dictionary,
    with the keys: [XXXXXXXXXXXXXXXXX].

    INPUT:

    times        -     This is an input set of timepoints to predict with the fitted network. Just a 1-d array.

    validate     -     (Optional) One can input a dataframe representing the testing data of the vector time series 
                       process to cross-validate the 'bants' predictions against. Simply set this to be a pandas 
                       dataframe of the same form as 'train_df' in the 'fit' method.
                       
    '''

        # Make prediction times and possible testing data available to class object
        self.predict_times = times
        if validate is not None: self.test_df = validate



    
