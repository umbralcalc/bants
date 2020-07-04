
'''
BANTS - BAyesian Networks for Time Series forecasting

This is the main 'bants' class to be used on generic N-dimensional datasets. The structure is intended to be as simple
as possible for rapid use in a commerical data science context. For more details on the mathematics behind 'bants'
please refer to the notes/how_bants_works.ipynb Jupyter Notebook in the Git repository.

'''

import numpy as np
import pandas as pd
import tensorflow as tf

# Rename some modules
tfp = tf.probability
tfd = tfp.distributions

# Initialize the 'bants' method class
class bants:


    # Initialisation needs only the network type as input. Only type so far is 'AR-GP'.
    def __init__(self,net_type):       

        # Methods and functions    
        self.fit
        self.predict
        self.optimise_ARGP_hyperp
        self.kconv
        
        # Set network type
        self.net_type = net_type
        
        # Initialise empty dictionaries of network parameters to learn, fitting information and prediction results
        self.params = {}
        self.info = {}
        self.results = {}

        # If network type is 'AR-GP' then set kernel types
        if self.net_type == 'AR-GP':
            # Default type of convolution kernel for each column is always 'SquareExp'. The other option, for oscillatory 
            # columns in the data over time, is 'Periodic'. 
            self.column_kernel_types = ['SquareExp']

            
    # Kernel convolution for the 'AR-GP' network
    def kconv(self,df,h):
    '''
    Method of kernel convolution on input dataframe values according to whichever kernel types were specified
    in the bants.column_kernel_types list. This is used mainly in bants.optimise_ARGP_hyperp but can be used
    indpendently on different dataframes for experimentation.
    
    INPUT:
    
    df       -     This is the input dataframe of values to perform the convolution on.
    
    h        -     This is an input 1-d array of the same length as the number of columns in the dataframe.
    
    OUTPUT:
    
    conv_d   -     This is an output array of convolved data the same shape as the input dataframe.     
                       
    '''
        
        #conv_d = (self.column_kernel_types == 'SquareExp')*(df.values*np.exp(-df.index)) + \
        #         (self.column_kernel_types == 'Periodic')*(df.values*np.exp())
        
        return conv_d
            
            
    # Subroutine for the 'AR-GP' network
    def optimise_ARGP_hyperp(self):
    '''
    Method of second-order gradient descent to optimise the 'AG-GP' network hyperparameters as defined in 
    notes/how_bants_works.ipynb. Optimisation outputs are written to bants.params and bants.info accordingly. 
                       
    '''
        # Define the function to optimise over to obtain optimal network hyperparameters
        def func_to_opt(params,df=self.train_df,N=self.N):
            
            # Extract hyperparameters
            nu = params[0]
            h = params[1:N+1]
            Psi = np.zeros((N,N))
            Psi[np.tril_indices(N)] = params[N+1:]
            
            # Compute the kernel-convolved signal for each data point
            Mt = self.kconv(df,h)
            
            # Compute the scale matrix
            Sm = Psi/(nu-N+1.0)
            
            # Sum log-evidence contributions by each data point
            mvt = tfd.MultivariateStudentTLinearOperator(df=nu,loc=Mt,scale=tf.linalg.LinearOperatorLowerTriangular(Sm))
            lnE = np.sum(np.log(mvt.prob(df.values).eval()),axis=0)
            
            # Output corresponding value to minimise
            return -lnE
    
        #self.params =
        #self.info =


    # Method of 'fit' is analogous to the scikit-learn pattern
    def fit(self,train_df):
    '''
    Method to first infer the maximum likelihood for the scale parameter of each convolution kernel, followed by a
    computation of the log Bayesian evidence through optimisation of the prior hyperparameters of the network. The 
    latter optimisation uses a second-order gradient descent algorithm. Learned network parameters can be found in 
    the bants.params dictionary. Fitting information can be found in the bants.info dictionary.

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
        
        # If 'AR-GP' network then run appropriate optimisation subroutine
        if self.net_type == 'AR-GP':
            self.optimise_ARGP_hyperp()


    # Method of 'predict' is analogous to the scikit-learn pattern
    def predict(self,times,validate=None):
    '''
    Method to predict future timepoints of the N-dimensional time series using the fitted 'bants' network. All of 
    the results from the prediction, including a future point sampler and potentially the results of the 
    cross-validation against the testing data (if this was provided) can be found in the bants.results dictionary.

    INPUT:

    times        -     This is an input set of timepoints to predict with the fitted network. Just a 1-d array.

    validate     -     (Optional) One can input a dataframe representing the testing data of the vector time series 
                       process to cross-validate the 'bants' predictions against. Simply set this to be a pandas 
                       dataframe of the same form as 'train_df' in the 'fit' method.
                       
    '''

        # Make prediction times and possible testing data available to class object
        self.predict_times = times
        if validate is not None: self.test_df = validate



    
