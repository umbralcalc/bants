
'''
BANTS - BAyesian Networks for Time Series forecasting

This is the main 'bants' class to be used on generic N-dimensional datasets. The structure is intended to be as simple
as possible for rapid use in a commerical data science context. For more details on the mathematics behind 'bants'
please refer to the notes/how_bants_works.ipynb Jupyter Notebook in the Git repository. 

Note that due to speed requirements, 'bants' takes as input only time series data that are measured at equally-spaced 
time intervals - this is due to the fact that the kernel convolutions are much faster with a constant window function.

'''

import numpy as np
import pandas as pd
import scipy.special as sps

# Initialize the 'bants' method class
class bants:


    # Initialisation needs only the network type as input. Only type so far is 'AR-GP'.
    def __init__(self,net_type):       

        # Methods and functions    
        self.fit
        self.predict
        self.optimise_ARGP_hyperp
        self.kconv
        self.tD_logpdf
        
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
            
            # Set signal periods in dimensions of the dataframe index (time variable chosen) - this should be a list of 
            # the same length as the dimensions (columns) in the data where entries are relevant for 'Periodic' columns.
            self.signal_periods = []

    
    # Function to output multivariate t-distribution (see here: https://en.wikipedia.org/wiki/Multivariate_t-distribution)
    # log-Probability Density Function from input dataframe points. No scipy implementation so wrote this one.
    def tD_logpdf(self,df,nu,mu,Sigma):
        
        # Compute the log normalisation of the distribution using scipy loggammas
        log_norm = sps.loggamma((nu+self.Nd)/2.0) - sps.loggamma(nu/2.0) - \
                   ((self.Nd/2.0)*np.log(np.pi*nu)) - (0.5*np.log(np.linalg.det(Sigma)))
        
        # Compute the log density function for each of the samples
        x_minus_mu = df.values-np.tensordot(np.ones(self.Ns),mu,axes=0)
        inverseSigma = np.linalg.inv(Sigma)
        contraction = x_minus_mu*np.tensordot(inverseSigma,x_minus_mu,axes=([0],[1]))
        log_densfunc = -((nu+self.Nd)/2.0)*np.log(1.0+(contraction/nu)) 
        
        # Output result
        return log_norm + log_densfunc
    

    # Kernel convolution for the 'AR-GP' network
    def kconv(self,df,h,truncn=self.trunc_kconv_data):
    '''
    Method of kernel convolution on input dataframe values according to whichever kernel types were specified
    in the bants.column_kernel_types list. This is used mainly in bants.optimise_ARGP_hyperp but can be used
    indpendently on different dataframes for experimentation.
    
    INPUT:
    
    df       -     This is the input dataframe of values to perform the convolution on.
    
    h       -     This is an input 1-d array of the same length as the number of columns in the dataframe.
    
    OUTPUT:
    
    conv_d   -     This is an output array of convolved data the same shape as the input dataframe.     
                       
    '''
        # Extract the change in time from the dataframe
        delta_t = df.index[1]-df.index[0]
        
        # Generate an array of centred distances to be used to make the window array of the kernel
        dtimes = delta_t*np.arange(0,self.Ns,1)
        
        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type,h,perd):
            
            # Output weights corresponding to the convolution kernels
            if kern_type == 'SquareExp': return np.exp(-dtimes**2.0/(2.0*(h**2.0)))
            if kern_type == 'Periodic': return np.exp(-2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/(h**2.0))
    
        # Evaluate the convolution on the data dependent on the choice of kernel
        conv_d = np.asarray([((self.column_kernel_types[i] == 'SquareExp')*np.convolve(df.values[:,i],\
                 kern_array('SquareExp',h[i],self.signal_periods[i]))[:self.Ns]) + \
                 ((self.column_kernel_types[i] == 'Periodic')*np.convolve(df.values[:,i],\
                 kern_array('Periodic',h[i],self.signal_periods[i]))[:self.Ns]) for i in range(0,self.Nd)]).T
        
        # Return convolved signals
        return conv_d
            
            
    # Subroutine for the 'AR-GP' network
    def optimise_ARGP_hyperp(self):
    '''
    Method of second-order gradient descent to optimise the 'AG-GP' network hyperparameters as defined in 
    notes/how_bants_works.ipynb. Optimisation outputs are written to bants.params and bants.info accordingly. 
                       
    '''
        # Define the function to optimise over to obtain optimal network hyperparameters
        def func_to_opt(params,df=self.train_df,N=self.Nd):
            
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
            lnE = np.sum(self.tD_logpdf(df,nu-N+1.0,Mt,Sm),axis=0)
            
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

        # Find dimensions and number of samples in dataset
        self.Nd = len(train_df.columns)
        self.Ns = len(train_df.index)

        # If number of columns is more than 1 then check to see if kernel types have been specified. If not then
        # simply set each column to the 'SquareExp' default option. 
        if (self.Nd > 1) and (len(self.column_kernel_types) > 1): 
            self.column_kernel_types = self.column_kernel_types*self.Nd
        
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



    
