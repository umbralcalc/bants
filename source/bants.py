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


    # Must initialise with the 'bants' directory declared
    def __init__(self,path):       

        self.path = path     


    # Method of 'fit' is analogous to the scikit-learn pattern
    def fit(self,train_df):
    '''
    Method to first infer the maximum likelihood for the scale parameter of each convolution kernel, followed by a
    computation of the log Bayesian evidence through optimisation of the prior hyperparameters of the network. The 
    latter optimisation uses a second-order gradient descent algorithm.

    INPUT:

    train_df     -     This is an input dataframe representing the training data of the vector time series process 
                       that one wishes to model. Simply set this to be a pandas dataframe with time as the index.

    OUTPUT:

    result       -     This is an output dictionary containing all of the information relevant to the inference of
                       the network from the training data. The keys are: '' XXXXXXX
                       
    '''

        # Make training data available to class object
        self.train_df = train_df


    # Method of 'predict' is analogous to the scikit-learn pattern
    def predict(self,times,validate=None):
    '''
    Method to 

    INPUT:

    times        -     XXXXXXX

    validate     -     (Optional) One can input a dataframe representing the testing data of the vector time series 
                       process to cross-validate the 'bants' predictions against. Simply set this to be a pandas 
                       dataframe of the same form as 'train_df' in the 'fit' method.

    OUTPUT:

    result       -     This is an output dictionary containing all of the statistics relevant to the prediction
                       made by 'bants' over the requested times, a future point sampler and potentially the results
                       of the cross-validation against the testing data if this was provided. The keys are: '' XXXXXXXX
                       
    '''

        # Make prediction times and possible testing data available to class object
        self.predict_times = train_df
        if validate is not None: self.test_df = validate



    
