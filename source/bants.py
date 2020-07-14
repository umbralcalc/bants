
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
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.special as sps
import scipy.optimize as spo
import scipy.stats as spst


# Initialize the 'bants' method class
class bants:


    # Initialisation needs only the network type as input. Only type so far is 'AR-GP'.
    def __init__(self,net_type):

        # Methods and functions    
        self.fit
        self.predict
        self.optimise_ARGP_hyperp
        self.pred_ARGP_sampler
        self.kconv
        self.tD_logpdf
        self.standardise
        
        # Set network type
        self.net_type = net_type
        
        # Initialise empty dictionaries of network parameters to learn, fitting information and prediction results
        self.params = {}
        self.info = {}
        self.results = {}
        
        # Store the mean and standard deviation of the training data and store if standardisation has been performed
        self.mean_train_df = []
        self.std_train_df = []
        self.standardised = False
        
        # Set the default maximum number of iterations, the log-evidence tolerance and the optimisation algorithm
        self.itmax = 1000
        self.lnEtol = 0.05
        self.optimiser = 'Nelder-Mead' # 'BFGS' is the other choice

        # If network type is 'AR-GP' then set kernel types
        if self.net_type == 'AR-GP':
            
            # Default type of convolution kernel for each column is always 'SquareExp'. The other option, for oscillatory 
            # columns in the data over time, is 'Periodic'. 
            self.column_kernel_types = ['SquareExp']
            
            # Set signal periods in dimensions of the dataframe index (time variable chosen) - this should be a list of 
            # the same length as the dimensions (columns) in the data where entries are relevant for 'Periodic' columns.
            self.signal_periods = [1.0]
            
            # Initial guesses for optimiser of the hyperparameters of the network
            self.nu_guess = None
            self.hsq_guess = None
            self.Psi_tril_guess = None


    # Function to output multivariate t-distribution (see here: https://en.wikipedia.org/wiki/Multivariate_t-distribution)
    # log-Probability Density Function from input dataframe points. No scipy implementation so wrote this one.
    def tD_logpdf(self,df,nu,mu,Sigma):

        # Parameter nu must be positive real
        if nu > 0.0:
            
            # Compute the log normalisation of the distribution using scipy loggammas
            log_norm = sps.loggamma((nu+self.Nd)/2.0) - sps.loggamma(nu/2.0) - \
                       ((self.Nd/2.0)*np.log(np.pi*nu)) - (0.5*np.log(np.linalg.det(Sigma)))
        
            # Compute the log density function for each of the samples
            x_minus_mu = df.values-mu
            inverseSigma = np.linalg.inv(Sigma)
            contraction = np.sum(x_minus_mu.T*np.tensordot(inverseSigma,x_minus_mu,axes=([0],[1])),axis=0)
            log_densfunc = np.sum(-((nu+self.Nd)/2.0)*np.log(1.0+(contraction/nu)))
            
            # Output result
            return log_norm + log_densfunc

        # Else return infinitely small value
        else:
            return -np.inf


    # Kernel convolution for the 'AR-GP' network
    def kconv(self,df,hsq):
        '''
        Method of kernel convolution on input dataframe values according to whichever kernel types were specified
        in the bants.column_kernel_types list. This is used mainly in bants.optimise_ARGP_hyperp but can be used
        independently on different dataframes for experimentation.
    
        INPUT:
    
        df       -     This is the input dataframe of values to perform the convolution on.
    
        hsq      -     This is an input 1-d array of the same length as the number of columns in the dataframe
                       containing the square-amplitude of the kernel scales for each time series dimension.
                       
        OUTPUT:
    
        conv_d   -     This is an output array of convolved data the same shape as the input dataframe.

        '''
        # Extract the change in time from the dataframe
        delta_t = df.index[1]-df.index[0]
        
        # Generate an array of centred distances to be used to make the window array of the kernel
        dtimes = delta_t*np.arange(0,self.Ns,1)[::-1]
        
        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type,hsq,perd):
            
            # Output weights corresponding to the convolution kernels
            if kern_type == 'SquareExp': return np.exp(-dtimes**2.0/(2.0*hsq))
            if kern_type == 'Periodic': return np.exp(-2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/hsq)

        # Evaluate the convolution on the data dependent on the choice of kernel
        conv_d = np.asarray([((self.column_kernel_types[i] == 'SquareExp')*np.convolve(df.values[:,i],\
                 kern_array('SquareExp',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                 np.sum(kern_array('SquareExp',hsq[i],self.signal_periods[i]))) + \
                 ((self.column_kernel_types[i] == 'Periodic')*np.convolve(df.values[:,i],\
                 kern_array('Periodic',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                 np.sum(kern_array('Periodic',hsq[i],self.signal_periods[i]))) for i in range(0,self.Nd)]).T
        
        # Return convolved signals
        return conv_d
    
    
    # Kernel convolution and its first derivative for the 'AR-GP' network
    def kconv_and_deriv(self,df,hsq):
        '''
        Method of kernel convolution on input dataframe values according to whichever kernel types were specified
        in the bants.column_kernel_types list. This method also computes the first derivatives of the convolved 
        data. This is used mainly in bants.optimise_ARGP_hyperp with the 'BFGS' optimiser but can be used independently 
        on different dataframes for experimentation.
    
        INPUT:
    
        df       -     This is the input dataframe of values to perform the convolution on.
    
        hsq      -     This is an input 1-d array of the same length as the number of columns in the dataframe
                       containing the square-amplitude of the kernel scales for each time series dimension.
                       
        OUTPUT:
    
        conv_d   -     This is an output array of convolved data the same shape as the input dataframe.
        
        Dconv_d  -     This is an output corresponding to the first derivative of the convolved data. Since the
                       cross-derivatives of the corresponding Jacobian are zero, this is merely a 1-d array of the
                       same length as conv_d.

        '''
        # Extract the change in time from the dataframe
        delta_t = df.index[1]-df.index[0]
        
        # Generate an array of centred distances to be used to make the window array of the kernel
        dtimes = delta_t*np.arange(0,self.Ns,1)[::-1]
        
        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type,hsq,perd):
            
            # Output weights corresponding to the convolution kernels
            if kern_type == 'SquareExp': return np.exp(-dtimes**2.0/(2.0*hsq))
            if kern_type == 'Periodic': return np.exp(-2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/hsq)
            
        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type,hsq,perd):
            
            # Output weights corresponding to the convolution kernels
            if kern_type == 'SquareExp': return np.exp(-dtimes**2.0/(2.0*hsq))
            if kern_type == 'Periodic': return np.exp(-2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/hsq)
            
        # Create function which returns an array corresponding to the first derviative of the unnormalised 
        # convolution window function for the chosen input kernel type.
        def Dkern_array_unnorm(kern_type,hsq,perd):
            
            # Output weights corresponding to the convolution kernels
            if kern_type == 'SquareExp': return (dtimes**2.0/(2.0*hsq*hsq))*np.exp(-dtimes**2.0/(2.0*hsq))
            if kern_type == 'Periodic': return (2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/(hsq*hsq))*\
                                                np.exp(-2.0*((np.sin(np.abs(np.pi*dtimes/perd)))**2.0)/hsq)

        # Evaluate the convolution on the data dependent on the choice of kernel
        conv_d = np.asarray([((self.column_kernel_types[i] == 'SquareExp')*np.convolve(df.values[:,i],\
                 kern_array('SquareExp',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                 np.sum(kern_array('SquareExp',hsq[i],self.signal_periods[i]))) + \
                 ((self.column_kernel_types[i] == 'Periodic')*np.convolve(df.values[:,i],\
                 kern_array('Periodic',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                 np.sum(kern_array('Periodic',hsq[i],self.signal_periods[i]))) for i in range(0,self.Nd)]).T
        
        # Evaluate the first derivative of convolution on the data dependent on the choice of kernel
        Dconv_d_u = np.asarray([((self.column_kernel_types[i] == 'SquareExp')*np.convolve(df.values[:,i],\
                    Dkern_array_unnorm('SquareExp',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                    np.sum(kern_array('SquareExp',hsq[i],self.signal_periods[i]))) + \
                    ((self.column_kernel_types[i] == 'Periodic')*np.convolve(df.values[:,i],\
                    Dkern_array_unnorm('Periodic',hsq[i],self.signal_periods[i]))[:self.Ns]/\
                    np.sum(kern_array('Periodic',hsq[i],self.signal_periods[i]))) for i in range(0,self.Nd)]).T
        DlnHn_h = np.asarray([((self.column_kernel_types[i] == 'SquareExp')*np.sum(Dkern_array_unnorm('SquareExp',\
                  hsq[i],self.signal_periods[i]))/np.sum(kern_array('SquareExp',hsq[i],self.signal_periods[i]))) + \
                  ((self.column_kernel_types[i] == 'Periodic')*np.sum(Dkern_array_unnorm('Periodic',\
                  hsq[i],self.signal_periods[i]))/np.sum(kern_array('Periodic',hsq[i],self.signal_periods[i]))) \
                  for i in range(0,self.Nd)]).T
        Dconv_d = Dconv_d_u - (DlnHn_h*conv_d)
        
        # Return convolved signals
        return conv_d, Dconv_d
    
    
    # Function to output random posterior prediction samples corresponding to the the 'AR-GP' network 
    def pred_ARGP_sampler(self,ftime,nsamples):
        '''
        Function which generates posterior predictive samples for the N-dimensional time series using the 
        'AR-GP' network, where its hyperparameters have been optimised by applying bants.fit to a dataframe. 

        INPUT:

        ftime        -     This is the timepoint (in units of the index of the train_df) for the forecast 
                           to run up to from the training data endpoint.
        
        nsamples     -     This is the number of random predictive samples to request for at each timestep.
        
        OUTPUT:
        
        pred_samps   -     This is an output array of dimensions (nfut,nsamples).
        
        '''
        # Compute the number of timesteps to predict over from ftime
        delta_t = self.train_df.index[1]-self.train_df.index[0]
        nfut = int(np.ceil((ftime-self.train_df.index[-1])/delta_t))
        
        # Extract optimised hyperparameters
        nu_opt = self.params['nu']
        hsq_opt = self.params['hsq']
        Psi_opt = np.zeros((self.Nd,self.Nd))
        Psi_opt[np.tril_indices(self.Nd)] = self.params['Psi_tril']
             
        # Psi is symmetric
        Psi_opt = Psi_opt + Psi_opt.T - np.diag(Psi_opt.diagonal())

        # Loop over future samples so that the previous samples can be used iteratively in the kernel convolution
        pred_samps = [np.asarray([np.random.multivariate_normal(self.kconv(self.train_df,hsq_opt),\
                      spst.invwishart.rvs(df=nu_opt,scale=Psi_opt)) for ns in range(0,nsamples)])]
        for nf in range(1,nfut):
            
            # Generate samples from the 'AR-GP' network by drawing covariance matrices from the inverse-Wishart 
            # distribution, drawing the corresponding normal variates and then storing them to output
            pred_samps.append(np.asarray([np.random.multivariate_normal(\
                       self.kconv(self.train_df.append(pred_samps[nf-1][ns]),hsq_opt),\
                       spst.invwishart.rvs(df=nu_opt,scale=Psi_opt)) for ns in range(0,nsamples)]))
            
        # Convert to numpy array and output result
        pred_samps = np.asarray(pred_samps)
        return pred_samps


    # Subroutine for the 'AR-GP' network
    def optimise_ARGP_hyperp(self,df):
        '''
        Method to optimise the 'AG-GP' network hyperparameters as defined in notes/how_bants_works.ipynb.
        Optimisation outputs are written to bants.params and bants.info accordingly.
    
        INPUT:
    
        df    -     This is the input dataframe of values to optimise the hyperparameters with respect to.
                       
        '''
        # If not set make a first guess for nu, h and the lower triangular elements of Psi
        if self.nu_guess is None: 
            self.nu_guess = self.Nd
        if self.hsq_guess is None:
            # Create autocorrelation function to make initial guess for kernel scale
            def acf(df,dim,lag): return np.corrcoef(np.array([df.values[:-lag,dim], df.values[lag:,dim]]))[0,1]
            # Loop over dimensions and append kernel scale guesses
            hs = []
            for dim in range(0,self.Nd):
                acs = np.asarray([acf(df,dim,lag) for lag in range(1,self.Ns-1)])
                acs = acs*(acs>0.0)
                hs.append(np.sum(((df.index[1:-1]-df.index[0])**2.0)*acs)/np.sum(acs))
            # Set computed guesses
            self.hsq_guess = np.asarray(hs)
        if self.Psi_tril_guess is None:
            Mt = self.kconv(df,self.hsq_guess)
            self.Psi_tril_guess = np.cov(Mt.T)[np.tril_indices(self.Nd)]
    
        # With initial guesses for the parameters implement optimisation with bants.itmax as the maximum
        # number of iterations permitted for the algorithm and bants.lnEtol as the log-evidence tolerance.
        # Options are the Nelder-Mead algorithm...
        if self.optimiser == 'Nelder-Mead':
            
            # Define the function to optimise over to obtain optimal network hyperparameters if Nelder-Mead
            def func_to_opt(params,df=df,N=self.Nd):
            
                # Extract hyperparameters
                nu = np.exp(params[0])      # Choose log space for nu for scaling and to avoid negative values
                hsq = np.exp(params[1:N+1]) # Choose log space for hsq for scaling and to avoid negative values
                Psi = np.zeros((N,N))
                Psi[np.tril_indices(N)] = params[N+1:]
             
                # Psi is symmetric
                Psi = Psi + Psi.T - np.diag(Psi.diagonal())
            
                # Compute the kernel-convolved signal for each data point
                Mt = self.kconv(df,hsq)
            
                # Compute the scale matrix
                Sm = Psi/(nu-N+1.0)
            
                # Sum log-evidence contributions by each data point
                lnE = np.sum(self.tD_logpdf(df,nu-N+1.0,Mt,Sm),axis=0)
            
                # Output corresponding value to minimise
                return -lnE          
            
            # Run Nelder-Mead algorithm and obtain result
            init_params = tf.constant(np.append(np.append(np.log(self.nu_guess),\
                                      np.log(self.hsq_guess)),self.Psi_tril_guess))
            res = tfp.optimizer.nelder_mead_minimize(func_to_opt, initial_vertex=init_params, \
                                max_iterations=self.itmax, func_tolerance=self.lnEtol)
            
            # Output results of optimisation to bants.params dictionary
            self.params['nu'] = np.exp(res.position[0].numpy())
            self.params['hsq'] = np.exp(res.position[1:self.Nd+1].numpy())
            self.params['Psi_tril'] = res.position[self.Nd+1:].numpy()
        
            # Output fitting information to bants.info dictionary
            self.info['converged'] = res.converged.numpy()
            self.info['n_evaluations'] = res.num_objective_evaluations.numpy()
            self.info['lnE_val'] = -res.objective_value.numpy()
            
            # Store the posterior prediction sampler with optimised hyperparameters
            self.results['sampler'] = self.pred_ARGP_sampler

        # Or the BFGS algorithm...
        if self.optimiser == 'BFGS':
            
            # Define the function to optimise over to obtain optimal network hyperparameters if Nelder-Mead
            def func_to_opt(params,df=df,N=self.Nd):
            
                # Extract hyperparameters
                nu = np.exp(params[0])      # Choose log space for nu for scaling and to avoid negative values
                hsq = np.exp(params[1:N+1]) # Choose log space for hsq for scaling and to avoid negative values
                Psi = np.zeros((N,N))
                Psi[np.tril_indices(N)] = params[N+1:]
             
                # Psi is symmetric
                Psi = Psi + Psi.T - np.diag(Psi.diagonal())
            
                # Compute the kernel-convolved signal for each data point
                Mt = self.kconv(df,hsq)
            
                # Compute the scale matrix
                Sm = Psi/(nu-N+1.0)
            
                # Sum log-evidence contributions by each data point
                lnE = np.sum(self.tD_logpdf(df,nu-N+1.0,Mt,Sm),axis=0)
            
                # Output corresponding value to minimise
                return -lnE
            
            # Define the gradient of the function to optimise over to obtain optimal network hyperparameters if BFGS
            def Dfunc_to_opt(params,df=df,N=self.Nd):
            
                # Extract hyperparameters
                nu = np.exp(params[0])      # Choose log space for nu for scaling and to avoid negative values
                hsq = np.exp(params[1:N+1]) # Choose log space for hsq for scaling and to avoid negative values
                Psi = np.zeros((N,N))
                Psi[np.tril_indices(N)] = params[N+1:]
            
                # Psi is symmetric
                Psi = Psi + Psi.T - np.diag(Psi.diagonal())
            
                # Compute the kernel-convolved signal and its first derivative for each data point
                Mt, DMt = self.kconv_and_deriv(df,hsq)
            
                # Compute the gradient values
                x_minus_mu = df.values-Mt
                inversePsi = np.linalg.inv(Psi)
                inversePsisq = np.matmul(inversePsi,inversePsi)
                vect = np.tensordot(inversePsi,x_minus_mu,axes=([0],[1]))
                contraction = np.sum(x_minus_mu.T*vect,axis=0)
                DlnE = np.zeros_like(params)
                # Multiplying by nu here for logarithmic derivative
                DlnE[0] = nu*(((sps.digamma((nu+1.0)/2.0)-sps.digamma((nu-N+1.0)/2.0))/2.0) - \
                            np.sum(0.5*np.log(1.0+contraction)))
                # Multiplying by hsq here for logarithmic derivative
                DlnE[1:N+1] = -hsq*np.sum((nu+1.0)*DMt.T*vect/np.tensordot(np.ones(N),\
                                                              1.0+contraction,axes=0),axis=1)
                DlnE[N+1:] = -0.5*np.identity(N)[np.tril_indices(N)] + \
                              (((nu+1.0)/2.0)*np.sum(x_minus_mu[:,np.tril_indices(N)[0]].T*\
                              np.tensordot(inversePsisq[np.tril_indices(N)],np.ones(self.Ns),axes=0)*\
                              x_minus_mu[:,np.tril_indices(N)[1]].T/\
                              np.tensordot(np.ones_like(params[N+1:]),1.0+contraction,axes=0),axis=1))
                
                # Output corresponding value to minimise and its gradient
                return -DlnE
            
            # Run BFGS algorithm and obtain result with scipy optimiser
            # (tfp lbfgs optimiser didn't seem to work for some reason)
            init_params = np.append(np.append(np.log(self.nu_guess),\
                                              np.log(self.hsq_guess)),self.Psi_tril_guess)
            res = spo.minimize(func_to_opt, init_params, method='BFGS', jac=Dfunc_to_opt, \
                               options={'gtol': self.lnEtol, 'maxiter': self.itmax})
    
            # Output results of optimisation to bants.params dictionary
            self.params['nu'] = np.exp(res.x[0])
            self.params['hsq'] = np.exp(res.x[1:self.Nd+1])
            self.params['Psi_tril'] = res.x[self.Nd+1:]
        
            # Output fitting information to bants.info dictionary
            self.info['converged'] = res.success
            self.info['n_evaluations'] = res.nit
            self.info['lnE_val'] = -res.fun
            
            # Store the posterior prediction sampler with optimised hyperparameters
            self.results['sampler'] = self.pred_ARGP_sampler
            

    # Standardisation procedure of the training data
    def standardise(self):
        
        # Compute mean and standard deviation of the training dataframe
        mean_df = self.train_df.mean()
        std_df = self.train_df.std()
        
        # Apply standardisation to the training dataframe
        self.train_df = (self.train_df - mean_df)/std_df
        
        # Store the values in lists for later use in prediction
        self.mean_train_df.append(mean_df.values)
        self.std_train_df.append(std_df.values)
        
        # Store the fact that standardisation has been performed for prediction later
        self.standardised = True


    # Method of 'fit' is analogous to the scikit-learn pattern
    def fit(self,train_df,standard=True):
        '''
        Method to infer the network structure by computing the log Bayesian evidence through optimisation of the prior
        hyperparameters of the network. Learned network parameters can be found in the bants.params dictionary. Fitting 
        information can be found in the bants.info dictionary.

        INPUT:

        train_df     -     This is an input dataframe representing the training data of the vector time series process 
                           that one wishes to model. Simply set this to be a pandas dataframe with time as the index.
                       
        standard     -     (Optional) Boolean to specify if the input data is standardised prior to fitting or not. The
                           default option is 'True'.
                       
        '''
        # Make training data available to class object
        self.train_df = train_df
        
        # If standardisation is specified then perform this
        if standard==True: self.standardise()

        # Find dimensions and number of samples in dataset
        self.Nd = len(train_df.columns)
        self.Ns = len(train_df.index)

        # If number of columns is more than 1 then check to see if kernel types have been specified. If not then
        # simply set each column to the 'SquareExp' default option. 
        if (self.Nd > 1) and (len(self.column_kernel_types) == 1): 
            self.column_kernel_types = self.column_kernel_types*self.Nd
            
        # If number of columns is more than 1 then check to see if signal periods have been specified. If not then
        # simply set each column to the default option of 1.0. 
        if (self.Nd > 1) and (len(self.signal_periods) == 1):
            self.signal_periods = self.signal_periods*self.Nd
        
        # If 'AR-GP' network then run appropriate optimisation subroutine
        if self.net_type == 'AR-GP':
            self.optimise_ARGP_hyperp(self.train_df)


    # Method of 'predict' is analogous to the scikit-learn pattern
    def predict(self,ftime,validate=None):
        '''
        Method to make posterior predictions for the N-dimensional time series using the fitted 'bants' network. All of 
        the results from the prediction, including a future point sampler and potentially the results of the 
        cross-validation against the testing data (if this was provided) can be found in the bants.results dictionary.

        INPUT:
        
        ftime        -     This is the timepoint (in units of the index of the train_df) for the forecast 
                           to generate predictive distributions up to from the training data endpoint. 

        validate     -     (Optional) One can input a dataframe representing the testing data of the vector time series 
                           process to cross-validate the 'bants' predictions against. Simply set this to be a pandas 
                           dataframe of the same form as 'train_df' in the 'fit' method.
                       
        '''
        # Make prediction times and possible testing data available to class object
        self.predict_times = times
        if validate is not None: self.test_df = validate
            
        # Compute the number of timesteps to predict over from ftime
        delta_t = self.train_df.index[1]-self.train_df.index[0]
        nfut = int(np.ceil((ftime-self.train_df.index[-1])/delta_t))
        
        
        #if self.standardised == True:

