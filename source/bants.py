"""
BANTS - BAyesiaN graphical models for Time Series forecasting

This is the main 'bants' class to be used on generic N-dimensional datasets. The structure is intended to be as simple
as possible for use in a generic data science context. For more details on the mathematics behind 'bants' please refer
to the notes/theory-notes.ipynb Jupyter Notebook in the Git repository.

"""

import numpy as np
import pandas as pd
import scipy.special as sps
import scipy.optimize as spo
import scipy.stats as spst


# Initialize the 'bants' method class
class bants:
    # Initialisation needs only the model type as input.
    def __init__(self, net_type):
        """Choose net_type from 'AR-G' or 'KM-G'"""

        # Methods and functions
        self.fit
        self.predict
        self.optimise_ARG_hyperp
        self.pred_ARG_sampler
        self.kconv
        self.tD_logpdf
        self.standardise

        # Set model type
        self.net_type = net_type

        # Initialise empty dictionaries of model parameters to learn (and default hyperparameters of maximum number
        # of iterations, log-evidence tolerance and learning rate in the case of gradient descent), the fitting information
        # and prediction results
        self.params = {"itmax": 1000, "lnEtol": 0.0001, "learn_rate": 0.01}
        self.info = {}
        self.results = {}

        # Store the mean and standard deviation of the training data and store if standardisation has been performed
        self.mean_train_df = None
        self.std_train_df = None
        self.standardised = False

        # Set the default optimisation algorithm
        self.optimiser = "GD"

        # If model type is 'AR-G' then set kernel types
        if self.net_type == "AR-G":
            # Default type of convolution kernel for each column is always 'SquareExp'. The other option, for oscillatory
            # columns in the data over time, is 'Periodic'.
            self.column_kernel_types = ["SquareExp"]

            # Set signal periods in dimensions of the dataframe index (time variable chosen) - this should be a list of
            # the same length as the dimensions (columns) in the data where entries are relevant for 'Periodic' columns.
            self.signal_periods = [1.0]

            # Initial params and guesses for optimiser of the hyperparameters of the model
            self.nu = None
            self.hsq_guess = None
            self.Psi_tril_guess = None

            # Set the appropriate prediction sampler
            self.results["sampler"] = self.pred_ARG_sampler

        # If model type if 'KM-G' then import k-means clustering from tslearn and setup
        if self.net_type == "KM-G":
            from tslearn.clustering import TimeSeriesKMeans

            # Initialise the k-means clustering method and hyperparameters
            self.tsk = None
            self.TimeSeriesKMeans = TimeSeriesKMeans
            self.params["kmeans_nclus"] = None
            self.params["kmeans_max_iter"] = 10
            self.params["kmeans_random_state"] = 42
            self.results["kmeans_clust_cent"] = None

            # Initial params and guesses for optimiser of the hyperparameters of the model
            self.nu = None
            self.u_guess = None
            self.U_flat_guess = None
            self.Psi_tril_guess = None

            # Set the appropriate prediction sampler
            self.results["sampler"] = self.pred_KMG_sampler

    # Function to output multivariate t-distribution (see here: https://en.wikipedia.org/wiki/Multivariate_t-distribution)
    # log-Probability Density Function from input dataframe points. No scipy implementation so wrote this one.
    def tD_logpdf(self, df, nu, mu, Sigma):
        # Compute the log normalisation of the distribution using scipy loggammas
        log_norm = (
            sps.loggamma((nu + self.Nd) / 2.0)
            - sps.loggamma(nu / 2.0)
            - ((self.Nd / 2.0) * np.log(np.pi * nu))
            - (0.5 * np.log(np.linalg.det(Sigma)))
        )

        # Compute the log density function for each of the samples
        x_minus_mu = df.values - mu
        inverseSigma = np.linalg.inv(Sigma)
        contraction = np.sum(
            x_minus_mu.T * np.tensordot(inverseSigma, x_minus_mu, axes=([0], [1])),
            axis=0,
        )
        log_densfunc = np.sum(
            -((nu + self.Nd) / 2.0) * np.log(1.0 + (contraction / nu))
        )

        # Output result
        return log_norm + log_densfunc

    # Kernel convolution for the 'AR-G' model
    def kconv(self, df, hsq):
        """
        Method of kernel convolution on input dataframe values according to whichever kernel types were specified
        in the bants.column_kernel_types list. This is used mainly in bants.optimise_ARG_hyperp but can be used
        independently on different dataframes for experimentation.

        Args:
        df
            This is the input dataframe of values to perform the convolution on.
        hsq
            This is an input 1-d array of the same length as the number of columns in the dataframe
            containing the square-amplitude of the kernel scales for each time series dimension.

        Returns:
        conv_d
            This is an output array of convolved data the same shape as the input dataframe.

        """
        # Extract the change in time from the dataframe
        delta_t = df.index[1] - df.index[0]

        # Generate an array of centred distances to be used to make the window array of the kernel
        # remembering to account for the offset by one lag to depend only on past values
        dtimes = delta_t * np.arange(1, self.Ns + 1, 1)

        # Compute effective number of timesteps represented by each kernel scale for boundary correction
        hn_eff = (np.sqrt(hsq) / delta_t).astype(int)

        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type, hsq, perd):
            # Output weights corresponding to the convolution kernels
            if kern_type == "SquareExp":
                return np.exp(-(dtimes**2.0) / (2.0 * hsq))
            if kern_type == "Periodic":
                return np.exp(
                    -2.0 * ((np.sin(np.abs(np.pi * dtimes / perd))) ** 2.0) / hsq
                )

        # Evaluate the convolution on the data dependent on the choice of kernel with early boundary correction
        # and remembering to account for the offset by one lag to depend only on past values
        conv_d = np.asarray(
            [
                (
                    (self.column_kernel_types[i] == "SquareExp")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        kern_array("SquareExp", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("SquareExp", hsq[i], self.signal_periods[i]))
                )
                + (
                    (self.column_kernel_types[i] == "Periodic")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        kern_array("Periodic", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("Periodic", hsq[i], self.signal_periods[i]))
                )
                for i in range(0, self.Nd)
            ]
        ).T

        # Return convolved signals
        return conv_d

    # Kernel convolution and its first derivative for the 'AR-G' model
    def kconv_and_deriv(self, df, hsq):
        """
        Method of kernel convolution on input dataframe values according to whichever kernel types were specified
        in the bants.column_kernel_types list. This method also computes the first derivatives of the convolved
        data. This is used mainly in bants.optimise_ARG_hyperp with the gradient-based optimisers but can be used
        independently on different dataframes for experimentation.

        Args:
        df
            This is the input dataframe of values to perform the convolution on.
        hsq
            This is an input 1-d array of the same length as the number of columns in the dataframe
            containing the square-amplitude of the kernel scales for each time series dimension.

        Returns:
        (conv_d, Dconv_d)
            conv_d
                This is an output array of convolved data the same shape as the input dataframe.
            Dconv_d
                This is an output corresponding to the first derivative of the convolved data. Since the
                cross-derivatives of the corresponding Jacobian are zero, this is merely a 1-d array of the
                same length as conv_d.

        """
        # Extract the change in time from the dataframe
        delta_t = df.index[1] - df.index[0]

        # Generate an array of centred distances to be used to make the window array of the kernel
        # remembering to account for the offset by one lag to depend only on past values
        dtimes = delta_t * np.arange(1, self.Ns + 1, 1)

        # Compute effective number of timesteps represented by each kernel scale for boundary correction
        hn_eff = (np.sqrt(hsq) / delta_t).astype(int)

        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type, hsq, perd):
            # Output weights corresponding to the convolution kernels
            if kern_type == "SquareExp":
                return np.exp(-(dtimes**2.0) / (2.0 * hsq))
            if kern_type == "Periodic":
                return np.exp(
                    -2.0 * ((np.sin(np.abs(np.pi * dtimes / perd))) ** 2.0) / hsq
                )

        # Create function which returns an array corresponding to the convolution window function for
        # the chosen input kernel type.
        def kern_array(kern_type, hsq, perd):
            # Output weights corresponding to the convolution kernels
            if kern_type == "SquareExp":
                return np.exp(-(dtimes**2.0) / (2.0 * hsq))
            if kern_type == "Periodic":
                return np.exp(
                    -2.0 * ((np.sin(np.abs(np.pi * dtimes / perd))) ** 2.0) / hsq
                )

        # Create function which returns an array corresponding to the first derviative of the unnormalised
        # convolution window function for the chosen input kernel type.
        def Dkern_array_unnorm(kern_type, hsq, perd):
            # Output weights corresponding to the convolution kernels
            if kern_type == "SquareExp":
                return (dtimes**2.0 / (2.0 * hsq * hsq)) * np.exp(
                    -(dtimes**2.0) / (2.0 * hsq)
                )
            if kern_type == "Periodic":
                return (
                    2.0 * ((np.sin(np.abs(np.pi * dtimes / perd))) ** 2.0) / (hsq * hsq)
                ) * np.exp(
                    -2.0 * ((np.sin(np.abs(np.pi * dtimes / perd))) ** 2.0) / hsq
                )

        # Evaluate the convolution on the data dependent on the choice of kernel with early boundary correction
        # and remembering to account for the offset by one lag to depend only on past values
        conv_d = np.asarray(
            [
                (
                    (self.column_kernel_types[i] == "SquareExp")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        kern_array("SquareExp", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("SquareExp", hsq[i], self.signal_periods[i]))
                )
                + (
                    (self.column_kernel_types[i] == "Periodic")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        kern_array("Periodic", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("Periodic", hsq[i], self.signal_periods[i]))
                )
                for i in range(0, self.Nd)
            ]
        ).T

        # Evaluate the first derivative of convolution on the data dependent on the choice of kernel
        Dconv_d_u = np.asarray(
            [
                (
                    (self.column_kernel_types[i] == "SquareExp")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        Dkern_array_unnorm("SquareExp", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("SquareExp", hsq[i], self.signal_periods[i]))
                )
                + (
                    (self.column_kernel_types[i] == "Periodic")
                    * np.convolve(
                        np.append(
                            df.values[0, i] * np.ones(hn_eff[i]), df.values[:, i]
                        ),
                        Dkern_array_unnorm("Periodic", hsq[i], self.signal_periods[i]),
                    )[hn_eff[i] - 1 : self.Ns + hn_eff[i] - 1]
                    / np.sum(kern_array("Periodic", hsq[i], self.signal_periods[i]))
                )
                for i in range(0, self.Nd)
            ]
        ).T
        DlnHn_h = np.asarray(
            [
                (
                    (self.column_kernel_types[i] == "SquareExp")
                    * np.sum(
                        Dkern_array_unnorm("SquareExp", hsq[i], self.signal_periods[i])
                    )
                    / np.sum(kern_array("SquareExp", hsq[i], self.signal_periods[i]))
                )
                + (
                    (self.column_kernel_types[i] == "Periodic")
                    * np.sum(
                        Dkern_array_unnorm("Periodic", hsq[i], self.signal_periods[i])
                    )
                    / np.sum(kern_array("Periodic", hsq[i], self.signal_periods[i]))
                )
                for i in range(0, self.Nd)
            ]
        ).T
        Dconv_d = Dconv_d_u - (DlnHn_h * conv_d)

        # Return convolved signals in a tuple
        return conv_d, Dconv_d

    # Function to output random prediction samples corresponding to the the 'AR-G' model
    def pred_ARG_sampler(self, ftime, nsamples, compute_map=True):
        """
        Function which generates (posterior or MAP - MAP is default) predictive samples for the N-dimensional
        time series using the 'AR-G' model, where its hyperparameters have been optimised by applying
        bants.fit to a dataframe.

        Args:
        ftime
            This is the timepoint (in units of the index of the train_df) for the forecast
            to generate predictive distributions up to from the training data endpoint.
        nsamples
            This is the number of random predictive samples to request for at each timestep.

        Keywords:
        compute_map
            If True (which is the default) then compute the predictions with the MAP of the model.

        Returns:
        pred_samps
            This is an output array of dimensions (nfut,dim,nsamples), where dim is the number
            of dimensions in the vector time series.

        """
        # Compute the number of timesteps to predict over using ftime
        delta_t = self.train_df.index[1] - self.train_df.index[0]
        nfut = int(np.ceil((ftime - self.train_df.index[-1]) / delta_t))

        # Extract optimised hyperparameters
        hsq_opt = self.params["hsq"]
        Psi_opt = np.zeros((self.Nd, self.Nd))
        Psi_opt[np.tril_indices(self.Nd)] = self.params["Psi_tril"]

        # Psi is symmetric
        Psi_opt = Psi_opt + Psi_opt.T - np.diag(Psi_opt.diagonal())

        # Generate future timepoints and storage for predictions
        indices = np.asarray(
            [(float(nf) * delta_t) + self.train_df.index[-1] for nf in range(0, nfut)]
        )
        pred_samps = np.empty((nfut, self.Nd, 0))

        # Loop over samples (very slow at the moment) (this is a slow method with numpy/scipy and hence
        # will need to be customised in future to get more speed)
        for ns in range(0, nsamples):
            # Get the mode of the inverse-Wishart distribution if computing the MAP prediction
            if compute_map:
                Sigt = Psi_opt / (self.nu + self.Nd + 1.0)
            else:
                # Generate random covariance matrix from inverse-Wishart with optimal hyperparameters
                Sigt = spst.invwishart.rvs(df=self.nu, scale=Psi_opt)

            # Loop over future timepoints and generate samples from the 'AR-G' model iteratively
            out = [
                np.random.multivariate_normal(
                    self.kconv(self.train_df, hsq_opt)[-1], Sigt
                )
            ]
            for nf in range(1, nfut):
                # Generate updated dataframe using past predicted samples
                d_update = self.train_df.append(
                    pd.DataFrame(data=np.asarray(out)[:nf], index=indices[:nf])
                )

                # Draw the corresponding normal variate
                samp = np.random.multivariate_normal(
                    self.kconv(d_update, hsq_opt)[-1], Sigt
                )

                # Store sample to output
                out.append(samp)

            # Add new set of predicted future timepoints to output
            pred_samps = np.dstack((pred_samps, np.asarray(out)))

        # Output predictive samples with or without standardisation
        if self.standardised == True:
            mfact = np.tensordot(
                np.tensordot(np.ones(nfut), self.mean_train_df, axes=0),
                np.ones(nsamples),
                axes=0,
            )
            sfact = np.tensordot(
                np.tensordot(np.ones(nfut), self.std_train_df, axes=0),
                np.ones(nsamples),
                axes=0,
            )
            return mfact + (sfact * pred_samps)
        if self.standardised == False:
            return pred_samps

    # Subroutine for the 'AR-G' model
    def optimise_ARG_hyperp(self, df):
        """
        Method to optimise the 'AR-G' model hyperparameters as defined in notes/theory-notes.ipynb.
        Optimisation outputs are written to bants.params and bants.info accordingly.

        Args:
        df
            This is the input dataframe of values to optimise the hyperparameters with respect to.

        """
        # If not already set, then fix the number of degrees of freedom to correspond to the non-informative prior
        if self.nu is None:
            self.nu = self.Nd
        self.params["nu"] = self.nu

        # If not set make a first guess for h and the lower triangular elements of Psi
        if self.hsq_guess is None:
            # Create autocorrelation function to make initial guess for kernel scale
            def acf(df, dim, lag):
                return np.corrcoef(
                    np.array([df.values[:-lag, dim], df.values[lag:, dim]])
                )[0, 1]

            # Loop over dimensions and append kernel scale guesses
            hs = []
            for dim in range(0, self.Nd):
                acs = np.asarray([acf(df, dim, lag) for lag in range(1, self.Ns - 1)])
                acs = acs * (acs > 0.0)
                hs.append(
                    np.sum(((df.index[1:-1] - df.index[0]) ** 2.0) * acs) / np.sum(acs)
                )
            # Set computed guesses as converted array
            self.hsq_guess = np.asarray(hs)
        if self.Psi_tril_guess is None:
            Mt = self.kconv(df, self.hsq_guess)
            self.Psi_tril_guess = np.cov(Mt.T)[np.tril_indices(self.Nd)]

        # Define the function to optimise over to obtain optimal model hyperparameters
        def func_to_opt(params, df=df, N=self.Nd):
            # Extract hyperparameters
            hsq = np.exp(
                params[:N]
            )  # Choose log space for hsq for scaling and to avoid negative values
            Psi = np.zeros((N, N))
            Psi[np.tril_indices(N)] = params[N:]

            # Psi is symmetric
            Psi = Psi + Psi.T - np.diag(Psi.diagonal())

            # Compute the kernel-convolved signal for each data point
            Mt = self.kconv(df, hsq)

            # Compute the scale matrix
            Sm = Psi / (self.nu - N + 1.0)

            # Sum log-evidence contributions by each data point
            lnE = np.sum(self.tD_logpdf(df, self.nu - N + 1.0, Mt, Sm), axis=0)

            # Output corresponding value to minimise
            return -lnE

        # Define the gradient of the function to optimise over to obtain optimal model hyperparameters
        # if the chosen optimiser is gradient-based
        def Dfunc_to_opt(params, df=df, N=self.Nd):
            # Extract hyperparameters
            hsq = np.exp(
                params[:N]
            )  # Choose log space for hsq for scaling and to avoid negative values
            Psi = np.zeros((N, N))
            Psi[np.tril_indices(N)] = params[N:]

            # Psi is symmetric
            Psi = Psi + Psi.T - np.diag(Psi.diagonal())

            # Compute the kernel-convolved signal and its first derivative for each data point
            Mt, DMt = self.kconv_and_deriv(df, hsq)

            # Compute the gradient values
            x_minus_mu = df.values - Mt
            inversePsi = np.linalg.inv(Psi)
            inversePsisq = np.matmul(inversePsi, inversePsi)
            vect = np.tensordot(inversePsi, x_minus_mu, axes=([0], [1]))
            contraction = np.sum(x_minus_mu.T * vect, axis=0)
            DlnE = np.zeros_like(params)
            # Multiplying by hsq here for logarithmic derivative
            DlnE[:N] = -hsq * np.sum(
                (self.nu + 1.0)
                * DMt.T
                * vect
                / np.tensordot(np.ones(N), 1.0 + contraction, axes=0),
                axis=1,
            )
            DlnE[N:] = -0.5 * np.identity(N)[np.tril_indices(N)] + (
                ((self.nu + 1.0) / 2.0)
                * np.sum(
                    x_minus_mu[:, np.tril_indices(N)[0]].T
                    * np.tensordot(
                        inversePsisq[np.tril_indices(N)], np.ones(self.Ns), axes=0
                    )
                    * x_minus_mu[:, np.tril_indices(N)[1]].T
                    / np.tensordot(np.ones_like(params[N:]), 1.0 + contraction, axes=0),
                    axis=1,
                )
            )

            # Output corresponding value to minimise and its gradient
            return -DlnE

        # With initial guesses for the parameters implement optimisation with bants.params["itmax"] as the maximum
        # number of iterations permitted for the algorithm and bants.params["lnEtol"] as the log-evidence tolerance
        init_params = np.append(np.log(self.hsq_guess), self.Psi_tril_guess)

        # Run Nelder-Mead algorithm and obtain result with scipy optimiser
        if self.optimiser == "Nelder-Mead":
            res = spo.minimize(
                func_to_opt,
                init_params,
                method="Nelder-Mead",
                options={
                    "ftol": self.params["lnEtol"],
                    "maxiter": self.params["itmax"],
                },
            )

        # Run BFGS algorithm and obtain result with scipy optimiser
        if self.optimiser == "BFGS":
            res = spo.minimize(
                func_to_opt,
                init_params,
                method="L-BFGS-B",
                jac=Dfunc_to_opt,
                options={
                    "ftol": self.params["lnEtol"],
                    "maxiter": self.params["itmax"],
                },
            )

        # Run the GD algorithm (standard gradient descent) and obtain result
        if self.optimiser == "GD":
            # Initialise the results object, set the relevant hyperparameters and
            # then run the standard gradient descent algorithm
            res = results_obj(init_params, func_to_opt)
            lastf = res.fun
            lr = self.params["learn_rate"]
            ftol = self.params["lnEtol"]
            absdiff = ftol + 1.0
            while ((ftol < absdiff) & (res.nit < self.params["itmax"])) == True:
                # Iterate the loop, parameter and function values
                res.x -= lr * Dfunc_to_opt(res.x) / float(self.Ns)
                res.fun = func_to_opt(res.x)
                res.nit += 1

                # Compute difference in function values for tolerance
                absdiff = abs((res.fun - lastf) / lastf)
                lastf = res.fun

            # If specified tolerance was reached then trigger boolean
            if ftol > absdiff:
                res.success = True

        # Output results of optimisation to bants.params dictionary
        self.params["hsq"] = np.exp(res.x[: self.Nd])
        self.params["Psi_tril"] = res.x[self.Nd :]

        # Output fitting information to bants.info dictionary
        self.info["converged"] = res.success
        self.info["n_evaluations"] = res.nit
        self.info["lnE_val"] = -res.fun

    # Function to output random prediction samples corresponding to the the 'KM-G' model
    def pred_KMG_sampler(self, ftime, nsamples, compute_map=True, kmeans_window=50):
        """
        Function which generates (posterior or MAP - MAP is default) predictive samples for the N-dimensional
        time series using the 'KM-G' model, where its hyperparameters have been optimised by applying
        bants.fit to a dataframe.

        Args:
        ftime
            This is the timepoint (in units of the index of the train_df) for the forecast
            to generate predictive distributions up to from the training data endpoint.
        nsamples
            This is the number of random predictive samples to request for at each timestep.

        Keywords:
        compute_map
            If True (which is the default) then compute the predictions with the MAP of the model.
        kmeans_window
            Choose the window length of data used for refitting the k-means clustering for each
            predictive timestep.

        Returns:
        pred_samps
            This is an output array of dimensions (nfut,dim,nsamples), where dim is the number
            of dimensions in the vector time series.

        """
        # Compute the number of timesteps to predict over using ftime
        delta_t = self.train_df.index[1] - self.train_df.index[0]
        nfut = int(np.ceil((ftime - self.train_df.index[-1]) / delta_t))

        # Extract optimised hyperparameters
        u_opt = self.params["u"]
        U_opt = self.params["U_flat"].reshape((self.Nc, self.Nd))
        Psi_opt = np.zeros((self.Nd, self.Nd))
        Psi_opt[np.tril_indices(self.Nd)] = self.params["Psi_tril"]

        # Psi is symmetric
        Psi_opt = Psi_opt + Psi_opt.T - np.diag(Psi_opt.diagonal())

        # Create storage for predictions
        pred_samps = []

        # Get the mode of the inverse-Wishart distribution if computing the MAP prediction
        if compute_map:
            Sigt = [Psi_opt / (self.nu + self.Nd + 1.0) for ns in range(0, nsamples)]
        else:
            # Generate random covariance matrix from inverse-Wishart with optimal hyperparameters
            Sigt = [
                spst.invwishart.rvs(df=self.nu, scale=Psi_opt)
                for ns in range(0, nsamples)
            ]

        # Setup the samples to fit the k-means clustering to each new timestep
        datsamps = np.tensordot(
            self.train_df.values.swapaxes(0, 1), np.ones(nsamples), axes=0
        )[:, -kmeans_window:]

        # Loop over future timepoints (this is an unavoidably slow method if one wants to
        # use k-means clustering to compress the data at each timestep for consistency)
        for nf in range(0, nfut):
            # Fit the k-means clustering and compute the standardised centroids
            self.tsk.fit(datsamps)
            standardised_kmeans_clust_cent = (
                self.tsk.cluster_centers_.swapaxes(0, 1)
                - np.tensordot(
                    np.ones((kmeans_window, nsamples)),
                    self.results["kmeans_clust_means"],
                    axes=0,
                ).swapaxes(1, 2)
            ) / np.tensordot(
                np.ones((kmeans_window, nsamples)),
                self.results["kmeans_clust_stds"],
                axes=0,
            ).swapaxes(
                1, 2
            )

            # Use the optimal model hyperparameters to compute the new means for prediction
            mt = standardised_kmeans_clust_cent
            Mt = np.tensordot(mt, U_opt, axes=([1], [0])) + np.tensordot(
                np.ones((kmeans_window, nsamples)), u_opt, axes=0
            )

            # Generate predictive samples using a multivariate normal
            samps = np.asarray(
                [
                    np.random.multivariate_normal(Mt[-1, ns], Sigt[ns])
                    for ns in range(0, nsamples)
                ]
            )

            # Add new set of predicted future timepoints to output
            pred_samps.append(samps)

            # Add to the ensemble of data samples used for k-means clustering fit and
            # iterate the window forward in time
            datsamps = np.append(datsamps, samps[:, np.newaxis].swapaxes(0, 2), axis=1)[
                :, 1:
            ]

        # Reshape to match output
        pred_samps = np.asarray(pred_samps).swapaxes(1, 2)

        # Output predictive samples with or without standardisation
        if self.standardised == True:
            mfact = np.tensordot(
                np.tensordot(np.ones(nfut), self.mean_train_df, axes=0),
                np.ones(nsamples),
                axes=0,
            )
            sfact = np.tensordot(
                np.tensordot(np.ones(nfut), self.std_train_df, axes=0),
                np.ones(nsamples),
                axes=0,
            )
            return mfact + (sfact * pred_samps)
        if self.standardised == False:
            return pred_samps

    # Subroutine for the 'KM-G' model
    def optimise_KMG_hyperp(self, df):
        """
        Method to optimise the 'KM-G' model hyperparameters as defined in notes/theory-notes.ipynb.
        Optimisation outputs are written to bants.params and bants.info accordingly.

        Args:
        df
            This is the input dataframe of values to optimise the hyperparameters with respect to.

        """
        # If not already set, then fix the number of degrees of freedom to correspond to the non-informative prior
        if self.nu is None:
            self.nu = self.Nd
        self.params["nu"] = self.nu

        # If not set make a first guess for u and the lower triangular elements of U and Psi
        if self.u_guess is None:
            self.u_guess = np.zeros(self.Nd)
        if self.U_flat_guess is None:
            self.U_flat_guess = np.random.uniform(0, 1, size=int(self.Nc * self.Nd))
            self.U_flat_guess = self.U_flat_guess / np.sum(self.U_flat_guess)
        if self.Psi_tril_guess is None:
            vals = np.random.uniform(0, 1, size=(self.Nd, self.Ns))
            self.Psi_tril_guess = np.cov(vals)[np.tril_indices(self.Nd)]

        # Define the function to optimise over to obtain optimal model hyperparameters
        def func_to_opt(params, df=df, N=self.Nd, Nc=self.Nc):
            # Extract hyperparameters
            u = params[:N]
            U = params[N : N + (Nc * N)].reshape((Nc, N))
            Psi = np.zeros((N, N))
            Psi[np.tril_indices(N)] = params[N + (Nc * N) :]

            # Psi is symmetric
            Psi = Psi + Psi.T - np.diag(Psi.diagonal())

            # Calculate the mean for each data point using k-means and the model parameters
            mt = self.results["standardised_kmeans_clust_cent"]
            Mt = np.tensordot(mt, U, axes=([1], [0])) + np.tensordot(
                np.ones(self.Ns), u, axes=0
            )

            # Compute the scale matrix
            Sm = Psi / (self.nu - N + 1.0)

            # Sum log-evidence contributions by each data point
            lnE = np.sum(self.tD_logpdf(df, self.nu - N + 1.0, Mt, Sm), axis=0)

            # Output corresponding value to minimise
            return -lnE

        # Get the flattened U indices for convenient indexing in the first derivatives
        iflat = (
            np.tensordot(np.arange(0, self.Nc, 1), np.ones(self.Nd), axes=0)
            .astype(int)
            .flatten()
        )
        jflat = (
            np.tensordot(np.ones(self.Nc), np.arange(0, self.Nd, 1), axes=0)
            .astype(int)
            .flatten()
        )

        # Define the gradient of the function to optimise over to obtain optimal model hyperparameters
        # if the chosen optimiser is gradient-based
        def Dfunc_to_opt(
            params, df=df, N=self.Nd, Nc=self.Nc, iflat=iflat, jflat=jflat
        ):
            # Extract hyperparameters
            u = params[:N]
            U = params[N : N + (Nc * N)].reshape((Nc, N))
            Psi = np.zeros((N, N))
            Psi[np.tril_indices(N)] = params[N + (Nc * N) :]

            # Psi is symmetric
            Psi = Psi + Psi.T - np.diag(Psi.diagonal())

            # Calculate the mean for each data point using k-means and the model parameters
            mt = self.results["standardised_kmeans_clust_cent"]
            Mt = np.tensordot(mt, U, axes=([1], [0])) + np.tensordot(
                np.ones(self.Ns), u, axes=0
            )

            # Compute the gradient values
            x_minus_mu = df.values - Mt
            inversePsi = np.linalg.inv(Psi)
            inversePsisq = np.matmul(inversePsi, inversePsi)
            vect = np.tensordot(inversePsi, x_minus_mu, axes=([0], [1]))
            contraction = np.sum(x_minus_mu.T * vect, axis=0)
            DlnE = np.zeros_like(params)
            DlnE[:N] = np.sum(
                (self.nu + 1.0)
                * vect
                / np.tensordot(np.ones(N), 1.0 + contraction, axes=0),
                axis=1,
            )
            DlnE[N : N + (Nc * N)] = np.sum(
                (self.nu + 1.0)
                * mt.T[iflat]
                * vect[jflat]
                / np.tensordot(np.ones(Nc * N), 1.0 + contraction, axes=0),
                axis=1,
            )
            DlnE[N + (Nc * N) :] = -0.5 * np.identity(N)[np.tril_indices(N)] + (
                ((self.nu + 1.0) / 2.0)
                * np.sum(
                    x_minus_mu[:, np.tril_indices(N)[0]].T
                    * np.tensordot(
                        inversePsisq[np.tril_indices(N)], np.ones(self.Ns), axes=0
                    )
                    * x_minus_mu[:, np.tril_indices(N)[1]].T
                    / np.tensordot(
                        np.ones_like(params[N + (Nc * N) :]), 1.0 + contraction, axes=0
                    ),
                    axis=1,
                )
            )

            # Output corresponding value to minimise and its gradient
            return -DlnE

        # With initial guesses for the parameters implement optimisation with bants.params["itmax"] as the maximum
        # number of iterations permitted for the algorithm and bants.params["lnEtol"] as the log-evidence tolerance
        init_params = np.append(
            np.append(self.u_guess, self.U_flat_guess), self.Psi_tril_guess
        )

        # Run Nelder-Mead algorithm and obtain result with scipy optimiser
        if self.optimiser == "Nelder-Mead":
            res = spo.minimize(
                func_to_opt,
                init_params,
                method="Nelder-Mead",
                options={
                    "ftol": self.params["lnEtol"],
                    "maxiter": self.params["itmax"],
                },
            )

        # Run BFGS algorithm and obtain result with scipy optimiser
        if self.optimiser == "BFGS":
            res = spo.minimize(
                func_to_opt,
                init_params,
                method="L-BFGS-B",
                jac=Dfunc_to_opt,
                options={
                    "ftol": self.params["lnEtol"],
                    "maxiter": self.params["itmax"],
                },
            )

        # Run the GD algorithm (standard gradient descent) and obtain result
        if self.optimiser == "GD":
            # Initialise the results object, set the relevant hyperparameters and
            # then run the standard gradient descent algorithm
            res = results_obj(init_params, func_to_opt)
            lastf = res.fun
            lr = self.params["learn_rate"]
            ftol = self.params["lnEtol"]
            absdiff = ftol + 1.0
            while ((ftol < absdiff) & (res.nit < self.params["itmax"])) == True:
                # Iterate the loop, parameter and function values
                res.x -= lr * Dfunc_to_opt(res.x) / float(self.Ns)
                res.fun = func_to_opt(res.x)
                res.nit += 1

                # Compute difference in function values for tolerance
                absdiff = abs((res.fun - lastf) / lastf)
                lastf = res.fun

            # If specified tolerance was reached then trigger boolean
            if ftol > absdiff:
                res.success = True

        # Output results of optimisation to bants.params dictionary
        self.params["u"] = res.x[: self.Nd]
        self.params["U_flat"] = res.x[self.Nd : self.Nd + (self.Nc * self.Nd)]
        self.params["Psi_tril"] = res.x[self.Nd + (self.Nc * self.Nd) :]

        # Output fitting information to bants.info dictionary
        self.info["converged"] = res.success
        self.info["n_evaluations"] = res.nit
        self.info["lnE_val"] = -res.fun

    # Standardisation procedure of the training data
    def standardise(self):
        # Compute mean and standard deviation of the training dataframe
        mean_df = self.train_df.mean()
        std_df = self.train_df.std()

        # Apply standardisation to the training dataframe
        self.train_df = (self.train_df - mean_df) / std_df

        # Store the values for later use in prediction
        self.mean_train_df = mean_df.values
        self.std_train_df = std_df.values

        # Store the fact that standardisation has been performed for prediction later
        self.standardised = True

    # Method of 'fit' is analogous to the scikit-learn pattern
    def fit(self, train_df, standard=True):
        """
        Method to infer the model structure by computing the log Bayesian evidence through optimisation of the prior
        hyperparameters of the model. Learned model parameters can be found in the bants.params dictionary. Fitting
        information can be found in the bants.info dictionary.

        Args:
        train_df
            This is an input dataframe representing the training data of the vector time series process
            that one wishes to model. Simply set this to be a pandas dataframe with time as the index.

        Keywords:
        standard
            Boolean to specify if the input data is standardised prior to fitting or not. The default is 'True'.

        """
        # Make training data available to class object
        self.train_df = train_df

        # If standardisation is specified and hasn't already been done then perform this
        if standard == True and self.standardised == False:
            self.standardise()

        # Find number of samples in dataset
        self.Ns = len(train_df.index)

        # If 'AR-G' model then run appropriate optimisation subroutine
        if self.net_type == "AR-G":
            # Find dimensions in dataset
            self.Nd = len(train_df.columns)

            # If number of columns is more than 1 then check to see if kernel types have been specified. If not then
            # simply set each column to the 'SquareExp' default option.
            if (self.Nd > 1) and (len(self.column_kernel_types) == 1):
                self.column_kernel_types = self.column_kernel_types * self.Nd

            # If number of columns is more than 1 then check to see if signal periods have been specified. If not then
            # simply set each column to the default option of 1.0.
            if (self.Nd > 1) and (len(self.signal_periods) == 1):
                self.signal_periods = self.signal_periods * self.Nd

            self.optimise_ARG_hyperp(self.train_df)

        # If 'KM-G' model then run appropriate optimisation subroutines
        if self.net_type == "KM-G":
            # Set the number of clustering and dataset dimensions
            self.Nc = self.params["kmeans_nclus"]
            self.Nd = len(train_df.columns)

            if self.tsk is None:
                self.tsk = self.TimeSeriesKMeans(
                    n_clusters=self.params["kmeans_nclus"],
                    max_iter=self.params["kmeans_max_iter"],
                    metric="dtw",
                    random_state=self.params["kmeans_random_state"],
                )
            if self.results["kmeans_clust_cent"] is None:
                self.tsk.fit(
                    self.train_df.values.reshape((self.Ns, self.Nd, 1)).swapaxes(0, 1)
                )
                self.results["kmeans_clust_cent"] = self.tsk.cluster_centers_.swapaxes(
                    0, 1
                )[:, :, 0]
                self.results["kmeans_clust_means"] = np.mean(
                    self.results["kmeans_clust_cent"], axis=0
                )
                self.results["kmeans_clust_stds"] = np.std(
                    self.results["kmeans_clust_cent"], axis=0
                )
                self.results["standardised_kmeans_clust_cent"] = (
                    self.results["kmeans_clust_cent"]
                    - np.tensordot(
                        np.ones(self.Ns),
                        self.results["kmeans_clust_means"],
                        axes=0,
                    )
                ) / np.tensordot(
                    np.ones(self.Ns),
                    self.results["kmeans_clust_stds"],
                    axes=0,
                )
            self.optimise_KMG_hyperp(self.train_df)

    # Method of 'predict' is analogous to the scikit-learn pattern
    def predict(self, ftime, nsamples, compute_map=True, **kwargs):
        """
        Method to generate (posterior or MAP - MAP is default) predictive samples for the N-dimensional
        time series using any chosen model, where its hyperparameters must have already been optimised
        by applying bants.fit to a dataframe. The future point sampler which can also be found in the
        bants.results dictionary.

        Args:
        ftime
            This is the timepoint (in units of the index of the train_df) for the forecast
            to generate predictive distributions up to from the training data endpoint.
        nsamples
            This is the number of random predictive samples to request for at each timestep.

        Keywords:
        compute_map
            If True (which is the default) then compute the predictions with the MAP of the model.
        kmeans_window
            This is used only with the 'KM-G' model. Choose the window length of data used for
            refitting the k-means clustering for each predictive timestep.

        Returns:
        pred_samps
            This is an output array of dimensions (nfut,dim,nsamples), where dim is the number
            of dimensions in the vector time series.

        """
        # Generate and output the predictions
        return self.results["sampler"](
            ftime, nsamples, compute_map=compute_map, **kwargs
        )


class results_obj:
    def __init__(self, x, f):
        """Define a results object for output"""
        self.x = x
        self.fun = f(x)
        self.nit = 0
        self.success = False
