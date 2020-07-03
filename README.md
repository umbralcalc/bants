# BANTS

### BAyesian Networks for Time Series forecasting

This is a python class for building Bayesian networks to forecast generic n-dimensional time series data. 

#### Current networks one can use

- `'AR-GP'`: This network takes the time series, coverts it into autoregressive features using kernel convolutions and then couples the resulting variances in a Gaussian process model. The model assumes Dirac delta distribution priors over the kernel scales and an inverse-Wishart prior over the covariance matrix over the convolved signals. 

Work is still in progress on the class so other networks have yet to be coded.
