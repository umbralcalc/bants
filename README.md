# BANTS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4955584.svg)](https://doi.org/10.5281/zenodo.4955584)

### BAyesian Networks for Time Series forecasting

This is a python class for building Bayesian networks to forecast generic n-dimensional time series data. 

#### Current networks one can use

- `'AR-GP'`: This network takes the time series, compresses it into autoregressive features using kernel convolutions and then couples the resulting variances in a Gaussian process model. The model assumes Dirac delta distribution priors over the kernel scales and an inverse-Wishart prior over the covariance matrix over the convolved signals. 

- `'KM-GP'`: This network takes the time series, compresses it with k-means clustering with dynamic time warping (using the `tslearn` package) and then couples the resulting variances in a Gaussian process model. As above, the hyperparameters at the compression step are assumed to have Dirac delta distribution priors and an inverse-Wishart prior is used over the covariance matrix for the compressed signals.

The theoretical documentation on how each network is constructed can be found in `notes/theory-notes.ipynb`.
