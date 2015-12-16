#!/usr/bin/env python
# coding: utf-8

"""
emcee Bayesean deconvolution in one dimension
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import datetime

import pystan
import pickle
from hashlib import md5
import tempfile
import os

import emcee
import util
from scipy.stats import poisson, norm, lognorm

import processify

from joblib import Parallel, delayed
import multiprocessing

import pywt

# note: check this ref - http://arxiv.org/abs/1008.4686

def multiscale_recompose(coeffs_array, split_indices, N=None):
    """
    inverse of multiscale_decompose
    """
    coeffs_list = np.split(coeffs_array, split_indices)
    log_timeseries = pywt.waverec(coeffs_list, 'haar', pywt.MODES.cpd)
    timeseries = np.exp(log_timeseries)
    if N is not None:
        timeseries = timeseries[:N]
    return timeseries

def multiscale_decompose(timeseries):
    """
    decompose time-series into multiscale coefficients
    
    Presently uses the haar discrete wavelet transform of the log-timeseries
    """
    log_timeseries = np.log(timeseries)
    coeffs_list = pywt.wavedec(log_timeseries, 'haar', pywt.MODES.cpd)
    coeffs_array = np.concatenate(coeffs_list)
    split_indices = np.cumsum(np.array([len(c) for c in coeffs_list[:-1]]))
    return coeffs_array, split_indices


def fft_decompose(timeseries):
    """
    transform the timeseries into fourier components
    
    The real and imaginary parts interleaved into a real-valued array
    """
    parameters_complex = np.fft.rfft(timeseries)
    assert (parameters_complex.dtype == np.complex128), 'wrong assumption'
    parameters_real = parameters_complex.view(np.float64)
    return parameters_real

def fft_recompose(parameters_real):
    """
    Inverse of fft_decompose
    """
    parameters_complex = parameters_real.view(np.complex128)
    timeseries = np.fft.irfft(parameters_complex)
    return timeseries

def detector_model(true_count_rate, prf, detector_background):
    """
    Compute detector response
    
    Parameters
    ----------
    true_count_rate : 1-d ndarray
        true count rate in counts per sample interval
    
    prf : 1-d array
        one-sided point-response-function 
    
    Returns
    -------
    detector_count_rate : 1-d array
        detector count rate, i.e. true_count_rate convolved with prf
    """
    detector_count_rate = np.convolve(true_count_rate, prf, mode='valid')
    detector_count_rate += detector_background
    return detector_count_rate

#
# ... define log-prior, log-liklihood and log-probability.
# ... This pattern is provided by the emcee user manual.
# ... ref: http://dan.iel.fm/emcee/current/user/line/
#

def lnprior_uniform(p):
    """
    Uniform prior, p bounded at 0
    """
    # The parameters are stored as a vector of values
    ret = 0.0
    # lower bound of zero on true values
    if p.min() <= 0:
        ret =  -np.inf
    return ret


def lnprior_lognorm(p):
    """
    Log-normal prior
    """
    # note - for parameter definitions see 
    # http://nbviewer.ipython.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/lognormal_distributions.ipynb
    mu = 7.3 # Mean of log(X)
    sigma = 0.9 # Standard deviation of log(X)
    shape = sigma # Scipy's shape parameter
    scale = np.exp(mu) # Scipy's scale parameter
    ret = lognorm.logpdf(p, shape, loc=0, scale=scale)
    ret = ret.sum()
    return ret


def running_lognormal_parameters(p, observed_data, n_window=10, sigma_factor=2):
    """
    lognormal parameters fitting windowed observed data
    """
    mu = np.zeros(len(p))
    sigma = np.zeros(len(p))
    logobs = np.log(observed_data)
    N = len(p) - len(observed_data)
    for ii in range(len(observed_data) - n_window):
        windata = logobs[ii:ii+n_window]
        mu[ii+N] = windata.mean()
        sigma[ii+N] = windata.std()+np.log(sigma_factor)
    #extrapolate values at start and end of data series
    mu[:N] = mu[N]
    sigma[:N] = sigma[N]
    mu[-n_window:] = mu[-n_window-1]
    sigma[-n_window:] = sigma[-n_window-1]
    return mu, sigma


def lnprior_lognorm_from_data(p, observed_data, n_window=10, sigma_factor=2, _cache=[]):
    """
    Log-normal prior where the distribution parameters vary with time
    """
    if len(_cache) == 0 or len(_cache[0]) != len(p):
        mu, sigma = running_lognormal_parameters(p, observed_data, 
                                                 n_window=n_window, sigma_factor=sigma_factor)
        while len(_cache) > 0:
            _cache.pop()
        _cache.extend((mu,sigma))
    else:
        mu, sigma = _cache
    shape = sigma # Scipy's shape parameter
    scale = np.exp(mu) # Scipy's scale parameter    
    ret = lognorm.logpdf(p, shape, loc=0, scale=scale)
    ret = ret.sum()
    return ret
    
def lnprior_difference(p):
    """
    log-normal prior on step-by-step changes
    """
    # Parameters must all be > 0
    if p.min() <= 0:
        lp =  -np.inf
    else:
        dpdt = np.diff(np.log(p))
        mu = 0.0  # mean expected change - no change
        #sigma = np.log(2) #standard deviation - factor of two change
        sigma = np.log(1.5)
        #sigma = np.log(1.05) # much more smoothing - has an effect on the peak but not on the baseline noise
        lp = norm.logpdf(dpdt, mu, sigma).sum()
    return lp


def lnlike(p, observed_counts, one_sided_prf, detector_background):
    detector_count_rate = detector_model(p, one_sided_prf, detector_background)
    lp = poisson.logpmf(observed_counts, detector_count_rate)
    lp = lp.sum()
    return lp


def lnprior(p, observed_counts, one_sided_prf, detector_background, model_version='lognormal_difference'):
    if model_version == 'lognormal':
        lp = lnprior_lognorm(p)
    elif model_version == 'lognormal_difference':
        lp = lnprior_difference(p)
    elif model_version == 'uniform':
        lp = lnprior_uniform(p)
    elif model_version == 'lognormal_from_data':
        lp = lnprior_lognorm_from_data(p, observed_counts)
    else:
        raise ValueError("Unknown model_version={}".format(model_version))
    return lp

def lnprob(p, observed_counts, one_sided_prf, detector_background, model_version='lognormal_difference'):
    if model_version == 'lognormal':
        lp = lnprior_lognorm(p)
    elif model_version == 'lognormal_difference':
        lp = lnprior_difference(p)
    elif model_version == 'uniform':
        lp = lnprior_uniform(p)
    elif model_version == 'lognormal_from_data':
        lp = lnprior_lognorm_from_data(p, observed_counts)
    else:
        raise ValueError("Unknown model_version={}".format(model_version))
    
    if np.isfinite(lp):
        lp += lnlike(p, observed_counts, one_sided_prf, detector_background)
    else:
        lp = -np.inf
    return lp


def lnlike_multiscale(p, observed_counts, one_sided_prf, detector_background):
    # reconstruct true count rate from multiscale parameters
    N = len(observed_counts) + len(one_sided_prf) - 1
    # perform a decomposition to get the split points
    _, split_idx = multiscale_decompose(np.ones(N))
    true_count_rate = multiscale_recompose(p, split_idx, N)
    lp = lnlike(true_count_rate, observed_counts, one_sided_prf, 
                detector_background)
    return lp

def lnprob_multiscale(p, observed_counts, one_sided_prf, detector_background, 
                                model_version='multiscale'):
    # prior - unconstrained
    lp = 0.0
    # likleyhood
    lp += lnlike_multiscale(p, observed_counts, one_sided_prf, detector_background)
    return lp

def lnprob_fft(p, observed_counts, one_sided_prf, detector_background, 
                                model_version='fft'):
    # prior - unconstrained
    lp = 0.0
    # likleyhood
    true_count_rate = fft_recompose(p)
    lp += lnlike(true_count_rate, observed_counts, one_sided_prf, detector_background)
    # guard against nan
    if not np.isfinite(lp):
        lp = -np.inf
    return lp


def gen_initial_guess(observed_counts, one_sided_prf, Nguess, randscale=1e-3, reg='tv'):
    """
    generate a list of initial guesses based on the RL deconvolution
    """
    M = len(one_sided_prf)
    symmetric_prf = np.r_[np.zeros(M-1), one_sided_prf]
    Ndim = len(observed_counts) + M - 1
    # pad first to avoid end effects
    pad0 = np.ones(2*M)*observed_counts[0]
    pad1 = np.ones(M)*observed_counts[-1]
    observed_counts_padded = np.r_[pad0, observed_counts, pad1]

    initial_guess = util.deconvlucy1d(observed_counts_padded, symmetric_prf, 
                                     iterations=1000, reg=reg)
    
    ## smooth the initial guess
    #from micromet.util import smooth
    #initial_guess = smooth(initial_guess, 11)
    
    initial_guess = initial_guess[M+1:-M]
    
    
    # starting locations for each walker are in a small range around the
    # Richardson-Lucy starting point
    p0 = [initial_guess*2**(np.random.randn(Ndim)*randscale) for 
                            ii in range(Nguess)]
    
    #p0 = [initial_guess + np.random.randn(Ndim)*randscale for 
    #                        ii in range(Nguess)]
    return p0



def emcee_deconvolve(t, observed_counts, one_sided_prf, background_count_rate, 
                    iterations=500, nthreads=1, model_version='lognormal',
                    keep_burn_in_samples=False, thin=5, initial_guess_reg='tv',
                    a=2.0, walkers_per_dim=3):
    """
    Use emcee to deconvolve the observations
    """
    print("emcee_deconvolve: iterations={}, version={}".format(iterations, 
                                                               model_version))
    M = len(one_sided_prf)
    symmetric_prf = np.r_[np.zeros(M-1), one_sided_prf]
    
    # work out number of dimensions, Ndim
    N_true_count_rate = len(observed_counts) + M - 1
    if model_version == 'multiscale':
        coeffs, split_indices = multiscale_decompose(np.ones(N_true_count_rate))
        Ndim = len(coeffs)
    elif model_version == 'fft':
        coeffs = fft_decompose(np.ones(N_true_count_rate))
        Ndim = len(coeffs)
    else:
        Ndim = N_true_count_rate

    # Number of walkers needs to be at least 2x number of dimensions
    Nwalker = Ndim * walkers_per_dim
    Nwalker = max(Nwalker, 256) # don't run with less than 256 walkers
    #Nwalker = max(Nwalker, 700) # for testing
    # number of walkers must be even.
    # increment to the next multiple of 64 (for, maybe, easier load balancing)
    Nwalker += (64 - Nwalker % 64)
    
    # a starting position from a small number of Richardson-Lucy iterations
    p0 = gen_initial_guess(observed_counts, one_sided_prf, Nguess=Nwalker, 
                           randscale=1e-3, reg=initial_guess_reg)
    
    # transform the parameters, if we're using the multiscale or fft models
    if model_version == 'multiscale':
        for ii in range(Nwalker):
            p0[ii], split_indices = multiscale_decompose(p0[ii])
    elif model_version == 'fft':
        #check decompose/recompose maintins vector length
        assert len(p0[0]) == len(fft_recompose(fft_decompose(p0[0])))
        assert len(p0) % 2 == 0, 'FFT model requires input length to be even'
        for ii in range(Nwalker):
            p0[ii] = fft_decompose(p0[ii])
    
    args = (observed_counts,one_sided_prf,background_count_rate, model_version)
    if model_version == 'multiscale':
        sampler = emcee.EnsembleSampler(Nwalker,Ndim,lnprob_multiscale,
                                        args=args,
                                        threads=nthreads,
                                        a=a)
    elif model_version == 'fft':
        sampler = emcee.EnsembleSampler(Nwalker,Ndim,lnprob_fft,
                                        args=args,
                                        threads=nthreads,
                                        a=a)
    else:
        sampler = emcee.EnsembleSampler(Nwalker,Ndim,lnprob,
                                        args=args,
                                        threads=nthreads,
                                        a=a)
        # experiment with PTSampler
        #ntemps = 20
        ## initial guess needs shape  (ntemps, nwalkers, ndim)
        #p0 = np.array(p0)[np.newaxis, :, :]
        #p0 = np.concatenate([p0]*ntemps)
        #sampler = emcee.PTSampler(ntemps, Nwalker, Ndim, lnlike, lnprior,
        #                                logpargs=args,
        #                                loglargs=args[:-1],
        #                                threads=nthreads)
    
    # burn-in and discard
    pos,prob,state = sampler.run_mcmc(p0, iterations, 
                                storechain=keep_burn_in_samples, thin=thin)
    
    # sample
    pos,prob,state = sampler.run_mcmc(pos, iterations, thin=thin)
    
    # transform model parameters back to the true_count_rate timeseries
    if model_version == 'multiscale':
        Nsamples = sampler.flatchain.shape[0]
        A = np.zeros((Nsamples, N_true_count_rate))
        for ii in range(Nsamples):
            A[ii,:] = multiscale_recompose(sampler.flatchain[ii,:],
                                           split_indices,
                                           N_true_count_rate)
    elif model_version == 'fft':
        Nsamples = sampler.flatchain.shape[0]
        A = np.zeros((Nsamples, N_true_count_rate))
        for ii in range(Nsamples):
            A[ii,:] = fft_recompose(sampler.flatchain[ii,:])
    else:
        A = sampler.flatchain
    
    mean_est = A.mean(axis=0)
    low = np.percentile(A, 10.0, axis=0)
    high = np.percentile(A, 90.0, axis=0)

    t_ret = util.extrapolate_time(t, M-1)
    
    return sampler, A, t_ret, mean_est, low, high

def deconv1d_df_wrapper(args_kwargs_tuple):
    func, args, kwargs = args_kwargs_tuple
    print(args_kwargs_tuple)
    return func(*args, **kwargs)

def deconv1d_df(t, observed_counts, one_sided_prf, background_count_rate, column_name='deconv', same_time=True,
               deconv_func=emcee_deconvolve, **kwargs):
    """
    deconvolve and then return results in a pandas.DataFrame
    """
    #print("working on chunk with length {}".format(len(observed_counts)))
    with util.timewith("deconvolve chunk with {} elements".format(len(observed_counts))) as timer:
        results = deconv_func(t, observed_counts, one_sided_prf, 
                                        background_count_rate, **kwargs)
        sampler, A, t_ret = results[:3]
        mean_est = A.mean(axis=0)
        percentiles = np.percentile(A, [10, 16, 50, 84, 90], axis=0)
        d = {column_name + '_mean': mean_est,
             column_name + '_p10': percentiles[0],
             column_name + '_p16': percentiles[1],
             column_name + '_p50': percentiles[2],
             column_name + '_p84': percentiles[3],
             column_name + '_p90': percentiles[4]}
    df = pd.DataFrame(data=d, index=t_ret)
    if same_time:
        df = df.ix[t]
    return df


def deconv_in_chunks(df, column_name, one_sided_prf, background_count_rate, max_chunklen=240, deconv_func=emcee_deconvolve, n_jobs=1, deconv_func_kwargs=dict(), 
parallel_backend='direct_multiprocessing'):
    """
    Baysean Deconvolution based on emcee
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data Frame containing the time-series to be deconvolved
    
    column_name : string
        Name of the column to deconvolve
    
    one_sided_prf : one-dimensional array
        The point-response function (one sided - i.e. output depends only on past
        values)
    
    
    Returns
    -------
    dfret : pandas.DataFrame
        DataFrame containing devolvolved time-series and error estimates
    """
    # observered data
    g = df[column_name]
    
    chunk_overlap = len(one_sided_prf)
    # check that the chunk length is large enough for the psf
    assert(max_chunklen > 2.1*chunk_overlap)
    
    # divide inputs into chunks for the real worker function
    i0 = 0
    g_chunks = []
    while i0+chunk_overlap < len(g):
        chunk = g.iloc[i0:i0+max_chunklen]
        # sometimes the chunk sizes don't work out
        if len(chunk) > 2*chunk_overlap:
            g_chunks.append(chunk)
        else:
            print("warning: deconv_in_chunks discarding {} data points".format(
                                                len(chunk)-chunk_overlap))
        i0 += max_chunklen - 2*chunk_overlap
    
    #call the worker function
    deconv_func_kwargs.setdefault('column_name', column_name+'_deconv')
    deconv_func_kwargs.setdefault('deconv_func', deconv_func)
    
    #deconv_results = [deconv1d_df(itm.index.values, itm, one_sided_prf, background_count_rate, **deconv_func_kwargs) for itm in g_chunks]
    
    if parallel_backend == 'direct_multiprocessing':
        # parallel version using multiprocessing (needed to run STAN)
        p = multiprocessing.Pool(n_jobs)
        parallel_args_list = (delayed(deconv1d_df)(itm.index.values, 
                                                  itm, one_sided_prf, 
                                                  background_count_rate, 
                                                  same_time = True,
                                                  **deconv_func_kwargs)
                             for itm in g_chunks)
        deconv_results = p.map(deconv1d_df_wrapper, parallel_args_list)
        p.close()
    else:
        # parallel version using joblib
        par = Parallel(n_jobs=n_jobs, verbose=50, backend=parallel_backend)
        
        deconv_results = par(delayed(deconv1d_df)(itm.index.values, 
                                                  itm, one_sided_prf, 
                                                  background_count_rate, 
                                                  same_time = True,
                                                  **deconv_func_kwargs)
                             for itm in g_chunks)
    

    
    #reconstruct results, discarding overlap
    deconv_results = [itm.iloc[chunk_overlap:-chunk_overlap] for itm in deconv_results]
    deconv_results = pd.concat(deconv_results)

    dfret = df.copy()
    for cname in deconv_results.columns:
        dfret[cname] = deconv_results[cname]
    
    return dfret
    

if __name__ == "__main__":
    pass
