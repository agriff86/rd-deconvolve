#!/usr/bin/env python
# coding: utf-8

"""
EMCEE deconvolution using the fast parameterised model of the 750l radon
detector based on W&Z's 1996 paper
"""


from __future__ import (absolute_import, division,
                        print_function)


import glob
import datetime
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('pdf')  # for running command-line only
import matplotlib.pyplot as plt


from scipy.stats import poisson, norm, lognorm

import emcee

from . import util
from . import fast_detector
from . import theoretical_model as tm

lamrn = 2.1001405267111005e-06
lama = 0.0037876895112565318
lamb = 0.00043106167945270227
lamc = 0.00058052527685087548

def is_string(s):
    return isinstance(s, basestring)

def detector_model_wrapper(timestep, initial_state, external_radon_conc,
                           internal_airt_history,
                           parameters, interpolation_mode=1,
                           return_full_state=False):
    """
    TODO:
    """
    t = np.arange(0, timestep*len(external_radon_conc), timestep, dtype=np.float)
    params = fast_detector.parameter_array_from_dict(parameters)
    soln = fast_detector.detector_model(timestep, interpolation_mode,
                                  external_radon_conc, internal_airt_history,
                                  initial_state, params)
    df = pd.DataFrame(index=t/60.0, data=soln)
    df.columns = 'Nrnd,Nrnd2,Nrn,Fa,Fb,Fc,Acc_counts'.split(',')
    eff = parameters['eff']
    df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)
    if return_full_state:
        #TODO - this is supposed to include the initial values
        assert(False)
    return df

def detector_model_observed_counts(timestep, initial_state, external_radon_conc,
                           internal_airt_history,parameters, interpolation_mode=0):
    """just return the observed_counts timeseries"""
    params = fast_detector.parameter_array_from_dict(parameters)
    soln = fast_detector.detector_model(timestep, interpolation_mode,
                                  external_radon_conc, internal_airt_history,
                                  initial_state, params)
    return np.diff(soln[:,-1])




def calc_detector_efficiency(parameters):
    """
    Compute steady-state counting efficiency (counts per Bq/m3 of radon)
    """
    Y0 = fast_detector.calc_steady_state(Nrn = 1.0/lamrn, Q=parameters['Q'],
                                        rs=parameters['rs'],
                                        lamp=parameters['lamp'],
                                        V_tank=parameters['V_tank'],
                                        recoil_prob=parameters['recoil_prob'],
                                        eff=parameters['eff'])
    counts_per_second = Y0[-1]
    steady_state_efficiency = counts_per_second / 1.0
    return steady_state_efficiency


def gen_initial_guess(observed_counts, one_sided_prf, reg='tv'):
    """
    an initial guess based on the RL deconvolution

    use emcee.utils.sample_ball to generate perturbed guesses for each walker
    """
    N = len(observed_counts)
    M = len(one_sided_prf)
    symmetric_prf = np.r_[np.zeros(M-1), one_sided_prf]
    Ndim = len(observed_counts) + M - 1
    # pad first to avoid end effects
    pad0 = np.ones(M)*observed_counts[0]
    pad1 = np.ones(M)*observed_counts[-1]
    observed_counts_padded = np.r_[pad0, observed_counts, pad1]
    initial_guess = util.deconvlucy1d(observed_counts_padded, symmetric_prf,
                                     iterations=1000, reg=reg)
    # exclude padding from return value
    initial_guess = initial_guess[M-1:M+N-1]
    return initial_guess


##### utility functions for fit_parameters_to_obs
def unpack_parameters(p, model_parameters):
    """
    unpack paramters from vector and return dict
    """
    nhyper = model_parameters['nhyper']
    nstate = model_parameters['nstate']
    variable_parameter_names = model_parameters['variable_parameter_names']
    Y0 = p[:nstate]
    parameters = {'Y0':Y0}
    #variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay'
    #nhyper = len(variable_parameter_names)
    variable_parameters = p[nstate:nhyper+nstate]
    radon_concentration_timeseries = p[nhyper+nstate:]
    parameters.update( zip(variable_parameter_names, variable_parameters) )
    return parameters, Y0, variable_parameters, radon_concentration_timeseries

def pack_parameters(Y0, variable_parameters, radon_concentration_timeseries=[]):
    return np.r_[Y0, variable_parameters, radon_concentration_timeseries]


def detector_model_specialised(p, parameters):
    """
    Detector model, specialised for use with emcee
    """
    (varying_parameters, Y0, variable_parameters,
        radon_concentration_timeseries) = unpack_parameters(p, parameters)
    parameters.update(varying_parameters)
    # link recoil probability to screen efficiency
    parameters['recoil_prob'] = 0.5*(1.0-parameters['rs'])
    N = len(radon_concentration_timeseries)
    if N==0:
        if parameters.has_key('transform_radon_timeseries'):
            assert not parameters['transform_radon_timeseries']
        # this means that radon_conc_is_known
        radon_concentration_timeseries = parameters['radon_conc']
        N = len(radon_concentration_timeseries)
    timestep = parameters['tres']
    internal_airt_history = parameters['internal_airt_history']
    if parameters.has_key('transform_radon_timeseries') and \
                                       parameters['transform_radon_timeseries']:
        radon_concentration_timeseries = \
        fast_detector.inverse_transform_radon_concs(radon_concentration_timeseries)
    #print(external_radon_conc[0:4])
    cr = detector_model_observed_counts(timestep, parameters['Y0'],
                                 radon_concentration_timeseries,
                                 internal_airt_history,
                                 parameters,
                                 interpolation_mode=
                                            parameters['interpolation_mode'])
    detector_count_rate = cr

    if parameters.has_key('detector_background_cps'):
       detector_count_rate += parameters['detector_background_cps'] * parameters['tres']

    return detector_count_rate


def poisson_pmf_for_testing(population_mean, obs_count):
    """
    Implementation is only sensible for very small inputs (max 15 or so)

    http://en.wikipedia.org/wiki/Poisson_distribution
    $\!f(k; \lambda)= \Pr(X{=}k)= \frac{\lambda^k e^{-\lambda}}{k!}$

    from scipy.stats import poisson
    obs_count = np.arange(1, 10)
    pmf = [poisson_pmf_for_testing(5, itm) for itm in obs_count]
    plt.plot(obs_count, pmf)

    plt.plot(obs_count, poisson.pmf(obs_count, 5))

    # check scipy version works with mu=100
    obs_count = np.arange(50, 151)
    plt.plot(obs_count, poisson.pmf(obs_count, 100))

    """
    k = obs_count
    lam = population_mean
    pmf = lam**k * np.exp(-lam) / np.math.factorial(k)
    return pmf


def lnlike(p, parameters):
    observed_counts = parameters['observed_counts']
    Nobs = len(observed_counts)
    detector_count_rate = detector_model_specialised(p, parameters)
    if not len(detector_count_rate) == Nobs-1:
        print(len(detector_count_rate), Nobs)
        assert False
    #scale counts so that total number of counts is preserved (?)
    # detector_count_rate
    lp = poisson.logpmf(observed_counts[1:], detector_count_rate)
    lp = lp.sum()
    #f, ax = plt.subplots()
    #ax.plot(observed_counts)
    #ax.plot(detector_count_rate)
    #plt.show()
    return lp


def lnprior_hyperparameters(p, parameters):
    """
    Prior constraints on hyper-parameters
    """
    variable_parameters_mu_prior = parameters['variable_parameters_mu_prior']
    variable_parameters_sigma_prior = parameters['variable_parameters_sigma_prior']
    ub = parameters['variable_parameter_upper_bounds']
    lb = parameters['variable_parameter_lower_bounds']
    (varying_parameters, Y0, variable_parameters,
        radon_concentration_timeseries) = unpack_parameters(p, parameters)
    if not np.alltrue(variable_parameters <= ub):
        exidx = variable_parameters > ub
        print(np.array(parameters['variable_parameter_names'])[exidx],
              ub[exidx],
              variable_parameters[exidx])
        #print('parameter upper bound exceeded.')
        lp = -np.inf
    elif not np.alltrue(variable_parameters >= lb):
        #print('parameter lower bound exceeded.')
        lp = -np.inf
    else:
        # assume that all priors are normally-distributed
        lp = norm.logpdf(variable_parameters, variable_parameters_mu_prior,
                                            variable_parameters_sigma_prior).sum()
    return lp

def lnprior_Y0(Y0, parameters):
    """
    Prior on detector state at t=0
    """
    if Y0.min() <= 0.0:
        return -np.inf
    Y0_mu_prior = parameters['Y0_mu_prior']
    # note - for parameter definitions see
    # http://nbviewer.ipython.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/lognormal_distributions.ipynb
    sigma = np.log(2.0)/np.log(Y0_mu_prior) # Standard deviation of log(X) - factor of two
    shape = sigma # Scipy's shape parameter
    scale = Y0_mu_prior # Scipy's scale parameter = np.exp( mean of log(X) )
    ret = lognorm.logpdf(Y0, shape, loc=0, scale=scale)
    ret = ret[:-1].sum()  # the last state variable (Acc_counts) isn't constrained
    return ret

def lnprior_difference(radon_concentration_timeseries, parameters):
    """
    log-normal prior on step-by-step changes in radon concentration
    """
    if parameters.has_key('transform_radon_timeseries') and \
                                       parameters['transform_radon_timeseries']:
        radon_concentration_timeseries = \
        fast_detector.inverse_transform_radon_concs(radon_concentration_timeseries)
    p = radon_concentration_timeseries
    # Parameters must all be > 0
    if p.min() <= 0:
        lp =  -np.inf
        # print('rn conc < 0')
    else:
        dpdt = np.diff(np.log(p))
        if parameters.has_key('ignore_N_steps'):
            n = parameters['ignore_N_steps']
            if n > 0 and n < len(dpdt):
                dpdt = np.sort(dpdt)[n:-n]
            else:
                print('unexpected value for "ignore_N_steps":', n)
                print('it should be an integer, 1 or greater')
        mu = 0.0  # mean expected change - no change
        sigma = np.log(parameters['expected_change_std'])
        #sigma = np.log(2) #standard deviation - factor of two change
        #sigma = np.log(1.5)
        #sigma = np.log(1.05) # much more smoothing
        lp = norm.logpdf(dpdt, mu, sigma).sum()
    return lp

def lnprior_params(p, parameters):
    """
    comine priors
    """
    (varying_parameters, Y0, variable_parameters, radon_concentration_timeseries
       ) = unpack_parameters(p, parameters)
    lp = 0.0
    if len(radon_concentration_timeseries) > 0:
        # radon concentrations are not known (deconvolution)
        lp += lnprior_difference(radon_concentration_timeseries, parameters)

    lp += lnprior_Y0(Y0, parameters)
    lp += lnprior_hyperparameters(p, parameters)
    if parameters.has_key('total_efficiency'):
        # prior on total efficiency
        # 1. put all parameters together in one dictionary
        allparams = dict()
        allparams.update(parameters)
        allparams.update(varying_parameters)
        # 2. compute net efficiency
        mu = parameters['total_efficiency']
        sigma = mu*parameters['total_efficiency_frac_error']
        rs=allparams['rs']
        Y0 = fast_detector.calc_steady_state(1/lamrn,
                                Q=allparams['Q'], rs=rs,
                                lamp=allparams['lamp'],
                                V_tank=allparams['V_tank'],
                                recoil_prob=0.5*(1-rs),
                                eff=allparams['eff'])
        total_efficiency = Y0[-1]
        if total_efficiency <= 0.0: return -np.inf
        lp += norm.logpdf(total_efficiency, mu, sigma)
    return lp


def lnprob(p, parameters):
    # print(len(p), p/p00)

    lp = lnprior_params(p, parameters)
    if np.isfinite(lp):
        lp += lnlike(p, parameters)

    if lp == -np.Inf:
        pass
        #print('lp: minus Inf.')
        #print('lnprior:', lnprior_params(p, parameters))
        #print('lnlike:', lnlike(p, parameters))

    if np.isnan(lp):
        # this should not happen, but let's press on regardless with an
        # error message
        print('NaN during log-probability calculation, set to minus Inf.')
        print('parameters are:')
        print(p)
        lp = -np.inf
    return lp



def fit_parameters_to_obs(t, observed_counts, radon_conc=[],
                          internal_airt_history=[], parameters=dict(),
                          variable_parameter_names = (),
                          variable_parameters_mu_prior = np.array([]),
                          variable_parameters_sigma_prior = np.array([]),
                          walkers_per_dim=2, keep_burn_in_samples=False, thin=2,
                          nthreads=1,
                          iterations=200):
    """
    TODO: doc
    """
    # observed counts need to be integers
    assert np.alltrue(np.round(observed_counts)==observed_counts)
    # make a local copy of the paramters dictionary
    parameters_ = parameters
    # default values for parameters
    parameters = {
                    'transform_radon_timeseries':False
                 }
    parameters.update(parameters_)
    nhyper = len(variable_parameter_names)
    nstate = fast_detector.N_state
    transform_radon_timeseries = parameters['transform_radon_timeseries']

    # temporarily set to zero for MAP estimate (will restore before sampling)
    parameters['transform_radon_timeseries'] = False

    radon_conc_is_known = (len(radon_conc) == len(t))
    parameters['observed_counts'] = observed_counts

    if radon_conc_is_known:
        print("Trying to adjust hyper parameters to match observations")
        parameters['radon_conc'] = radon_conc
    else:
        print("Trying to deconvolve observations")

    # default - constant temperature of 20 degC
    if len(internal_airt_history) == 0:
        print("Internal air temperature not provided. Assuming constant 20degC")
        internal_airt_history = np.zeros(len(t)) + 273.15 + 20.0

    parameters.update( dict(variable_parameter_names=variable_parameter_names,
                            nhyper=nhyper,
                            nstate=nstate,
                            variable_parameters_mu_prior=variable_parameters_mu_prior,
                            variable_parameters_sigma_prior=
                                                variable_parameters_sigma_prior,
                            internal_airt_history=internal_airt_history))
    # Detector state at t=0, prior and initial guess
    Y0 = fast_detector.calc_steady_state(Nrn=1.0, Q=parameters['Q'], rs=parameters['rs'],
                        lamp=parameters['lamp'],
                        V_tank=parameters['V_tank'],
                        recoil_prob=parameters['recoil_prob'],
                        eff=parameters['eff'])
    Nrnd,Nrnd2,Nrn,Fa,Fb,Fc, Acc_counts = Y0
    expected_counts = parameters['eff']*(Fa*lama + Fc*lamc) * (t[1]-t[0])
    scale_factor = observed_counts[0] / expected_counts
    Y0 *= scale_factor
    Y0_mu_prior = Y0.copy()

    parameters.update( dict(Y0_mu_prior=Y0_mu_prior) )

    parameters['tres'] = t[1] - t[0]
    assert(np.allclose(np.diff(t), parameters['tres']))


    rl_radon_timeseries = []
    rltv_radon_timeseries = []

    if not radon_conc_is_known:
        # generate initial guess by (1) working out the PSF, (2) RL deconv.
        # determine PSF for these parameters
        psf_radon_conc = np.zeros(observed_counts.shape)
        psf_radon_conc[1] = 1.0/lamrn
        params_psf = dict()
        params_psf.update(parameters)
        params_psf['t_delay'] += parameters['tres']/2.0 # move to middle-of-interval
        df = detector_model_wrapper(parameters['tres'], Y0*0.0,
                                    psf_radon_conc,
                                    internal_airt_history=internal_airt_history,
                                    parameters=params_psf,
                                    interpolation_mode=parameters['interpolation_mode'])
        #work out when we've seen 99% of the total counts
        nac = df.Acc_counts/df.Acc_counts.iloc[-1]
        idx_90 = (nac > 0.999).nonzero()[0][0]
        if idx_90 % 2 == 0:
            idx_90 += 1 # must be odd
        #TODO: adding that small constant is a hack because RL deconv doesn't work
        #      when there's a zero in the one-sided prf (apparently)
        one_sided_prf = df['count rate'].values[:idx_90] + 0.000048
        one_sided_prf = one_sided_prf / one_sided_prf.sum()
        rl_radon_timeseries = gen_initial_guess(observed_counts, one_sided_prf,
                                                reg=None)
        rltv_radon_timeseries = gen_initial_guess(observed_counts,
                                                  one_sided_prf)

        radon_conc = rltv_radon_timeseries.copy()

        print("RLTV should preserve total counts, this should be close to 1:",
                        radon_conc.sum()/observed_counts.sum())

        # don't accept radon concentration less than 30 mBq/m3 in the guess
        mbq30 = 100 # TODO: this is counts, work out a proper threshold 30e-3/tm.lamrn
        rnavconc = radon_conc.mean()
        radon_conc[radon_conc < mbq30] = mbq30
        radon_conc = radon_conc/radon_conc.mean() * rnavconc

        # if we're simulating a calibration then begin with a guess of
        # constant ambient radon concentration
        if parameters.has_key('cal_source_strength') and \
                                        parameters['cal_source_strength'] > 0:
            # counts per counting interval, gets converted to atoms/m3 later
            radon_conc = radon_conc*0.0 + observed_counts[0]

        f, ax = plt.subplots()
        ax.plot(one_sided_prf)

        f, ax = plt.subplots()
        ax.plot(observed_counts)
        ax.plot(radon_conc)
        plt.show()

        rs = parameters['rs']
        Y0eff = fast_detector.calc_steady_state(1/lamrn,
                                    Q=parameters['Q'], rs=rs,
                                    lamp=parameters['lamp'],
                                    V_tank=parameters['V_tank'],
                                    recoil_prob=0.5*(1-rs),
                                    eff=parameters['eff'])
        total_efficiency = Y0eff[-1]

        if parameters.has_key('total_efficiency'):
            print("prescribed total eff:", parameters['total_efficiency'])
            # detector overall efficiency
            total_efficiency_correction = parameters['total_efficiency']
        else:
            total_efficiency_correction = total_efficiency

        print("computed total eff:", total_efficiency)

        radon_conc = (radon_conc / parameters['tres'] /
                            total_efficiency_correction / lamrn )

        p = pack_parameters(Y0_mu_prior, variable_parameters_mu_prior, radon_conc)
        modcounts = detector_model_specialised(p, parameters)
        print("Model initial guess should preserve total counts.")
        print("this should be close to 1:",
                                modcounts.sum()/observed_counts.sum())
        f, ax = plt.subplots()
        ax.plot(observed_counts, label='obs')
        ax.plot(np.r_[np.nan, modcounts], label='model') # padding needed? TODO: check
        ax.legend()


        plt.show()

        assert len(radon_conc) == len(observed_counts)

    # TEST: use scaled counts as the inial guess
    # radon_conc = (observed_counts / parameters['tres'] /
    #                         parameters['total_efficiency'] / lamrn )

    if radon_conc_is_known:
        p = pack_parameters(Y0_mu_prior, variable_parameters_mu_prior, [])
    else:
        p = pack_parameters(Y0_mu_prior, variable_parameters_mu_prior, radon_conc)

    p00 = p.copy()

    #print(p)
    #print(parameters)
    #print(unpack_parameters(p, parameters)[0])
    # we should now be able to compute the liklihood of the initial location p
    print("Initial guess P0 log-prob:",lnprob(p, parameters))
    assert np.isfinite(lnprob(p, parameters))
    # the function should return -np.inf for negative values in parameters
    # (with the exception of the delay time)
    #for ii in range(len(p)):
    #    pp = p.copy()
    #    pp[ii] *= -1
    #    print(ii, lnprob(pp, parameters))
    #assert(False)
    #print("check:", lnprob(np.r_[Y0_mu_prior, p[5:]], parameters))

    if radon_conc_is_known or not radon_conc_is_known:
        # take the starting location from the MAP

        def minus_lnprob(p,parameters):
            #print(p[nhyper+nstate:nhyper+nstate+4])
            ### p = np.r_[p[0: nstate+nhyper], np.exp(p[nstate+nhyper:])]

            #if not radon_conc_is_known:
            #    # special treatment for the radon concentration timeseries
            #    p_rn = p[nstate+nhyper:]
            #    radon_conc = inverse_transform_radon_concs(p_rn)
            #    p = np.r_[p[0: nstate+nhyper], radon_conc]

            p = inverse_transform_parameters(p, parameters)

            lp = lnprob(p,parameters)
            if False:
                f, axl = plt.subplots(1, 2, figsize=[4,1.5])
                axl[0].plot(parameters['observed_counts'])
                axl[0].plot(detector_model_specialised(p, parameters))
                axl[0].set_title(lp)
                axl[1].plot(radon_conc)
                plt.show()
            if (p.min() < 0) and False:
                print('Parameter less than 0 in minus_lnprob call')
                hparams = p[nstate:nstate+nhyper]
                print(zip(parameters['variable_parameter_names'], hparams))
                print(nstate, nhyper, np.where(p<0))
            if (p.min() < 0) and False:
                f, axl = plt.subplots(1, 2, figsize=[4,1.5])
                axl[0].plot(parameters['observed_counts'])
                axl[0].plot(detector_model_specialised(p, parameters))
                axl[0].set_title(lp)
                axl[1].plot(radon_conc)
                plt.show()
            #print(p[nstate:nstate+nhyper], lp)
            #print(lp)
            if not np.isfinite(lp):
                lp = -1e320
            return -lp
        from scipy.optimize import minimize
        method = 'Powell'
        # method = 'BFGS' # BFGS is not working
        #check that we can call this
        x = transform_parameters(p, parameters)
        print("minus_lnprob:", minus_lnprob(x, parameters))

        # use log radon conc in x0
        ### x0 = np.r_[ p[0: nstate+nhyper], np.log(p[nstate+nhyper:])]

        #if not radon_conc_is_known:
        #    # special treatment for the radon concentration timeseries
        #    radon_conc = p[nstate+nhyper:]
        #    p_rn = transform_radon_concs(radon_conc)
        #    p = np.r_[p[0: nstate+nhyper], p_rn]


        with util.timewith(name=method):
            x0 = transform_parameters(p, parameters)
            print('x0:',x0)
            ret = minimize(minus_lnprob, x0=x0, args=(parameters,), method=method,
                            options=dict(maxiter=100))

            #print("MAP P0 log-prob:", lnprob(ret.x, parameters))
            pmin = inverse_transform_parameters(ret.x, parameters)

            ### pmin = np.r_[pmin[0: nstate+nhyper], np.exp(pmin[nstate+nhyper:])]
            print("MAP P0 log-prob:", lnprob(pmin, parameters))

        print("MAP fitting results:")
        ret.pop('direc') # too much output on screen
        print(ret)

        print('(from MAP) pmin:', pmin)

        y1 = detector_model_specialised(pmin, parameters)
        y0 = observed_counts[1:]
        y_ig = detector_model_specialised(p00, parameters)
        f, ax = plt.subplots()
        ax.plot(y0, label='obs')
        ax.plot(y1, label='model')
        ax.plot(y_ig, label='model_guess_before_MAP')
        ax.legend()
        ax.set_title((pmin/p00)[nstate:nstate+nhyper])
        print(zip(variable_parameter_names, (pmin/p)[nstate:nstate+nhyper]))
        plt.show()

        map_radon_timeseries = []

        ## compare parameters with parameters_
        #for k in parameters_.keys():
        #    print(k, parameters_[k], parameters[k])

        if not radon_conc_is_known:
            f, ax = plt.subplots()
            ax.plot(pmin[nstate+nhyper:]*lamrn, label='deconv')
            ax.plot(p00[nstate+nhyper:]*lamrn, label='rl-tv')
            ax.legend()
            ax.set_ylabel('Bq/m3')

            map_radon_timeseries = pmin[nstate+nhyper:].copy()

        p = pmin.copy()

        plt.show()

    # restore the original value of this parameter
    parameters['transform_radon_timeseries'] = transform_radon_timeseries

    if not radon_conc_is_known and transform_radon_timeseries:
        print('Transforming radon timeseries for emcee sampling')
        orig = p[nstate+nhyper:].copy()
        fast_detector.transform_radon_concs_inplace(p[nstate+nhyper:])
        #f, ax = plt.subplots()
        #ax.plot(orig)
        #f, ax = plt.subplots()
        #ax.plot(p[nstate+nhyper:])
        #plt.show()
        check = fast_detector.inverse_transform_radon_concs(p[nstate+nhyper:])
        if not np.allclose(orig, check):
            print("transformed radon concs do not match inverse")
            print("(orig,inv) pairs follow")
            print( [itm for itm in zip(orig, check)] )
            assert False

    # Number of walkers needs to be at least 2x number of dimensions
    Ndim = len(p)
    Nwalker = Ndim * walkers_per_dim
    Nwalker = max(Nwalker, 60) # don't run with less than 60 walkers
    # number of walkers must be even.
    # increment to the next multiple of 4 (for, maybe, easier load balancing)
    Nwalker += (4 - Nwalker % 4)
    p00 = p.copy()
    p0 = emcee.utils.sample_ball(p, std=p/1000.0, size=Nwalker)

    # check that the lnprob function still works
    print("initial lnprob value:", lnprob(p, parameters))

    from multiprocessing.pool import ThreadPool
    if nthreads > 1:
        pool = ThreadPool(nthreads)
    else:
        pool = None

    # sampler
    sampler = emcee.EnsembleSampler(Nwalker,Ndim,lnprob,
                                    args=(parameters,),
                                    pool=pool,
                                    a=2.0)  #default value of a is 2.0
    # burn-in
    pos,prob,state = sampler.run_mcmc(p0, iterations,
                                    storechain=keep_burn_in_samples, thin=thin)

    # sample
    pos,prob,state = sampler.run_mcmc(pos, iterations, thin=thin)

    print('EnsembleSampler mean acceptance fraction during sampling:',
            sampler.acceptance_fraction.mean())

    assert sampler.acceptance_fraction.mean() > 0.05
    # restore the radon concentrations in sampler.chain to their true values
    # (instead of the sequence of coefficients)
    Nch, Nit, Np = sampler.chain.shape
    if transform_radon_timeseries:
        for ii in range(Nch):
            for jj in range(Nit):
                fast_detector.inverse_transform_radon_concs_inplace(
                                        sampler.chain[ii, jj, nstate+nhyper:])

    A = sampler.flatchain

    # put the initial guess (MAP estimate) into the chain
    A = np.vstack([pmin, A])

    mean_est = A.mean(axis=0)
    low = np.percentile(A, 10.0, axis=0)
    high = np.percentile(A, 90.0, axis=0)

    return (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
                        rl_radon_timeseries, rltv_radon_timeseries)


def overlapping_chunk_dataframe_iterator(df, chunksize, overlap=0):
    """
    A generator which produces an iterator over a dataframe with overlapping chunks
    """
    ix0 = 0
    ixstart = ix0+overlap
    ixend = ixstart+chunksize
    ix1 = ixend+overlap
    while ix1 <= len(df):
        yield df.iloc[ix0:ix1]
        ix0+=chunksize
        ixstart+=chunksize
        ixend+=chunksize
        ix1+=chunksize
    return

def chunkwise_apply(df, chunksize, overlap, func, func_args=(), func_kwargs={},
                    nproc=1):
    chunks = overlapping_chunk_dataframe_iterator(df, chunksize, overlap)
    if nproc == 1:
        results = [func(itm, *func_args, **func_kwargs) for itm in chunks]
    else:
        # parallel version
        from joblib import Parallel, delayed
        par = Parallel(n_jobs=nproc, verbose=50)
        results = par(delayed(func)(itm, *func_args, **func_kwargs)
                                 for itm in chunks)
    # add a chunk_id field
    for ii,itm in enumerate(results):
        itm['chunk_id'] = ii
    # strip the overlap
    if overlap>0:
        results = [itm.iloc[overlap:-overlap] for itm in results]
    df_ret = pd.concat(results)
    return df_ret


def emcee_deconvolve_tm(df, col_name='lld',
                    model_parameters={},
                    iterations=500, nthreads=1,
                    nproc=1,
                    keep_burn_in_samples=False, thin=1,
                    walkers_per_dim=3, chunksize=None, overlap=None, short_output=True):
    if chunksize is not None:
        assert overlap is not None
        chunks = overlapping_chunk_dataframe_iterator(df, chunksize, overlap)
        func_kwargs = dict(col_name=col_name,
                                model_parameters=model_parameters,
                                iterations=iterations,
                                nthreads=nthreads,
                                keep_burn_in_samples=keep_burn_in_samples,
                                thin=thin,
                                walkers_per_dim=walkers_per_dim,
                                chunksize=None, overlap=None, short_output=True)
        dfret  = chunkwise_apply(df,
                                             chunksize=chunksize,
                                             overlap=overlap,
                                             func=emcee_deconvolve_tm,
                                             func_kwargs=func_kwargs,
                                             nproc=nproc)

        return dfret
    #
    # default parameters for theoretical model of detector
    #
    rs = 0.76
    parameters = dict(
            Q = 0.0122,
            rs = rs,
            lamp = 1/180.0,
            eff = 0.15,
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.5*(1-rs),
            t_delay = 10.0,
            total_efficiency=0.128,
            total_efficiency_frac_error=0.05,
            background_count_rate=1/60.0)

    # the internal airt history should already have been converted to K
    internal_airt_history = df['airt'].values
    if not internal_airt_history.min() > 200.0:
        print("'airt' needs to be in K at the observation time")
        assert(False)
    if not internal_airt_history.max() < 400:
        print("error in 'airt'")
        assert(False)

    # update with prescribed parameters
    parameters.update(model_parameters)

    # background and total_efficiency might be specified as DataFrame columns
    eff_from_df = False
    if is_string(parameters['total_efficiency']):
        colname_te = parameters['total_efficiency']
        parameters['total_efficiency'] = df[colname_te].mean()
        eff_from_df = True
    bg_from_df = False
    if is_string(parameters['background_count_rate']):
        colname_te = parameters['background_count_rate']
        parameters['background_count_rate'] = df[colname_te].mean()
        bg_from_df = True

    # detector overall efficiency - check it's close to the prescribed efficiency
    # TODO: should eff be adjusted here?
    rs = parameters['rs']
    Y0eff = fast_detector.calc_steady_state(1/lamrn,
                                Q=parameters['Q'], rs=rs,
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=0.5*(1-rs),
                                eff=parameters['eff'])
    total_efficiency = Y0eff[-1]
    print("computed total eff:", total_efficiency, "  prescribed:",
                                                parameters['total_efficiency'])

    if eff_from_df:
        print('Adjusting "eff" parameter so that computed efficiency matches '+
              'prescribed')
        print('  old value of eff:', parameters['eff'])
        parameters['eff'] = parameters['total_efficiency'] / total_efficiency * parameters['eff']
        print('  new value of eff:', parameters['eff'])

    # priors
    variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    variable_parameters_mu_prior = np.array(
                            [parameters[k] for k in variable_parameter_names])
    variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                             parameters['Q']*0.2,
                                             0.05,
                                             1/100.0,
                                             1.,
                                             0.05*parameters['eff']])

    parameters['variable_parameter_lower_bounds'] = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
    parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, 2.0, np.inf, np.inf, np.inf])

    # extract time in seconds
    times = df.index.to_pydatetime()
    tzero = times[0]
    t = np.array([ (itm-tzero).total_seconds() for itm in times])
    tres = t[1] - t[0]

    with util.timewith("emcee deconvolution") as timer:
        #if chunksize is None:
        #    print(df[col_name])
        fit_ret = fit_parameters_to_obs(t, observed_counts=df[col_name].values,
             internal_airt_history = internal_airt_history,
             parameters=parameters,
             variable_parameter_names = variable_parameter_names,
             variable_parameters_mu_prior = variable_parameters_mu_prior,
             variable_parameters_sigma_prior = variable_parameters_sigma_prior,
             iterations=iterations,
             keep_burn_in_samples=keep_burn_in_samples,
             nthreads=nthreads)

    (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
    rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
    popt = A.mean(axis=0)

    #varying parameters
    params_chain = A[:, parameters['nstate']:parameters['nhyper']+parameters['nstate']]
    #radon concentration
    radon_conc_chain = A[:, parameters['nhyper']+parameters['nstate']:]

    # initial state
    b = sampler.chain[:, :, parameters['nhyper']+parameters['nstate']:]

    #varying parameters as DataFrame
    params_chain_df = pd.DataFrame(data = params_chain, columns=parameters['variable_parameter_names'])
    #organise outputs into a DataFrame
    mean_est = radon_conc_chain.mean(axis=0)
    percentiles = np.percentile(radon_conc_chain, [10, 16, 50, 84, 90], axis=0)
    # original counts scaled by net sensitivity
    scfac = 1.0 / tres / parameters['total_efficiency'] / lamrn
    scaled_obs = df[col_name] * scfac
    d = {col_name + '_mean': mean_est,
         col_name + '_map': map_radon_timeseries,
         col_name + '_rl': rl_radon_timeseries * scfac,
         col_name + '_rltv': rltv_radon_timeseries * scfac,
         col_name + '_p10': percentiles[0],
         col_name + '_p16': percentiles[1],
         col_name + '_p50': percentiles[2],
         col_name + '_p84': percentiles[3],
         col_name + '_p90': percentiles[4],
         col_name + '_scaled' : scaled_obs}

    # average-over-sampling-period values (only if interpolation_mode==1)
    if parameters['interpolation_mode'] == 1:
        tmp = radon_conc_chain.copy()
        # N_samples, N_times = tmp.shape
        tmp[:, 1:] = (tmp[:, 1:] + tmp[:, :-1]) / 2.0
        #tmp[0,:] = np.NaN
        mean_est = tmp.mean(axis=0)
        percentiles = np.percentile(tmp, [10, 16, 50, 84, 90], axis=0)
        d[col_name + 'av_mean'] = mean_est
        d[col_name + 'av_p10'] = percentiles[0]
        d[col_name + 'av_p16'] = percentiles[1]
        d[col_name + 'av_p50'] = percentiles[2]
        d[col_name + 'av_p84'] = percentiles[3]
        d[col_name + 'av_p90'] = percentiles[4]

    # a bunch of samples from the distribution
    N_samples = 1000 # TODO: make an argument
    Ns,Nt = radon_conc_chain.shape
    if N_samples>Ns:
        N_samples = Ns
    if N_samples > 0:
        if parameters['interpolation_mode'] == 1:
            instances = tmp
        else:
            instances = radon_conc_chain
        take_idx = np.floor(np.linspace(0,Ns-1, N_samples)).astype(np.int)
        sample_cols = []
        for ii in range(N_samples):
            k = col_name+'sample_'+str(ii)
            sample_cols.append(k)
            d[k] = instances[take_idx[ii]]

    dfret = pd.DataFrame(data=d, index=df.index)

    diagnostics = dict(raw_chain=sampler.chain,
                       parameters=parameters,
                       params_chain_df = params_chain_df,
                       radon_conc_chain=radon_conc_chain)

    f, ax = plt.subplots()
    plot_cols = [itm for itm in dfret.columns if col_name+'_' in itm]
    dfret[plot_cols].plot(ax=ax)
    if short_output:
        ret = dfret
    else:
        ret = dfret, diagnostics
    return ret


def get_10min_data():
    import util
    fnames = ['data-controlled-test-2/T1Mar15e.CSV',
              'data-controlled-test-2/T1Apr15e.CSV']
    dfobs = [util.load_radon(itm) for itm in fnames]
    dfobs = pd.concat(dfobs)
    dfobs = dfobs.sort().drop_duplicates(subset=['doy','time'])
    # resample to 10 minute counts
    dfobsrs = dfobs.resample('10min', label='right', closed='right',
                             how='sum')[['lld']]
    num_obs_per_interval = dfobs.resample('10min', label='right',
                                          closed='right', how='count')
    dfobsrs = (dfobsrs.ix[num_obs_per_interval.lld ==
                            num_obs_per_interval.lld.max()])
    # airt is taken at the time of measurement (not averages) and converted to K
    dfobsrs['airt'] = dfobs.ix[dfobsrs.index].airt + 273.15
    # check resample:
    # dfobs.head(11).lld.sum()-dfobs.head(1).lld.values == dfobsrs.lld[0]
    return dfobsrs

def test_df_deconvolve(nproc):
    dfobsrs = get_10min_data()
    t0 = datetime.datetime(2015,4,3,9)
    dt = datetime.timedelta(days=5)
    df = dfobsrs.ix[t0:t0+dt]

    parameters = dict(
            Q = 0.0122,
            rs = 0.8,
            lamp = 1/180.0,
            eff = 0.15*0.92753,
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.02,
            t_delay = 30.0,
            interpolation_mode = 0,
            transform_radon_timeseries = True)

    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])

    # place a constraint on the net efficiency
    parameters['total_efficiency'] = 0.128 # from 'find_parameters_by_optimisation.py'
    parameters['total_efficiency_frac_error'] = 0.05

    # detector overall efficiency - check it's close to the prescribed efficiency
    rs = parameters['rs']
    Y0eff = fast_detector.calc_steady_state(1/lamrn,
                                Q=parameters['Q'], rs=rs,
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=0.5*(1-rs),
                                eff=parameters['eff'])
    total_efficiency = Y0eff[-1]
    print("computed total eff:", total_efficiency, "  prescribed:", parameters['total_efficiency'])

    # adjust eff so that total_efficiency matches prescirbed
    adj = total_efficiency/parameters['total_efficiency']
    parameters['eff'] = parameters['eff'] / adj

    # constraint on how smooth the deconvolved time series should be
    # no constraint is bad for acceptance fraction
    parameters['expected_change_std'] = 1000. #1.1 smooth, 100.0 unconstrained
    parameters['expected_change_std'] = 3. #1.1 smooth, 100.0 unconstrained

    df = emcee_deconvolve_tm(df,
                             iterations=3000,
                             thin=100,
                             chunksize=12*6,# was 24*6
                             overlap=2*6,   # I was experimenting with 9*6
                                            # to see the next peak
                             model_parameters=parameters,
                             nproc=nproc,
                             nthreads=4)


    return df


def test_df_deconvolve_goulburn(nproc, one_night_only=False):
    df, prf30, prf30_symmetric = util.get_goulburn_data(missing_value=np.NaN)
    # drop problematic first value (lld=1)
    df.lld.iloc[0] = np.NaN
    df = df.dropna(subset=['lld'])

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = df.airt.values.copy()
    df.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    df.airt += 273.15

    df.airt.plot()
    plt.show()

    # drop the calibration period
    df = df.ix[datetime.datetime(2011, 11, 2, 18):]


    parameters = dict(
            Q = 0.0122,
            rs = 0.9,
            lamp = 1/180.0,
            eff = 0.14539,
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.02,
            t_delay = 60.0,
            interpolation_mode = 1,
            expected_change_std = 1.25,
            transform_radon_timeseries = True)

    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
    parameters['t_delay'] = 30.0

    # place a constraint on the net efficiency
    parameters['total_efficiency'] = 0.154 # from Scott's cal
    parameters['total_efficiency_frac_error'] = 0.05

    parameters['expected_change_std'] = 1.05 # for TESTING
    parameters['expected_change_std'] = 1.25

    dfobs = df.copy()

    if one_night_only:
        df = emcee_deconvolve_tm(df.head(48),
                                 iterations=3000,
                                 thin=100,
                                 chunksize=None,
                                 overlap=None,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=4)

    else:
        df = emcee_deconvolve_tm(df,
                                 iterations=3000, #3000,  # try e.g. 3000
                                 thin=100, #100,         # also big, e.g. 100
                                 chunksize=43, # was 87
                                 overlap=12,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=4)

    df = df.join(dfobs)

    return df


def test_df_deconvolve_cabauw(nproc, one_night_only=False):
    df = util.load_cabauw(20)
    df = df.dropna(subset=['lld'])

    t0 = datetime.datetime(2012,6,1,12)
    t1 = datetime.datetime(2013,2,1,12)
    df = df.ix[t0:t1]

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = df.airt.values.copy()
    df.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    df.airt += 273.15

    df.airt.plot()
    plt.show()

    expected_net_eff = 0.30 # from eyeballing the output of calibrations

    parameters = {}
    parameters.update(util.standard_parameters)
    parameters['Q_external'] = df.exflow[df.exflow>100].mean() / 1000.0 / 60.0
    parameters['V_delay'] = 400.0 / 4000.0
    parameters['V_tank'] = 1.500
    parameters['interpolation_mode'] = 1
    parameters['lamp'] /= 2.0 # bigger tank, lower plateout (but dunno for sure)
    parameters['t_delay'] = 60.0
    parameters['detector_background_cps'] = 50./1800

    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
    parameters['t_delay'] = 30.0

    # place a constraint on the net efficiency
    parameters['total_efficiency'] = expected_net_eff
    parameters['total_efficiency_frac_error'] = 0.05

    parameters['expected_change_std'] = 1.05 # for TESTING
    parameters['expected_change_std'] = 1.25

    # check net sensitivity/efficiency
    ne = tm.calc_detector_efficiency(parameters)
    # set eff so that net eff equals expectation
    parameters['eff'] = parameters['eff'] * expected_net_eff / ne

    dfobs = df.copy()

    if one_night_only:
        df = emcee_deconvolve_tm(df.head(48),
                                 iterations=3000,
                                 thin=100,
                                 chunksize=None,
                                 overlap=None,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=1)

    else:
        df = emcee_deconvolve_tm(df,
                                 iterations=3000, #3000,  # try e.g. 3000
                                 thin=100, #100,         # also big, e.g. 100
                                 chunksize=43, # was 87
                                 overlap=12,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=4)

    df = df.join(dfobs)

    return df


def test_df_deconvolve_richmond(nproc, one_night_only=False):
    df = util.load_richmond()

    # a long run of stable nights/clear days
    t0 = datetime.datetime(2011,6,23,12)
    t1 = datetime.datetime(2011,7,19,22)
    # let's just do 14 days - higher efficiency use of the cluster
    t1 = datetime.datetime(2011,7,7,22)

    df = df.ix[t0:t1]

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = df.airt.values.copy()
    df.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    df.airt += 273.15

    #df.airt.plot()
    #plt.show()

    expected_net_eff = 0.322 #from Dec 2011 cal (MAP estimate)

    parameters = {}
    parameters.update(util.standard_parameters)
    parameters['Q_external'] = df.exflow.mean() / 1000.0 / 60.0
    parameters['V_delay'] = 400.0 / 4000.0
    parameters['V_tank'] = 1.500
    parameters['interpolation_mode'] = 1
    parameters['lamp'] /= 2.0 # bigger tank, lower plateout (but dunno for sure)
    parameters['t_delay'] = 60.0
    parameters['detector_background_cps'] = 500./3600

    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
    parameters['t_delay'] = 30.0

    # place a constraint on the net efficiency
    parameters['total_efficiency'] = expected_net_eff
    parameters['total_efficiency_frac_error'] = 0.05

    parameters['expected_change_std'] = 1.05 # for TESTING
    parameters['expected_change_std'] = 1.25

    # check net sensitivity/efficiency
    ne = tm.calc_detector_efficiency(parameters)
    # set eff so that net eff equals expectation
    parameters['eff'] = parameters['eff'] * expected_net_eff / ne

    dfobs = df.copy()

    if one_night_only:
        df = emcee_deconvolve_tm(df.head(48),
                                 iterations=3000,
                                 thin=100,
                                 chunksize=None,
                                 overlap=None,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=1)

    else:
        df = emcee_deconvolve_tm(df,
                                 iterations=3000, #3000,  # try e.g. 3000
                                 thin=100, #100,         # also big, e.g. 100
                                 chunksize=240, # this is 5 times larger than I'd like!
                                 overlap=120,
                                 model_parameters=parameters,
                                 nproc=nproc,
                                 nthreads=4)

    df = df.join(dfobs)

    return df

def test_deconvolve(iterations=100):
    import util
    dfobsrs = get_10min_data()

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = dfobsrs.airt.values.copy()
    dfobsrs.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    dfobsrs.airt += 273.15

    dfobsrs.ix[datetime.datetime(2015,3,27):].plot()
    # low background - begin on 4 April
    # higher background - begin on 6th April

    dt = datetime.timedelta(days=1)
    t0 = datetime.datetime(2015, 4, 4)
    dfss = dfobsrs.ix[t0:t0+dt]

    ## just the background fluctuations
    #dfss = dfss.head(40)

    ## tight focus on the peak
    dfss = dfss.ix[t0+datetime.timedelta(hours=12): t0+datetime.timedelta(hours=18):]

    parameters = dict(
            Q = 0.0122,
            rs = 0.8,
            lamp = 1/180.0,
            eff = 0.15*0.92753,
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.02,
            t_delay = 60.0,
            interpolation_mode = 0,
            transform_radon_timeseries = True)

    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
    parameters['t_delay'] = 30.0

    parameters['transform_radon_timeseries'] = True # on background, this changes the acceptance fraction from 0.17 to ... 0.17 (?)

    # place a constraint on the net efficiency
    parameters['total_efficiency'] = 0.128 # from 'find_parameters_by_optimisation.py'
    parameters['total_efficiency_frac_error'] = 0.05

    # detector overall efficiency - check it's close to the prescribed efficiency
    rs = parameters['rs']
    Y0eff = fast_detector.calc_steady_state(1/lamrn,
                                Q=parameters['Q'], rs=rs,
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=0.5*(1-rs),
                                eff=parameters['eff'])
    total_efficiency = Y0eff[-1]
    print("computed total eff:", total_efficiency, "  prescribed:", parameters['total_efficiency'])

    # adjust eff so that total_efficiency matches prescirbed
    adj = total_efficiency/parameters['total_efficiency']
    parameters['eff'] = parameters['eff'] / adj

    # constraint on how smooth the deconvolved time series should be
    parameters['expected_change_std'] = 1000. #1.1 smooth, 100.0 unconstrained
    parameters['expected_change_std'] = 2. #1.1 smooth, 100.0 unconstrained
    parameters['expected_change_std'] = 3.
    #parameters['expected_change_std'] = 1.25 #1.1 smooth, 100.0 unconstrained

    ## ignore the step change in calibration for the smoothing
    #parameters['ignore_N_steps']=1

    # priors
    variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    variable_parameters_mu_prior = np.array(
                            [parameters[k] for k in variable_parameter_names])
    variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                             parameters['Q']*0.2,
                                             0.05,
                                             1/100.0,
                                             1.,
                                             0.05*parameters['eff']])

    # extract time in seconds
    times = dfss.index.to_pydatetime()
    tzero = times[0]
    t = np.array([ (itm-tzero).total_seconds() for itm in times])
    tres = t[1] - t[0]

    if iterations > 1000:
        thin = 10
    else:
        thin = 1

    with util.timewith("emcee deconvolution") as timer:
        fit_ret = fit_parameters_to_obs(t, observed_counts=dfss.lld.values,
             parameters=parameters,
             variable_parameter_names = variable_parameter_names,
             variable_parameters_mu_prior = variable_parameters_mu_prior,
             variable_parameters_sigma_prior = variable_parameters_sigma_prior,
             iterations=iterations,
             thin=thin,
             keep_burn_in_samples=False,
             nthreads=4)

    (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
    rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
    popt = A.mean(axis=0)

    params_chain = A[:, parameters['nstate']:parameters['nhyper']+parameters['nstate']]
    radon_conc_chain = A[:, parameters['nhyper']+parameters['nstate']:]

    b = sampler.chain[:, :, parameters['nhyper']+parameters['nstate']:]

    df_params_chain = pd.DataFrame(data = params_chain, columns=parameters['variable_parameter_names'])

    f, ax = plt.subplots()
    n = parameters['nhyper']+parameters['nstate']
    ax.plot(mean_est[n:])
    ax.plot(low[n:])
    ax.plot(high[n:])

    import joblib
    sampler, A, mean_est, low, high, parameters
    data_dump = sampler.chain, A, mean_est, low, high, parameters
    joblib.dump(data_dump, 'data_dump.bin', compress=1)

    # recover data with:
    # dump_data = joblib.load('data_dump.bin')
    # chain, A, mean_est, low, high, parameters = dump_data

    return fit_ret


def test_deconvolve_iterations():
    iterations_list = [10, 100, 1000, 10000]
    avgs = []
    for it in iterations_list:
        fit_ret = test_deconvolve(it)
        (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
                        rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
        avgs.append(((high/mean_est)[20:]).mean())
        sampler, A, mean_est, low, high, parameters, fit_ret = 0,0,0,0,0,0,0
    f, ax = plt.subplots()
    ax.semilogx(iterations_list, avgs)

    return iterations_list, avgs

def test_fit_to_obs():
    import util
    fnames = ['data-controlled-test-2/T1Mar15e.CSV',
              'data-controlled-test-2/T1Apr15e.CSV']

    dfobs = [util.load_radon(itm) for itm in fnames]
    dfobs = pd.concat(dfobs)

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = dfobs.airt.values.copy()
    dfobs.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    dfobs.airt += 273.15

    #dom = 23   # 18--26 are spikes
    #dom = 29
    #for dom in [29]: #range(18,30): #this should be an argument
    t0 = datetime.datetime(2015, 3, 17, 11)
    for didx in range(23):
        t0 += datetime.timedelta(days=1)
        print('processing ', t0.date())
        #t0 = datetime.datetime(2015,3,dom,11)
        dt = datetime.timedelta(hours=12)
        dt = datetime.timedelta(days=3)

        f, ax = plt.subplots()
        dfobs.lld[t0:t0+dt].plot()


        # work out the net counts and mean background
        #t1 = datetime.datetime(2015,3,dom,13)
        #t2 = datetime.datetime(2015,3,dom,20)
        t1 = t0+datetime.timedelta(hours=2)
        t2 = t1+datetime.timedelta(hours=(20-13))

        dfss = dfobs.ix[t1-datetime.timedelta(hours=6):
                        t1+datetime.timedelta(hours=12)].copy()

        is_spike = dfss.lld.max() > 20000
        if is_spike:
            inj_minutes = 1
        else:
            inj_minutes = 60


        nhrs = 20-13
        total_count = dfobs.lld.cumsum()[t2] - dfobs.lld.cumsum()[t1]

        background_count_rate = dfobs.lld[t1-datetime.timedelta(hours=6):t1].mean()
        background_count = background_count_rate * nhrs * 60

        # we know that the data are one-minute averages, and that the injection goes
        # from 1300-1400 (for square wave) or 1300-1301 (for spike)
        injection_count_rate = (total_count - background_count) / inj_minutes

        parameters = dict(
            Q = 800.0 / 60.0 / 1000.0, # from Whittlestone's paper, L/min converted to m3/s
            rs = 0.7, # from Whittlestone's paper (screen retention)
            lamp = np.log(2.0)/120.0, # from Whittlestone's code (1994 tech report)
            eff = 0.33, # Whittlestone's paper
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.02,
            t_delay = 60.0,
            interpolation_mode=0,
            expected_change_std = 2.0
            )

        # tweak parameters for better match to obs
        parameters['Q_external'] *= 1.03
        parameters['rs'] = 0.92
        parameters['recoil_prob'] = 0.04
        parameters['t_delay'] = 60.0
        parameters['eff'] = 0.15

        # extract time in seconds
        times = dfss.index.to_pydatetime()
        tzero = times[0]
        t = np.array([ (itm-tzero).total_seconds() for itm in times])
        tres = t[1] - t[0]

        # work out detector efficiency for given parameters
        total_eff = calc_detector_efficiency(parameters)
        print("predicted detector counts per Bq/m3 radon:", total_eff)

        # provide our estimate of the radon concentration flowing into the tank
        Sa = 3.08e4 / 1000 / 60 # source steady-state radon delivery rate (Bq/s)
        Q_e = parameters['Q_external']
        Nrn_steady_state = Sa/lamrn/Q_e

        # prints: 770.0
        print("predicted radon concentration during 1-h injection (Bq/m3):",
                 Nrn_steady_state*lamrn)

        # for the 1-min injection, total counts are 16.428x
        # (np.mean(crs[3:]) / np.mean(cri), find_parameters_by_optimisation.py)
        # larger than the one hour injection.
        # Also, it's 60x quicker.  Therefore...
        Nrn_1min_inj = Nrn_steady_state * 60 * 16.428
        if inj_minutes == 60:
            Nrn_inj = Nrn_steady_state
        else:
            Nrn_inj = Nrn_1min_inj

        #radon_conc_bq = np.r_[ np.ones(6*60+1)*background_count_rate,
        #                    np.ones(inj_minutes)*injection_count_rate,
        #                    np.ones(11*60+60-inj_minutes)*background_count_rate
        #                     ] / total_eff / tres / lamrn

        Nrn_bg = background_count_rate / total_eff / tres / lamrn
        radon_conc_bq = np.r_[ np.ones(6*60+1)*Nrn_bg,
                            np.ones(inj_minutes)*Nrn_inj,
                            np.ones(11*60+60-inj_minutes)*Nrn_bg
                             ]

        dfss['radon_conc'] = radon_conc_bq



        # run the model
        # to ensure that the initial guess isn't going totally off the mark
        Y0 = fast_detector.calc_steady_state(dfss.radon_conc.values[0],
                                Q=parameters['Q'], rs=parameters['rs'],
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=parameters['recoil_prob'],
                                eff=parameters['eff'])

        lldmod = detector_model_observed_counts(tres,
                                           Y0,
                                           dfss.radon_conc.values,
                                           dfss.airt.values,
                                           parameters,
                                           interpolation_mode=0)

        dfss['lldmod'] = np.r_[np.NaN, lldmod]

        f, ax = plt.subplots()
        dfss[['lld','lldmod']].plot(ax=ax)


        f, ax = plt.subplots()
        tinj = t1
        dfss[['lld','lldmod']].ix[tinj:tinj+datetime.timedelta(minutes=120)].plot(ax=ax)

        if True:

            # priors for the paramters we want to allow to vary
            variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
            variable_parameters_mu_prior = np.array(
                                    [parameters[k] for k in variable_parameter_names])
            variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                                     parameters['Q']*0.02,
                                                     0.05,
                                                     1/100.0,
                                                     5.,
                                                     0.1])

            fit_ret = fit_parameters_to_obs(t, observed_counts=dfss.lld.values,
             radon_conc=dfss.radon_conc.values,
             internal_airt_history=dfss.airt.values,
             parameters=parameters,
             variable_parameter_names = variable_parameter_names,
             variable_parameters_mu_prior = variable_parameters_mu_prior,
             variable_parameters_sigma_prior = variable_parameters_sigma_prior,
             iterations=250,
             keep_burn_in_samples=False)

            (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
                rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
            popt = A.mean(axis=0)

            (varying_parameters, Y0, variable_parameters_array, radon_concentration_timeseries
                    ) = unpack_parameters(popt, parameters)

            opt_parameters = dict()
            opt_parameters.update(parameters)
            opt_parameters.update(varying_parameters)

            lldmodopt = detector_model_observed_counts(tres,
                                           Y0,
                                           dfss.radon_conc.values,
                                           dfss.airt.values,
                                           opt_parameters,
                                           interpolation_mode=0)

            dfss['lldmodopt'] = np.r_[np.NaN, lldmodopt]

            fname = 'day-{:02}-chain.npy'.format(didx)
            np.save(fname, sampler.chain)

            dfss[['lld','lldmod','lldmodopt']].plot()
            plt.show()

        else:
            fit_ret = None

    return dfss, fit_ret



def test_detector_model():
    """check the detector model for internal consistency"""
    parameters = dict(
                Q = 800.0 / 60.0 / 1000.0,
                rs = 0.7,
                lamp = np.log(2.0)/120.0,
                eff = 0.15,
                Q_external = 40.0 / 60.0 / 1000.0,
                V_delay = 200.0 / 1000.0,
                V_tank = 750.0 / 1000.0,
                t_delay = .0)
    parameters['recoil_prob'] = 0.5*(1-parameters['rs'])

    tres = 60*1

    # arbitrary
    expected_counts = 1000

    # detector overall efficiency
    rs = parameters['rs']
    Y0 = fast_detector.calc_steady_state(1/lamrn,
                                Q=parameters['Q'], rs=rs,
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=0.5*(1-rs),
                                eff=parameters['eff'])
    total_efficiency = Y0[-1]

    # make a timeseries which gives us 'expected counts' counts

    radon_conc = np.zeros(24*3600/tres)
    radon_conc[10] = 1/lamrn * expected_counts / tres / total_efficiency

    # run it through the specialised detector model
    ##
    ##
    variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    variable_parameters_mu_prior = np.array(
                            [parameters[k] for k in variable_parameter_names])
    variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                             parameters['Q']*0.02,
                                             0.05,
                                             1/100.0,
                                             5.,
                                             0.1])

    nhyper = len(variable_parameter_names)
    nstate = fast_detector.N_state

    radon_conc_is_known = True
    #parameters['observed_counts'] = observations
    #parameters['radon_conc'] = radon_conc

    parameters.update( dict(variable_parameter_names=variable_parameter_names,
                            nhyper=nhyper,
                            nstate=nstate,
                            variable_parameters_mu_prior=variable_parameters_mu_prior,
                            variable_parameters_sigma_prior=
                                                variable_parameters_sigma_prior))
    ##
    parameters['tres'] = tres
    p = pack_parameters(Y0*0.0, variable_parameters_mu_prior, radon_conc)
    modcounts = detector_model_specialised(p, parameters)

    # sum of modcounts should equal expected_counts
    print("should be close to 1:", modcounts.sum()/expected_counts)

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    #return np.exp(p) / (1 + np.exp(p))
    # use the fact that 2*inv_logit(2*p)-1 == tanh(p)
    return (np.tanh(p/2.0)+1.0)/2.0

def transform_constrained_to_unconstrained(x, a=0, b=1):
    if np.isfinite(a) and np.isfinite(b):
        y = logit( (x-a)/(b-a) )
    elif np.isfinite(a) and np.isinf(b):
        y = np.log(x-a)
    elif np.isinf(a) and np.isinf(b):
        y = x
    else:
        raise NotImplementedError
    return y

def transform_unconstrained_to_constrained(y, a=0, b=1):
    if np.isfinite(a) and np.isfinite(b):
        x = a + (b-a) * inv_logit(y)
    elif np.isfinite(a) and np.isinf(b):
        x = np.exp(y) + a
    elif np.isinf(a) and np.isinf(b):
        x = y
    else:
        raise NotImplementedError
    return x

def transform_radon_concs(radon_conc):
    p = np.zeros(radon_conc.shape)
    rnsum = radon_conc.sum()
    p[0] = np.log(rnsum)
    acc = rnsum
    for ii in range(len(radon_conc) - 1):
        tmp = radon_conc[ii] / acc
        p[ii+1] = transform_constrained_to_unconstrained(tmp)
        acc -= radon_conc[ii]
    resid = radon_conc[-1] - acc
    print(resid)
    return p

def inverse_transform_radon_concs(p):
    radon_conc = np.empty(p.shape)
    acc = np.exp(p[0])
    for ii in range(len(radon_conc) - 1):
        tmp = transform_unconstrained_to_constrained(p[ii+1])
        rn = tmp * acc
        radon_conc[ii] = rn
        acc -= rn
    radon_conc[-1] = acc
    return radon_conc


def transform_parameters(p, parameters):
    nhyper = parameters['nhyper']
    nstate = parameters['nstate']
    lb = parameters['variable_parameter_lower_bounds']
    ub = parameters['variable_parameter_upper_bounds']
    # state variables: bounded by zero, take log
    state = p[:nstate]
    logstate = np.log(state)

    # variable parameters
    hyper = p[nstate:nstate+nhyper].copy()
    for ii in range(len(hyper)):
        a = lb[ii]
        b = ub[ii]
        t = transform_constrained_to_unconstrained(hyper[ii], a, b)
        assert(np.isfinite(t))
        hyper[ii] = t

    if len(p) > nstate+nhyper:
        radon_conc_p = transform_radon_concs(p[nstate+nhyper:])
    else:
        radon_conc_p = np.array([])

    x = np.r_[logstate, hyper, radon_conc_p]

    return x

def inverse_transform_parameters(x, parameters):
    nhyper = parameters['nhyper']
    nstate = parameters['nstate']
    lb = parameters['variable_parameter_lower_bounds']
    ub = parameters['variable_parameter_upper_bounds']
    # state variables: bounded by zero, take log
    logstate = x[:nstate]
    state = np.exp(logstate)
    state[state == np.inf] = 1e308
    # TODO: make bounds etc configurable
    # 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    hyper = x[nstate:nstate+nhyper].copy()
    for ii in range(len(hyper)):
        a = lb[ii]
        b = ub[ii]
        t = transform_unconstrained_to_constrained(hyper[ii], a, b)
        assert(np.isfinite(t))
        hyper[ii] = t
    #
    if len(x) > nstate+nhyper:
        radon_conc = inverse_transform_radon_concs(x[nstate+nhyper:])
    else:
        radon_conc = np.array([])
    p = np.r_[state, hyper, radon_conc]
    return p



def test_lnprob_functions():
    """
    Test out lnprob_XXXX
    """
    parameters = dict()
    parameters.update(util.standard_parameters)
    parameters['transform_radon_timeseries'] = False
    tres = 60.0
    radon_conc = np.zeros(6*60)
    radon_conc[1] = 1/lamrn * 100

    internal_airt_history = np.zeros(radon_conc.shape) + 20 + 273.15
    parameters['internal_airt_history'] = internal_airt_history

    Y0 = fast_detector.calc_steady_state(1/lamrn * 1e-6,
                                Q=parameters['Q'], rs=parameters['rs'],
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=parameters['recoil_prob'],
                                eff=parameters['eff'])

    parameters['Y0_mu_prior'] = Y0
    parameters['tres'] = tres


    lldmod = detector_model_observed_counts(tres,
                               Y0,
                               radon_conc,
                               internal_airt_history,
                               parameters,
                               interpolation_mode=1)

    lldmod += 1.0/60*tres

    observations = poisson.rvs(lldmod)
    # make an extra one at the start
    observations = np.r_[observations[0], observations]

    f, ax = plt.subplots()
    ax.plot(lldmod)
    ax.plot(observations)


    ##
    variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    variable_parameter_lower_bounds = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
    variable_parameter_upper_bounds = np.array([np.inf, np.inf, 1.0, np.inf, np.inf, np.inf])
    variable_parameters_mu_prior = np.array(
                            [parameters[k] for k in variable_parameter_names])
    variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                             parameters['Q']*0.02,
                                             0.05,
                                             1/100.0,
                                             5.,
                                             0.1])

    nhyper = len(variable_parameter_names)
    nstate = fast_detector.N_state

    radon_conc_is_known = True
    parameters['observed_counts'] = observations
    parameters['radon_conc'] = radon_conc

    parameters.update( dict(variable_parameter_names=variable_parameter_names,
                            nhyper=nhyper,
                            nstate=nstate,
                            variable_parameter_lower_bounds=variable_parameter_lower_bounds,
                            variable_parameter_upper_bounds=variable_parameter_upper_bounds,
                            variable_parameters_mu_prior=variable_parameters_mu_prior,
                            variable_parameters_sigma_prior=
                                                variable_parameters_sigma_prior))
    ##

    p = pack_parameters(Y0, variable_parameters_mu_prior, [])

    lnprob(p, parameters)

    # try and minimise the function by varying parameters
    def minus_lnprob(p,parameters):
        p = inverse_transform_parameters(p, parameters)
        lp = lnprob(p,parameters)
        #print(p)
        #print(lp)
        #print(p)
        if not np.isfinite(lp):
            #print('infinite lp, p is:', p)
            lp = -1e30
        return - lp
    #check we can call this function
    x0 = transform_parameters(p, parameters)
    print('minus lnprob:', minus_lnprob(x0, parameters))
    from scipy.optimize import minimize
    methods = 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'COBYLA'
    for method in methods:
        with util.timewith(name=method):
            opts = dict(maxiter=1000)
            if method == 'BFGS':
                opts['eps'] = 1/100.0
            ret = minimize(minus_lnprob, x0=p, args=(parameters,), method=method,
                            options=opts)
            print('\n\n{} ==============================='.format(method))
            print(ret)
            pmin = inverse_transform_parameters(ret.x, parameters)
            print("P_opt log-prob:", lnprob(pmin, parameters))


def test_lamp_vs_rs():
    """
    Plot lamp values vs rs required for constant net efficiency
    """
    parameters = dict(
                Q = 800.0 / 60.0 / 1000.0,
                rs = 0.7,
                lamp = np.log(2.0)/120.0,
                eff = 0.33,
                Q_external = 40.0 / 60.0 / 1000.0,
                V_delay = 200.0 / 1000.0,
                V_tank = 750.0 / 1000.0,
                recoil_prob = 0.02,
                t_delay = 1.0)

    param0 = dict()
    param0.update(parameters)

    calc_detector_efficiency(parameters)

    def net_eff(rs, lamp):
        parameters.update( dict(rs=rs, lamp=lamp, recoil_prob=0.5*(1-rs)) )
        return calc_detector_efficiency(parameters)

    vnet_eff = np.vectorize(net_eff)


    g_rs = np.linspace(0.5, 1.0, 250)
    g_lamp = np.linspace(1/(6*60), 1/10.0 , 250)

    X,Y = np.meshgrid(g_rs, g_lamp)

    Z = vnet_eff(X,Y)

    f, ax = plt.subplots()
    levels = np.array([0.1, 0.2, 0.3, 0.4])
    ctr = ax.contour(X,Y,Z, levels, colors='k')
    ax.clabel(ctr)
    ax.set_xlabel('Screen capture efficiency')
    ax.set_ylabel('Plateout time constant')

    # same as above, but using platout time (==1/plateout time constant)
    def net_eff(rs, t_p):
        parameters.update( dict(rs=rs, lamp=1/t_p, recoil_prob=0.5*(1-rs)) )
        return calc_detector_efficiency(parameters)

    vnet_eff = np.vectorize(net_eff)


    g_rs = np.linspace(0.5, 1.0, 250)
    g_tp = np.linspace(10.0, 6*60, 250)

    X,Y = np.meshgrid(g_rs, g_tp)

    Z = vnet_eff(X,Y)

    f, ax = plt.subplots()
    levels = np.array([0.1, 0.2, 0.3, 0.4])
    ctr = ax.contour(X,Y,Z, levels, colors='k')
    ax.clabel(ctr)
    ax.set_xlabel('Screen capture efficiency')
    ax.set_ylabel('Plateout time')

    # look at how the internal flow rate affects net efficiency

    parameters.update(param0)
    Q0 = parameters['Q']
    f, ax = plt.subplots()
    for T_p in [6, 60, 600, np.Inf]:
        parameters.update(lamp=1./T_p)
        def net_eff(Q):
            parameters.update( dict(Q=Q) )
            return calc_detector_efficiency(parameters)
        vnet_eff = np.vectorize(net_eff)

        g_Q = np.linspace(1e-10, Q0*2, 500)
        ne = vnet_eff(g_Q)
        ax.plot(g_Q*1000*60, ne, label='plateout time = {} sec'.format(T_p))
        ax.set_xlabel('Internal loop flow rate (Q)')
        ax.set_ylabel('Net Efficiency (cps / Bqm3)')
    ax.legend(loc='best')




if __name__ == "__main__":

    #test_lamp_vs_rs()
    #test_lnprob_functions()

    #fit_ret = test_deconvolve()
    #import os
    #print(os.getcwd())

    #fit_ret = test_deconvolve_iterations()

    #df = test_df_deconvolve_goulburn(nproc=1, one_night_only=True)

    #df = test_df_deconvolve(nproc=2)  # run this on the cluster with nproc=8
    #df.to_csv('data-processed/tm_deconvolution_lab_test_10min.csv')
    #df.to_pickle('data-processed/tm_deconvolution_lab_test_10min.pkl')

    #df = test_df_deconvolve_goulburn(nproc=1)  # run this on the cluster with nproc=8
    #df.to_csv('data-processed/tm_deconvolution_glb.csv')
    #df.to_pickle('data-processed/tm_deconvolution_glb.pkl')


    #df = test_df_deconvolve_cabauw(nproc=1)  # run this on the cluster with nproc=8
    #df.to_csv('data-processed/tm_deconvolution_cabauw20m_10min.csv')
    #df.to_pickle('data-processed/tm_deconvolution_cabauw20m_10min.pkl')

    df = test_df_deconvolve_richmond(nproc=16)  # run this on the cluster with nproc=16 (nthread=4)
    df.to_csv('data-processed/tm_deconvolution_richmond_6min.csv')
    df.to_pickle('data-processed/tm_deconvolution_richmond_6min.pkl')

    if False:
        dfss, fit_ret = test_fit_to_obs()
        plt.show()

        (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
            rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
        Ns = parameters['nstate']
        Nv = parameters['nhyper']
        for ii in range(Nv):
            f,ax = plt.subplots()
            blah = ax.plot(sampler.chain[:,:,ii+Ns].T)
            ax.set_title(parameters['variable_parameter_names'][ii])

    plt.show()
