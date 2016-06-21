#!/usr/bin/env python
# coding: utf-8

"""
Deconvolve radon observations from simulated observations (700L detector)
"""

from __future__ import (absolute_import, division,
                        print_function)

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from scipy import stats
import rddeconv

from rddeconv.emcee_deconvolve_tm import emcee_deconvolve_tm, lamrn
from rddeconv.theoretical_model import detector_model_observed_counts
from rddeconv.fast_detector import N_state, calc_steady_state


#
# ... set instrument parameters here, because they are used to generate
# ... fake observations
#
parameters = dict()
parameters.update(rddeconv.util.standard_parameters)
parameters.update(dict(
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
        transform_radon_timeseries = True))

parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
parameters['t_delay'] = 30.0

# place a constraint on the net efficiency
parameters['total_efficiency'] = 0.154 # from Scott's cal
parameters['total_efficiency_frac_error'] = 0.05

parameters['expected_change_std'] = 1.05 # for TESTING
parameters['expected_change_std'] = 1.25



def linear_chirp(freq0, t, k, phase0=0):
    """
    x(t) = \sin\left[\phi_0 + 2\pi \left(f_0 t + \frac{k}{2} t^2 \right) \right]
    """
    x = np.sin(phase0 + 2 * np.pi * (freq0 * t + k/2.0 * t**2))
    return x

def sawtooth(t, period):
    x = (t%period) * 1/period
    return x

def generate_obs(samplerate=60*10,
                 maxt=24*3600, amplitude=10.0, background=1.0,
                 random_state=None):
    """generate synthetic data"""
    t = np.arange(0, maxt, samplerate)
    rn = ((sawtooth(t+9*3600, 24*3600)-0.5) * amplitude * 2)
    rn[rn<0] = 0.0
    rn += background
    rn = rn/lamrn

    Y0 = calc_steady_state(Nrn=rn[0], Q=parameters['Q'], rs=parameters['rs'],
        lamp=parameters['lamp'], V_tank=parameters['V_tank'],
        recoil_prob=parameters['recoil_prob'], eff=parameters['eff'])

    air_temp = np.zeros(rn.shape) + 273.15 + 20.0
    lld_expected = detector_model_observed_counts(samplerate, Y0, rn, air_temp,
                                         parameters)
    lld_expected = np.r_[lld_expected[0], lld_expected]

    lld = stats.poisson.rvs(lld_expected, random_state=random_state)


    time0 = datetime.datetime(2000,1,1,9+6)
    times = [time0 + datetime.timedelta(seconds=samplerate*ii) for ii in range(len(rn))]

    df = pd.DataFrame(index=times, data=dict(lld=lld, lld_expected=lld_expected,
        true_radon=rn))

    df['airt'] = air_temp

    return df



def test_df_deconvolve_synthetic(nproc, one_night_only=True, daytime_min=1.0, nighttime_max=10.0):
    """
    run the deconvolution method

    1. load/munge data
    2. set instrument parameters and priors
    3. run deconvolution
    """

    #
    # ... load/munge data
    #
    df = generate_obs(background=daytime_min,
        amplitude=nighttime_max-daytime_min)

    # note: using default priors, defined in emcee_deconvolve_tm

    if one_night_only:
        chunksize = None
        overlap = None
    else:
        chunksize = 43
        overlap = 12

    dfobs = df.copy()

    #
    # ... run deconvolution
    #
    df = emcee_deconvolve_tm(df,
                             iterations=3000, #3000,  # try e.g. 3000
                             thin=100, #100,         # also big, e.g. 100
                             chunksize=chunksize,
                             overlap=overlap,
                             model_parameters=parameters,
                             nproc=nproc,
                             nthreads=4)

    df = df.join(dfobs)

    return df

if __name__ == "__main__":
    df = test_df_deconvolve_synthetic(nproc=1, one_night_only=True)
    df.to_csv('tm_deconvolution_synthetic_small_amplitude.csv')
    df.to_pickle('tm_deconvolution_synthetic_small_amplitude.pkl')

    # save a picture comparing raw (lld_scaled) with deconvolved (lld_mean) obs
    fig, ax = plt.subplots()
    # plot, with conversion to Bq/m3 from atoms/m3

    dflam = df*lamrn
    dflam[['lld_scaled','lld_mean', 'true_radon']].plot(ax=ax)
    ax.fill_between(dflam.index, dflam.lld_p10, dflam.lld_p90, facecolor='black', alpha=0.2, zorder=0)
    ax.set_ylabel('Radon concentration inside detector (Bq/m3)')
    fig.savefig('tm_deconvolution_synthetic_small_amplitude.png')


    df = test_df_deconvolve_synthetic(nproc=1, one_night_only=True,
        daytime_min=8, nighttime_max=100.0)
    df.to_csv('tm_deconvolution_synthetic_large_amplitude.csv')
    df.to_pickle('tm_deconvolution_synthetic_large_amplitude.pkl')

    # save a picture comparing raw (lld_scaled) with deconvolved (lld_mean) obs
    fig, ax = plt.subplots()
    # plot, with conversion to Bq/m3 from atoms/m3
    dflam = df*lamrn
    dflam[['lld_scaled','lld_mean', 'true_radon']].plot(ax=ax)
    ax.fill_between(dflam.index, dflam.lld_p10, dflam.lld_p90, facecolor='black', alpha=0.2, zorder=0)
    ax.set_ylabel('Radon concentration inside detector (Bq/m3)')
    fig.savefig('tm_deconvolution_synthetic_large_amplitude.png')