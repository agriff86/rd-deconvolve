#!/usr/bin/env python
# coding: utf-8

"""
Deconvolve radon observations from a 1500L radon detector


Input data are assumed to be in a subdirectory called `raw-data`.
The script attempts to load all CSV files in `raw-data` and then 
saves a single csv file with the corrected timeseries. 
"""

import sys
import os
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('PDF') # prevent graphics from being displayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import glob

try:
    import rddeconv
except ImportError:
    # assume we're running from within source tree but don't want to install
    sys.path.append(PROJECT_DIR)
    import rddeconv


from rddeconv.emcee_deconvolve_tm import emcee_deconvolve_tm, lamrn

def _load_single_file(fname):
    """Load a single file in the usual csv format 

    Parameters
    ----------
    fname : string
        name of file to read
    """
    df = pd.read_csv(fname)
    df.columns = [itm.strip().lower() for itm in df.columns]
    df['time'] = [datetime.datetime.strptime(itm, '%H:%M').time() for itm in df.time]
    time = [ datetime.datetime.combine(datetime.date(int(itm[1]['year']),
                                                         int(itm[1]['month']),
                                                         int(itm[1]['dom'])),
                                           itm[1]['time']) for itm in df.iterrows()]
    df.index = time
    #clean up negative values (negative values for lld probably indicate missing data)
    df.loc[df.lld<0, 'lld'] = np.NaN
    return df


def load_data(fname_glob):
    """Load raw radon data files

    Parameters
    ----------
    fname_glob : str
        File name or file name glob (e.g. "raw-data/*.CSV")
    """
    fnames = sorted(glob.glob(fname_glob))
    df = pd.concat([_load_single_file(fn) for fn in fnames]).sort_index()
    return df
        


def main_df_deconvolve(nproc, one_night_only=False):
    """Run the deconvolution method

    1. load/munge data
    2. set instrument parameters and priors
    3. run deconvolution


    Parameters
    ----------
    nproc : int
        Number of processes to use
    one_night_only : bool, optional
        If true, then only run one night worth of data (for testing), by default False

    Returns
    -------
    pandas.DataFrame
        Deconvolved data
    """

    #
    # ... load/munge data
    #
    df = load_glb(fname_glb=os.path.join(EXAMPLE_DIR,'raw-data/Goulburn_Nov_2011_Internal_DB_v01_raw.csv'))
    # drop problematic first value (lld=1)
    df.lld.iloc[0] = np.NaN
    df = df.dropna(subset=['lld'])

    # we want the air temperature to be *at* the report time, rather than an
    # average over each half hour
    atv = df.airt.values.copy()
    df.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
    # and convert to K
    df.airt += 273.15


    # drop the calibration period
    df = df.loc[datetime.datetime(2011, 11, 2, 18):]

    # drop the bad data at the end of the record
    df = df.loc[:datetime.datetime(2011, 11, 10, 12)]

    #
    # ... set instrument parameters
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

    # note: using default priors, defined in emcee_deconvolve_tm

    if one_night_only:
        df = df.head(48 * nproc)
    
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
                             nthreads=1,
                             stop_on_error=True)

    df = df.join(dfobs)

    return df

if __name__ == "__main__":

    #
    # ... check on emcee version
    #
    import emcee
    print("EMCEE sampler, version: {}".format(emcee.__version__))

    os.chdir(EXAMPLE_DIR)
    df = test_df_deconvolve_goulburn(nproc=20, one_night_only=True)
    df.to_csv('tm_deconvolution_glb.csv')

    # save a picture comparing raw (lld_scaled) with deconvolved (lld_mean) obs
    fig, ax = plt.subplots()
    # plot, with conversion to Bq/m3 from atoms/m3
    (df[['lld_scaled','lld_mean']]*lamrn).plot(ax=ax)
    ax.set_ylabel('Radon concentration inside detector (Bq/m3)')
    fig.savefig('tm_deconvolution_glb.png')
