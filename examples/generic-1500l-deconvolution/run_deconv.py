"""
Deconvolution for a 1500L radon detector
"""

#%%
import glob
import numpy as np
import pandas as pd
import datetime
import os
import logzero
import sys
import matplotlib as mpl
mpl.use('agg') # don't try to use X libraries

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import rddeconv
except ImportError:
    # assume we're running from within source tree but don't want to install
    sys.path.append(PROJECT_DIR)
    import rddeconv


import rddeconv
from rddeconv.util import load_standard_csv
from rddeconv.emcee_deconvolve_tm import emcee_deconvolve_tm


# logzero.logfile('deconv.log', mode='w')

import logging
LOG_FILENAME = 'deconv.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG, filemode='w')

ddir0 = './data-raw'
ddir1 = './data-intermediate'
ddir2 = './data-processed'


#%%
df = pd.read_csv(os.path.join(ddir1, 'deconvolution_input_data.csv'), index_col=0, parse_dates=True)

# %%

parameters = {}
parameters.update(rddeconv.standard_parameters_1500L)
parameters = {
    "rs": 0.95,
    "lamp": 1 / 180.0,
    "V_delay": 200.0 / 1000.0,
    "num_delay_volumes": 2,
    "expected_change_std": 1.25,
    "V_tank":1500/1000.,
}


chunksize = 48 * 2
overlap = 12

#%%

nproc = 4

#%%

# decolvolution using pymc3
# Presently, the emcee backend is recommended because it is less resource intensive
# to run.
if False:
    deconv_fnames = rddeconv.deconvolve_dataframe_in_chunks(
            df,
            parameters,
            chunksize=chunksize,
            Noverlap=overlap,
            joblib_tasks=nproc,
            figdir="./figs",
            fname_base=os.path.join(ddir1, 'deconvolution_result',
            )
        )

#%%
if False:
    params_1vol = {}
    params_1vol.update(parameters)
    params_1vol['num_delay_volumes'] = 1
    deconv_fnames = rddeconv.deconvolve_dataframe_in_chunks(
            df,
            parameters,
            chunksize=chunksize,
            Noverlap=overlap,
            joblib_tasks=nproc,
            figdir="./figs_1vol",
            fname_base=os.path.join(ddir1, 'deconvolution_result_1vol',
            )
        )


#%%

# preprocessing required for emcee deconvolution

# we want the air temperature to be *at* the report time, rather than an
# average over each half hour
atv = df.airt.values.copy()
df.airt = np.r_[(atv[1:] + atv[:-1])/2.0, atv[-1]]
# and convert to K
df.airt += 273.15

#parameters['total_efficiency_frac_error'] = 0.05

#parameters['recoil_prob'] = 0.5*(1-parameters['rs'])
#parameters['t_delay'] = 30.0

# interpolation mode: 0 - piecewise constant; 1 - piecewise linear
parameters['interpolation_mode'] = 1
# internally, the sampler can transform the radon timeseries
# to 
parameters['transform_radon_timeseries'] = True

# parameters which are to be taken from data frame
# (instead of assigning a value, assign the name of the dataframe
# column to use.)
parameters['background_count_rate'] = 'background_rate'
parameters["total_efficiency"] = 'total_efficiency'
# parameters["Q"] = 'Q'
# parameters["Q_external"] = 'Q_external'

# emcee model uses two variable-size delay volumes, so handle this
if parameters['num_delay_volumes'] == 2:
    parameters['V_delay_2'] = parameters['V_delay']

# deconvolution using emcee
if True:
    # number of processors to use (change this to suit your system)
    nproc=12
    df = emcee_deconvolve_tm(df,
                                iterations=3000, #3000,  # try e.g. 3000
                                thin=100, #100,         # also big, e.g. 100
                                chunksize=chunksize,
                                overlap=overlap,
                                model_parameters=parameters,
                                nproc=nproc,
                                nthreads=1,
                                stop_on_error=True)

    df.to_csv(os.path.join(ddir2, 'emcee-deconv-results.csv'))
