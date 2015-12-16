#!/usr/bin/env python
# coding: utf-8

"""
Baysean deconvolution

Save data, don't produce plots.  A test for running on the cluster.

"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import datetime
import pylab
import util
import util_stan
import sys
import util_emcee

# parse command line - if nothing specified use 'all' as default
if len(sys.argv) < 2:
   job_opt = 'all'
else:
   job_opt = sys.argv[1]

#
# .... load observations
#
t0 = datetime.datetime(2014,12,5,13,6)
t_injection = [t0 + datetime.timedelta(days=ii) for ii in range(5)]
# time range for deconvolution test
t_deconv = datetime.datetime(2014, 12, 10, 0)
# load data
data_dir = './data-controlled-test/'
fname = data_dir + '700L_test2.xlsx'
sheet_name = 'F1Dec14'
df700 = util.load_radon(fname, sheet_name)
#choose the data we wish to deconvolve
df = df700[t_deconv:]
#df = df[['lld']]

background_count_rate = 10  #TODO: make this realistic

##choose a subset - faster computation for testing
#df = df.head(250)  #500 gives two peaks

## 7am to 7am am, peak npeak (there are 5 total)
#npeak = 1
#tidx0 = t_deconv+datetime.timedelta(days=npeak)+datetime.timedelta(hours=7)
#tidx1 = tidx0 + datetime.timedelta(days=1)
#df = df[tidx0:tidx1]

# load point response function (prf)
one_sided_prf = pd.read_csv('one_sided_prf.csv', index_col=0).prf.values
symmetric_prf = pd.read_csv('symmetric_prf.csv', index_col=0).prf.values

# make the prf smaller for less computation
one_sided_prf = one_sided_prf[:61]  # drops to 10-4 at 60, 10-3 at 30
one_sided_prf /= one_sided_prf.sum()


observed_data = df.lld.values
N = len(observed_data)


# model versions: lognormal_difference, lognormal_from_data, lognormal, uniform

if True:
    if job_opt in ['1', 'all']:
        # stan methods
        for model_version in "standard log_difference".split():
            deconv_func_kwargs = dict(model_version = model_version,
                                      column_name = 'lld_'+model_version+'_deconv',
                                      chains=8, n_jobs=8)
            df = util_emcee.deconv_in_chunks(df, 'lld', one_sided_prf, 20, 
                                                    max_chunklen=205, 
                                                    deconv_func=util_stan.stan_deconvolve, 
                                                    n_jobs=1,
                                                    deconv_func_kwargs=deconv_func_kwargs,
                                                    parallel_backend='threading')
            df.to_csv('data-processed/lab_test_deconv.csv')
            df.to_pickle('data-processed/lab_test_deconv.pkl')
    if job_opt in ['2', 'all']:    
        #emcee methods
        for model_version in "lognormal_difference lognormal_from_data lognormal uniform multiscale".split():
            deconv_func_kwargs = dict(model_version = model_version, 
                                      column_name = 'lld_'+model_version+'_deconv')
            df = util_emcee.deconv_in_chunks(df, 'lld', one_sided_prf, 10, max_chunklen=205,
                                                n_jobs=8,
                                                deconv_func_kwargs=deconv_func_kwargs)
            df.to_csv('data-processed/lab_test_deconv.csv')
            df.to_pickle('data-processed/lab_test_deconv.pkl')

if True:

    # deconvolution on real data
    prf30, prf30_symmetric = util.prf30_from_prf6(one_sided_prf)

    def load_glb():
        """load Goulburn radon data"""
        fname_glb = 'data/Goulburn_Nov_2011_Internal_DB_v01_raw.csv'
        df_glb = pd.read_csv(fname_glb)
        df_glb.columns = [itm.strip().lower() for itm in df_glb.columns]
        df_glb['time'] = [datetime.datetime.strptime(itm, '%H:%M').time() for itm in df_glb.time]
        time = [ datetime.datetime.combine(datetime.date(int(itm[1]['year']),
                                                             int(itm[1]['month']),
                                                             int(itm[1]['dom'])),
                                               itm[1]['time']) for itm in df_glb.iterrows()]
        df_glb.index = time
        #clean up negative values
        df_glb.lld[df_glb.lld<0] = 500
        return df_glb

    df_glb = load_glb()
    if job_opt in ['3', 'all']:
        # stan methods
        for model_version in "standard log_difference".split():
            deconv_func_kwargs = dict(model_version = model_version,
                                      column_name = 'lld_'+model_version+'_deconv',
                                      n_jobs=8, chains=8)
            df_glb = util_emcee.deconv_in_chunks(df_glb, 'lld', prf30, 30, 
                                                max_chunklen=105, 
                                                deconv_func=util_stan.stan_deconvolve, 
                                                n_jobs=1,  
                                                deconv_func_kwargs=deconv_func_kwargs,
                                                parallel_backend='threading')
            df_glb.to_csv('data-processed/goulburn_deconv.csv')
            df_glb.to_pickle('data-processed/goulburn_deconv.pkl')
    if job_opt in ['4', 'all']:
        # emcee methods
        for model_version in "lognormal_difference lognormal_from_data lognormal uniform multiscale".split():
            deconv_func_kwargs = dict(model_version = model_version, 
                                      iterations=2000,
                                      column_name = 'lld_'+model_version+'_deconv')
            
            df_glb = util_emcee.deconv_in_chunks(df_glb, 'lld', prf30, 30, max_chunklen=105,
                                                n_jobs=8,
                                                deconv_func_kwargs=deconv_func_kwargs)
            df_glb.to_csv('data-processed/goulburn_deconv.csv')
            df_glb.to_pickle('data-processed/goulburn_deconv.pkl')



if __name__ == "__main__":
    pass
