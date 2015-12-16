#!/usr/bin/env python
# coding: utf-8

"""
Baysean deconvolution using pystan

note: to run this under the anaconda distribution first do

pip install pystan


"""

from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import util
import util_stan
import util_emcee

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
df = df[['lld']]



background_count_rate = 10  #TODO: make this realistic

##choose a subset - faster computation for testing
#df = df.head(250)  #500 gives two peaks

#7am to 7am am, peak npeak (there are 5 total)
npeak = 1
tidx0 = t_deconv+datetime.timedelta(days=npeak)+datetime.timedelta(hours=7)
tidx1 = tidx0 + datetime.timedelta(days=1)
tidx0 -= datetime.timedelta(hours=13) # make the period a little longer
df = df[tidx0:tidx1]

# load point response function (prf)
one_sided_prf = pd.read_csv('one_sided_prf.csv', index_col=0).prf.values
symmetric_prf = pd.read_csv('symmetric_prf.csv', index_col=0).prf.values

# make the prf smaller for less computation
one_sided_prf = one_sided_prf[:61]  # drops to 10-4 at 60, 10-3 at 30
one_sided_prf /= one_sided_prf.sum()


observed_data = df.lld.values
N = len(observed_data)

if False:
    return_vals = util_emcee.emcee_deconvolve(df.index.values, 
                                                df.lld.values,
                                                one_sided_prf,
                                                background_count_rate=background_count_rate,
                                                model_version='lognormal',
                                                nthreads=3,
                                                walkers_per_dim=10,
                                                thin=10)


    fit, A, t_result, mean_est, low, high = return_vals

    t_result = np.arange(len(t_result))
    t_orig = t_result[-len(observed_data):]


    f,ax = plt.subplots()
    ax.fill_between(t_result, low, high, alpha=0.2, color='k', linewidth=0)
    ax.plot(t_result, mean_est)
    ax.plot(t_orig, observed_data)
    ax.set_yscale('log')

    # parameter covariance matrix
    f, ax = plt.subplots(figsize=[10,10])
    cc = np.corrcoef(A, rowvar=False)
    cc[np.identity(cc.shape[0], dtype=np.bool)] = np.NaN
    #im = ax.imshow(cc, interpolation='nearest', clim=[-0.25,0.25], 
    #                    cmap=mpl.cm.RdBu_r); 
    im = ax.imshow(cc, clim=[-0.25,0.25], 
                        cmap=mpl.cm.RdBu_r); 
    f.colorbar(im)
    
    # if the model is multiscale, parameters are not the same as A
    f, ax = plt.subplots(figsize=[10,10])
    cc = np.corrcoef(fit.flatchain, rowvar=False)
    cc[np.identity(cc.shape[0], dtype=np.bool)] = np.NaN
    #im = ax.imshow(cc, interpolation='nearest', clim=[-0.25,0.25], 
    #                    cmap=mpl.cm.RdBu_r); 
    im = ax.imshow(cc, clim=[-0.25,0.25], 
                        cmap=mpl.cm.RdBu_r); 
    f.colorbar(im)

    # autocorrelation of some of the samples
    f, ax = plt.subplots()
    a = fit.chain[10,:,15]
    res = ax.acorr(a-a.mean(), maxlags=None)

    # free up some memory - this doesn't really work
    del fit, A, return_vals
    import gc
    gc.collect()
    for ii in range(1000):
        temp_object = [1,2,3]


# don't include the peaks
df = df.head(184)
# try constant lld
#df['lld'] = 175



# test with emcee deconvolution
deconv_func_kwargs = dict(model_version='fft', iterations=500, thin=5)
dfret2 = util_emcee.deconv_in_chunks(df, 'lld', one_sided_prf, 6, max_chunklen=184, n_jobs=4, deconv_func_kwargs=deconv_func_kwargs)
dfret2.to_pickle('data-processed/lab_test_emcee_deconv.pkl')

# simulated data test
dfsim = util.get_simulated_data()
dfsim = util_emcee.deconv_in_chunks(dfsim, 'lld', one_sided_prf, 6, max_chunklen=184, n_jobs=4, deconv_func_kwargs=deconv_func_kwargs)
dfsim.to_pickle('data-processed/simulated_lab_test_emcee_deconv.pkl')


if False:
    # test with stan deconvolution
    deconv_func_kwargs = dict(model_version='log_difference')
    dfret = util_emcee.deconv_in_chunks(df, 'lld', one_sided_prf, 6, max_chunklen=184, deconv_func=util_stan.stan_deconvolve, n_jobs=1, deconv_func_kwargs=deconv_func_kwargs)

    dfret.to_pickle('data-processed/lab_test_stan_deconv.pkl')

    f, ax = plt.subplots()
    util.plot_with_uncertainty(dfret, 'lld_deconv', ax)
    df.lld.plot(ax=ax)
    ax.set_title('stan')
    
    deconv_func_kwargs = dict(model_version='log_difference')
    dfsim_stan = util_emcee.deconv_in_chunks(dfsim, 'lld', one_sided_prf, 6, max_chunklen=184, deconv_func=util_stan.stan_deconvolve, n_jobs=1, deconv_func_kwargs=deconv_func_kwargs)

    dfsim_stan.to_pickle('data-processed/simulated_lab_test_stan_deconv.pkl')
    
    
    f, ax = plt.subplots()
    util.plot_with_uncertainty(dfsim_stan, 'lld_deconv', ax)
    dfsim_stan.lld.plot(ax=ax)
    dfsim_stan.truth.plot(ax=ax)
    ax.set_title('stan, simulated data')

f, ax = plt.subplots()
util.plot_with_uncertainty(dfret2, 'lld_deconv', ax)
df.lld.plot(ax=ax)
ax.set_title('emcee')

f, ax = plt.subplots()
util.plot_with_uncertainty(dfsim, 'lld_deconv', ax)
dfsim.lld.plot(ax=ax)
dfsim.truth.plot(ax=ax)
ax.set_title('emcee, simulated data')

plt.show()
assert(False)

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

## exclude the calibration
#df_glb = df_glb.iloc[48*2:-12*2]


#rl deconvolution
df_glb['lld_rltv'] = util.deconvlucy1d(df_glb.lld, prf30_symmetric, reg='tv', iterations=1000)
df_glb['lld_rl'] = util.deconvlucy1d(df_glb.lld, prf30_symmetric, reg=None, iterations=1000)
f, ax = plt.subplots()
df_glb[['lld', 'lld_rl', 'lld_rltv']].plot(ax=ax)


# look at detailed diagnostics
if True:
    
    #dftmp = df_glb.head(24*2)
    dftmp = df_glb.iloc[48*2:48*3+1]
    
    observed_data = dftmp.lld.values
    
    # check out the initial guess
    ig = util_emcee.gen_initial_guess(dftmp.lld.values, prf30, Nguess=1, 
                           randscale=0)[0]
    ig2 = util_emcee.gen_initial_guess(dftmp.lld.values, prf30, Nguess=1, 
                           randscale=0, reg='none')[0]
    
    return_vals = util_emcee.emcee_deconvolve(dftmp.index.values, 
                                                dftmp.lld.values,
                                                prf30,
                                                background_count_rate=30,
                                                model_version='lognormal_difference',
                                                iterations=1000,
                                                thin=10,
                                                walkers_per_dim=10,
                                                initial_guess_reg='tv')


    observed_data = dftmp.lld.values
    fit, A, t_result, mean_est, low, high = return_vals
    t_result = np.arange(len(t_result))
    t_orig = t_result[-len(observed_data):]

    
    f,ax = plt.subplots()
    ax.plot(t_result, mean_est)
    ax.fill_between(t_result, low, high, alpha=0.2, color='k', linewidth=0)
    ax.plot(t_orig, observed_data)
    ax.plot(t_result, ig)
    ax.plot(t_result, ig2)
    ax.set_yscale('log')

    # show some chains
    f, ax = plt.subplots()
    idxpeak = np.argmax(mean_est)
    a = fit.chain[:,:,idxpeak]  #[idx_walker, idx_iteration, idx_time
    blah = ax.plot(a.T, alpha=0.05)
    ax.plot(a.mean(axis=0))
    
    f, axl = plt.subplots(3,1)
    for ii in range(3):
        idxpeak = 37+ii
        ax = axl[ii]
        a = fit.chain[:,:,idxpeak]  #[idx_walker, idx_iteration, idx_time
        blah = ax.plot(a.T, alpha=0.05)
        blah = ax.plot(a.mean(axis=0))

    f, ax = plt.subplots()
    ax.plot(fit.acceptance_fraction)
    ax.set_ylabel('acceptance fraction')
    
    f, ax = plt.subplots()
    ax.plot(fit.lnprobability.mean(axis=0))
    ax.set_ylabel('log-probability')
    
    
    # for the 'multiscale' model, fit.chain is different from A
    f, ax = plt.subplots()
    nWalker = fit.chain.shape[0]
    idxpeak = np.argmax(mean_est)
    blah = plt.plot(A[:,idxpeak].reshape((nWalker,-1)).T, alpha=0.05)
    
    # autocorrelation of some of the samples
    f, ax = plt.subplots()
    idxpeak = np.argmax(mean_est)
    a = fit.chain[10,:,idxpeak]  #[idx_walker, idx_iteration, idx_time]
    res = ax.acorr(a-a.mean(), maxlags=None)
    
    f, ax = plt.subplots()
    idxpeak = np.argmax(mean_est)
    a = fit.chain[10,:,15]  #[idx_walker, idx_iteration, idx_time]
    res = ax.acorr(a-a.mean(), maxlags=None)   

plt.show()

# model versions: lognormal_difference, lognormal_from_data, lognormal, uniform

kwargs = dict(model_version = 'lognormal_difference', iterations=500)


dfret = util_emcee.deconv_in_chunks(df_glb, 'lld', prf30, 30, max_chunklen=105, **kwargs)

#dfret = util_emcee.deconv_in_chunks(df_glb, 'lld', prf30, 30, max_chunklen=105, deconv_func=util_stan.stan_deconvolve, n_jobs=1, model_version='standard')


f, ax = plt.subplots()
util.plot_with_uncertainty(dfret, 'lld_deconv', ax)
df_glb.lld.plot(ax=ax)

plt.show()

if __name__ == "__main__":
    pass
