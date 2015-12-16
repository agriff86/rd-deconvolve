#!/usr/bin/env python
# coding: utf-8

"""
Richardson-Lucy deconvolution in one dimension
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import datetime
import time
import glob


# a standard set of detector parameters
standard_parameters = dict(
            Q = 0.0122, 
            rs = 0.95, 
            lamp = 1/180.0,
            eff = 0.17815,
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 700.0 / 1000.0,
            t_delay = 60.0,
            interpolation_mode = 1,
            expected_change_std = 1.1,
            total_efficiency = 0.154, #scott's cal
            total_efficiency_frac_error = 0.025,
            transform_radon_timeseries = True,
            cal_source_strength = 0.0,
            cal_begin = 0.0,
            cal_duration = 0.0)
standard_parameters['recoil_prob'] = 0.5*(1-standard_parameters['rs'])



def deconvlucy1d(g, psf, iterations=-1, reg='none', lambda_reg=0.002, tol=0.02):
    """
    Richardson-Lucy Deconvolution

    Parameters
    ----------
    g : one-dimensionsal array
        The time-series to be decovoluved

    psf : one-dimensional array
        The point-spread function

    iterations : integer
        Number of iterations to use in the algorithm

    reg : {'none', 'tv'}
        Regularisation method: 'tv' for total variation, 'none' for no
        regularisation

    lambda_reg : float
        Regularisation parameter

    tol : float
        Stop iterations when the relative change is less than tol

    Returns
    -------
    f : one-dimensional array
        Devolvolved time-series
    """
    FLAG_DEBUG = False
    #initial guess for the output
    f = np.ones(g.shape)*g.mean()
    #reversed point spread function
    psfr = psf[::-1]
    M = len(psfr)
    ii = 0
    while True:
        if reg == 'tv':
            tmp = np.diff(f)/np.abs(np.diff(f))
            #use zero where diff(f) is zero
            tmp[np.diff(f) == 0.0] = 0.0
            grad_term = np.diff(tmp)
            grad_term = np.r_[0, grad_term, 0]
            reg_factor = 1.0/(1.0 - lambda_reg * grad_term)
        tmp = (g/np.convolve(f, psf, mode='same'))
        step = np.convolve(tmp, psfr, mode='same')

        if reg == 'tv':
            step *= reg_factor

        f *= step

        # do we need to stop?
        if iterations <= 0:
            rel_change = np.abs(step[M:-M]-1.0).max()
            if ii % 100 == 0 and FLAG_DEBUG:
                print(rel_change)
                print(rel_change<tol)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2,1)
                ax[0].semilogy(step[10:-10])
                ax[1].semilogy(f[10:-10])
                ax[1].semilogy(g[10:-10])
                plt.show()

            if rel_change < tol:
                print("stopping after {} iterations".format(ii))
                break
        else:
            if ii > iterations:
                break
        ii += 1
    return f

def parse_hhmm_string(s):
    return datetime.datetime.strptime(s, '%H:%M').time()

def load_radon(fname, sheet_name=None, subhrs=0):
    """load radon data from Sylvester's excel spreadsheet
    subhrs is to work around different conventions (hour 0-23 or 1-24 depending
    on who created the file)"""
    if sheet_name is None:
        df = pd.read_csv(fname)
        df.columns = [itm.strip().lower() for itm in df.columns]
        df['time'] = df.time.apply(parse_hhmm_string)
    else:
        df = pd.read_excel(fname, sheet_name)
    df.columns = [itm.strip().lower() for itm in df.columns]

    time = [ datetime.datetime.combine(datetime.date(int(itm[1]['year']),
                                                     int(itm[1]['month']),
                                                     int(itm[1]['dom'])),
                                       itm[1]['time']) for itm in df.iterrows()]

    df.index = time
    if subhrs != 0:
        df.index = df.index + datetime.timedelta(hours=subhrs)

    return df

def load_radon_v1():
    def convdate(s):
        """parse a timestamp"""
        fmt = '%Y-%m-%d %H:%M'
        timestamp = datetime.datetime.strptime(s,fmt)
        return timestamp
    #this is radon concentration converted to Bq/m3 at 1013 mbar & 0 degC 
    fname = 'radon-at-stp-bern-and-jfj.csv'
    d = np.recfromcsv(rn_ddir+'/'+fname, missing=' ', converters={0:convdate})
    d.dtype.names = ['time'] + list(d.dtype.names[1:])
    #zero radon concentrations screw with me later, and don't make sense anyway
    d.be[d.be==0] = np.NaN
    #radon correlates better with other tracers if we de-lag it
    #extra note: I think this is just a local time to UTC thing
    nlag = 1
    d.jfj[0:-nlag] = d.jfj[nlag:]
    d.be[0:-nlag] = d.be[nlag:]
    return d


def load_kopri():
    """
    load kopri radon data
    """
    fnames = sorted(glob.glob('data-kopri/KO_??_??.CSV'))
    dfl = []
    for fname in fnames:
        df = pd.read_csv(fname, parse_dates=[' Time'])
        df.columns = [itm.strip().lower() for itm in df.columns]
        t = []
        for ii in range(len(df)):
            d = datetime.date(df.year[ii], df.month[ii], df.dom[ii])
            tod = df.time[ii].time()
            t.append(datetime.datetime.combine(d,tod))
        df.index = t
        dfl.append(df)
    df = pd.concat(dfl)
    return df



def load_cabauw(level=20):
    """
    load cabauw radon data
    
    level is 20 or 200 (m)
    """
    if level==20:
        fnames = sorted(glob.glob('data-cabauw/CA*.CSV'))
    elif level==200:
        fnames = sorted(glob.glob('data-cabauw/CB*.CSV'))
    dfl = []
    for fname in fnames:
        df = pd.read_csv(fname, parse_dates=[' Time'])
        df.columns = [itm.strip().lower() for itm in df.columns]
        t = []
        for ii in range(len(df)):
            d = datetime.date(df.year[ii], df.month[ii], df.dom[ii])
            tod = df.time[ii].time()
            t.append(datetime.datetime.combine(d,tod))
        df.index = t
        dfl.append(df)
    df = pd.concat(dfl)
    # sort and drop duplicates
    dupecols = 'year month dom time'.split()
    df = df.drop_duplicates(subset=dupecols).sort()
    
    return df

def load_richmond():
    """
    load richmond data (6-minute samples)
    """
    fnames = sorted(glob.glob('data-richmond-2011/*.CSV'))
    dfl = []
    for fname in fnames:
        df = pd.read_csv(fname, parse_dates=[' Time'])
        df.columns = [itm.strip().lower() for itm in df.columns]
        t = []
        for ii in range(len(df)):
            d = datetime.date(df.year[ii], df.month[ii], df.dom[ii])
            tod = df.time[ii].time()
            t.append(datetime.datetime.combine(d,tod))
        df.index = t
        dfl.append(df)
    df = pd.concat(dfl)
    # sort and drop duplicates
    dupecols = 'year month dom time'.split()
    df = df.drop_duplicates(subset=dupecols).sort()
    # rename some columns
    df['exflow'] = df.flow
    df['airt'] = df.temp
    return df





def prf30_from_prf6(one_sided_prf, tag='v3'):
    """
    create a point response function at 30 minute resolution
    (from a prf at 6 minute resolution)
    """
    prf = one_sided_prf
    import warnings
    if not np.allclose(prf.sum(), 1.0, rtol=1e-3):
        warnings.warn("one_sided_prf.sum() = {}, should be 1.0".format(prf.sum()))

    # there are different models we can use: v1 - a radon spike at t=0
    #                                        v2 - a radon spike at t=15min
    #                                        v3 - a radon pulse from t=0 to t=30
    #  v2 and v3 are, numerically, pretty much the same

    prf30v1 = np.reshape(prf[:-1], (-1,5)).mean(axis=1) # t=0
    prf30v2 = np.reshape(np.r_[0,0,prf[:-3]], (-1,5)).mean(axis=1) # t=15min
    #add up five six minute ones, each shifted by one step
    prf30v3 = np.zeros(len(prf[:-1]))
    for ii in range(5):
        prf30v3[ii:] = prf30v3[ii:] + prf[:-ii-1] / 5.0
    prf30v3 = np.reshape(prf30v3, (-1,5)).mean(axis=1)

    #which version to use?
    prf_options = dict(v1=prf30v1,
                       v2=prf30v2,
                       v3=prf30v3)
    prf30 = prf_options[tag]

    # ensure that the result sums to 1
    prf30 /= prf30.sum()

    prf30_symmetric = np.r_[np.zeros(len(prf30)-1), prf30]

    return prf30, prf30_symmetric

def plot_with_uncertainty(df, column_name, ax):
    t = df.index
    low = df[column_name+'_p10'].values
    high = df[column_name+'_p90'].values
    r1 = ax.fill_between(t, low, high, alpha=0.2, color='k', linewidth=0)
    #low = df[column_name+'_p16'].values
    #high = df[column_name+'_p84'].values
    #r2 = ax.fill_between(t, low, high, alpha=0.2, color='k', linewidth=0)
    r2 = ax.plot(t, df[column_name+'_mean'].values, 'k')
    return [r1,r2]

class timewith():
    """Timer implemented as a context manager

    Example:

    # prints something like:
    # fancy thing done with something took 0.582462072372 seconds
    # fancy thing done with something else took 1.75355315208 seconds
    # fancy thing finished took 1.7535982132 seconds
    with timewith('fancy thing') as timer:
        expensive_function()
        timer.checkpoint('done with something')
        expensive_function()
        expensive_function()
        timer.checkpoint('done with something else')

    ref: https://zapier.com/engineering/profiling-python-boss/
    """
    def __init__(self, name=''):
        self.name = name
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name=''):
        print('{timer} {checkpoint} took {elapsed} seconds'.format(
            timer=self.name,
            checkpoint=name,
            elapsed=self.elapsed,
        ).strip())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass



def extrapolate_time(t, n):
    """
    Extrapolate a uniformly-spaced time array by adding n points before t[0]
    """
    dt = t[1] - t[0]
    extrap = t[0] + np.arange(-n,0,1) * dt
    return np.concatenate([extrap, t])


def get_raw_data():
    import util
    fnames = ['data-controlled-test-2/T1Mar15e.CSV',
              'data-controlled-test-2/T1Apr15e.CSV']
    dfobs = [util.load_radon(itm) for itm in fnames]
    dfobs = pd.concat(dfobs)
    return dfobs

def get_10min_data():
    dfobs = get_raw_data()
    # resample to 10 minute counts
    dfobsrs = dfobs.resample('10min', label='right', closed='right', how='sum')[['lld']]
    num_obs_per_interval = dfobs.resample('10min', label='right', closed='right', how='count')
    dfobsrs = dfobsrs.ix[num_obs_per_interval.lld == num_obs_per_interval.lld.max()]
    # check resample:
    # dfobs.head(11).lld.sum()-dfobs.head(1).lld.values == dfobsrs.lld[0]    
    return dfobsrs

def get_simulated_data():
    """
    Get data simulating a square wave lab test
    """
    # load point response function (prf)
    one_sided_prf = pd.read_csv('one_sided_prf.csv', index_col=0).prf.values
    symmetric_prf = pd.read_csv('symmetric_prf.csv', index_col=0).prf.values

    # make the prf smaller for less computation
    one_sided_prf = one_sided_prf[:61]  # drops to 10-4 at 60, 10-3 at 30
    one_sided_prf /= one_sided_prf.sum()
    
    t0 = datetime.datetime(2014,12,1)
    t1 = datetime.timedelta(hours=24)
    dt = datetime.timedelta(minutes=6)
    tp0 = t0+datetime.timedelta(hours=12)
    tp1 = t0+datetime.timedelta(hours=13)
    t = [t0+ii*dt for ii in range(int(t1.total_seconds() / dt.total_seconds()))]
    t = np.array(t)
    detector_background = 6
    ambient = 100
    peak = 40000
    
    true = np.zeros(len(t)) + ambient
    true[ (t>tp0) & (t<=tp1) ] = peak
    detector_expected = np.convolve(true, one_sided_prf, mode='valid')
    detector_expected += detector_background
    detector_expected = np.r_[ [detector_expected[0]]*(len(one_sided_prf)-1), 
                               detector_expected]
    
    simulated_obs = np.random.poisson(detector_expected)
    
    df = pd.DataFrame(index=t, data=dict(truth=true,
                                         detector_expected=detector_expected,
                                         lld=simulated_obs))
    return df, one_sided_prf, symmetric_prf

def get_goulburn_data(missing_value=500):
    
    one_sided_prf = pd.read_csv('one_sided_prf.csv', index_col=0).prf.values
    symmetric_prf = pd.read_csv('symmetric_prf.csv', index_col=0).prf.values
    # make the prf smaller for less computation
    one_sided_prf = one_sided_prf[:61]  # drops to 10-4 at 60, 10-3 at 30
    one_sided_prf /= one_sided_prf.sum()

    prf30, prf30_symmetric = prf30_from_prf6(one_sided_prf)

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
        df_glb.lld[df_glb.lld<0] = missing_value
        return df_glb

    df_glb = load_glb()
    
    return df_glb, prf30, prf30_symmetric



if __name__ == "__main__":
    pass
