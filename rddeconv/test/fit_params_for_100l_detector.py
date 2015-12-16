#!/usr/bin/env python
# coding: utf-8

"""
    Fit parameters to 100L detector delay-volume tests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import datetime
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import pygrib

import util
from tm_optimisation import fit_to_obs

#from micromet.map import create_map
#from micromet.plot import figure_template, fancy_ylabel

def load_data():
    fname = '100-l-volume-delay-tests/data/100L_test.csv'
    df = pd.read_csv(fname, parse_dates=True, index_col='TIMESTAMP')
    df.columns = [itm.lower() for itm in df.columns]
    df['time'] = df.index
    df['airt'] = df.temp
    return df

def get_peaks():
    """
    return a list of injection peaks

     - injection starting at 1300 each day, for 1 minute
    """
    df = load_data()
    injstart = (df.index.time == datetime.time(13,0)).nonzero()[0]
    npts = 12
    ret = []
    for ii in injstart:
        dfss = df.iloc[ii:ii+npts].copy()
        dfss['seconds_since_injection'] = \
            [itm.total_seconds() for itm in (dfss.time - dfss.time.iloc[0])]
        #dfss.index = dfss.seconds_since_injection
        ret.append(dfss)
    return ret




if __name__ == "__main__":
    fdir='./100-l-volume-delay-tests/figures'
    mpl.rcParams['font.sans-serif'] = ['Source Sans Pro']
    import theoretical_model as tm
    onemBq = 1.0/tm.lamrn/1000.0 #radon atoms per m3 for one mBq/m3 conc

    from micromet.plot import figure_template

    # color palette from seaborn
    seaborn_palettes = dict(
        deep=["#4C72B0", "#55A868", "#C44E52",
              "#8172B2", "#CCB974", "#64B5CD"],
        muted=["#4878CF", "#6ACC65", "#D65F5F",
               "#B47CC7", "#C4AD66", "#77BEDB"],
        pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
                "#D0BBFF", "#FFFEA3", "#B0E0E6"],
        bright=["#003FFF", "#03ED3A", "#E8000B",
                "#8A2BE2", "#FFC400", "#00D7FF"],
        dark=["#001C7F", "#017517", "#8C0900",
              "#7600A1", "#B8860B", "#006374"],
        colorblind=["#0072B2", "#009E73", "#D55E00",
                    "#CC79A7", "#F0E442", "#56B4E9"],
    )
    mpl.rcParams["axes.color_cycle"] = list(seaborn_palettes['colorblind'])
    def savefig(fig, name, fdir=fdir):
        import os
        fig.savefig(os.path.join(fdir, name+'.pdf'), transparent=True, dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fdir, name+'.png'), transparent=False, dpi=300, bbox_inches='tight')


    parameters = dict(**util.standard_parameters) # deep copy
    parameters['inj_source_strength'] = 21.2e3 * 1000 # approximate, unflushed 21.2 kBq source
    parameters['inj_begin'] = 0.0     # start injecting at T=0
    parameters['inj_duration'] = 60.0 # 60 seconds source injection


    parameters['V_delay'] = 200/1000.0 # 200L delay volume
    parameters['V_tank'] = 100/1000.0  # 100L detector
    parameters['Q'] = 1.2 / 1000.0 # Martin 2004, 1.2 l/sec
    parameters['Q_external'] = 14 / 60.0 / 1000.0 # mean from full time series
    parameters['t_delay'] = 90.0

    parameters['rs'] = 0.95
    parameters['recoil_prob'] = 0.025

    #load obs data, first 4 peaks use 200l delay, remaining use 50L delay
    peaklist = get_peaks()

    if False:
        # test the theoretical model

        dt_output = 60*30
        tmax = 6*60*60
        npts = tmax/dt_output

        dfmod = tm.detector_model_wrapper(timestep=dt_output,
                                  initial_state=np.zeros(tm.N_state),
                                  external_radon_conc=np.ones(npts) * onemBq,
                                  internal_airt_history=np.ones(npts)*(273.15+20),
                                  parameters=parameters)


    chains_list = []

    parameters.pop('transform_radon_timeseries')
    parameters.pop('total_efficiency')
    parameters.pop('total_efficiency_frac_error')


    count = 0
    for dfss in peaklist:
        count += 1
        if count > 4:
            delay_vol = 50 / 1000.0
        else:
            delay_vol = 200 / 1000.0
        import emcee_deconvolve_tm
        import theoretical_model as tm

        parameters['V_delay'] = delay_vol # 200L delay volume

        variable_parameter_names = 'Q_external', 'Q', 'V_delay', 'V_tank', 't_delay', 'eff'
        parameters['variable_parameter_lower_bounds'] = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
        parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        variable_parameters_mu_prior = np.array(
                                [parameters[k] for k in variable_parameter_names])
        variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.01,
                                                 parameters['Q']*0.5,
                                                 parameters['V_delay']*0.05,
                                                 parameters['V_tank']*0.05,
                                                 60.0,
                                                 1.0])



        # how does the initial guess look?
        t = dfss.seconds_since_injection.values
        Y = tm.detector_model_wrapper(np.diff(t)[0],
                    initial_state=np.array([0.,0,0,0,0]),
                    external_radon_conc=np.zeros(len(dfss)) + onemBq/100.0,
                    internal_airt_history=dfss.airt.values + 273.15,
                    parameters=parameters,
                    interpolation_mode=1,
                    return_full_state=False)
        dfss['lldmod'] = np.r_[0.0, Y.Acc_counts.diff().values[1:]]
        f, ax = plt.subplots()
        dfss[['lld','lldmod']].plot(ax=ax)
        ax.set_title(dfss.index.to_pydatetime()[0].date().strftime('%b %Y'))

        # adjust eff to match to obs
        parameters['eff'] = parameters['eff'] * dfss.lld.mean()/dfss.lldmod.mean()

        variable_parameters_mu_prior = np.array(
                                    [parameters[k] for k in variable_parameter_names])



        fit_ret = emcee_deconvolve_tm.fit_parameters_to_obs(t, observed_counts=dfss.lld.values,
             radon_conc=onemBq/100 * np.ones(len(dfss)), # or, could be []
             internal_airt_history=dfss.airt.values,
             parameters=parameters,
             variable_parameter_names = variable_parameter_names,
             variable_parameters_mu_prior = variable_parameters_mu_prior,
             variable_parameters_sigma_prior = variable_parameters_sigma_prior,
             iterations=100,
             thin=1,
             keep_burn_in_samples=False,
             nthreads=1)


        (sampler, A, mean_est, low, high, parameters, map_radon_timeseries,
            rl_radon_timeseries, rltv_radon_timeseries) = fit_ret
        popt = A.mean(axis=0)
        pmap = A[0,:]
        print("(from bayes_fit...) pmap:", pmap)

        params_chain = A[:, parameters['nstate']:parameters['nhyper']+parameters['nstate']]
        radon_conc_chain = A[:, parameters['nhyper']+parameters['nstate']:]

        b = sampler.chain[:, :, parameters['nhyper']+parameters['nstate']:]

        df_params_chain = pd.DataFrame(data = params_chain, columns=parameters['variable_parameter_names'])

        # calculate net eff for each row in the chain
        net_effs = []
        for ii in range(len(df_params_chain)):
            pm = {}
            pm.update(parameters)
            for k in df_params_chain.columns:
                pm[k] = df_params_chain.iloc[ii][k]
            ne = tm.calc_detector_efficiency(pm)
            net_effs.append(ne)
        df_params_chain['net_eff'] = net_effs
        chains_list.append(df_params_chain)

        # how does the optimised solution look?
        p_mod = dict()
        p_mod.update(parameters)
        p_mod.update(df_params_chain.mean())
        deconv_radon_conc = radon_conc_chain.mean(axis=0)
        from scipy.stats import scoreatpercentile

        dfss['lldmod_opt'] = np.r_[np.NaN,
           emcee_deconvolve_tm.detector_model_specialised(popt, p_mod)]
        #from MAP solution
        dfss['lldmod_map'] = np.r_[np.NaN,
           emcee_deconvolve_tm.detector_model_specialised(pmap, p_mod)]

        f, ax = plt.subplots()
        dfss[['lld','lldmod_opt', 'lldmod_map']].plot(ax=ax)

    cols = df_params_chain.columns
    for ii,dft in enumerate(chains_list):
        dft['day_number'] = ii + 1
    df_params_summary = pd.concat(chains_list)

    for k in cols:
        fig, ax = plt.subplots()
        df_params_summary.boxplot(k, by='day_number', ax=ax)

    for ii,k in enumerate(df_params_chain.columns()):

        pd.concat()


    assert(False)




    f, ax = plt.subplots()
    fitparams = []
    results = []
    airt_mean = []
    airt = []

    peaklist = get_peaks()
    count = 0

    for dfss in peaklist:
        parameters['t_delay'] = 300.0
        parameters['Q_external'] = dfss.flow.mean() / 60.0 / 1000.0
        if count > 3:
            parameters['V_delay'] = 50/1000.0
        else:
            parameters['V_delay'] = 200/1000.0
        t = dfss.time[0]
        dfss['relhum'] = 60.0
        fitres = fit_to_obs(dfss, parameters)
        fitparams.append(fitres.x)
        results.append(fitres)
        airt_mean.append(dfss.airt.ix[t:t+datetime.timedelta(hours=2)].mean())
        airt.append(dfss.airt.ix[t:t+datetime.timedelta(minutes=20)])
        plt.show()
        count += 1
