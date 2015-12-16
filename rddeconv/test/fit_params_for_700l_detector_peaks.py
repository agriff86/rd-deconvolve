#!/usr/bin/env python
# coding: utf-8

"""
    Fit parameters to 700L detector spike tests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    dfraw = util.get_raw_data()
    # dead-time correction - N = Nm/(1-Nm*tau/T)
    T = 60.0 # 60 second counting interval
    tau = 1/250e3 # upper limit estimate
    #tau = 2*tau
    dtc = 1 - dfraw.lld * tau / T
    dfraw['lld_obs'] = dfraw.lld.copy()
    dfraw.lld = np.round(dfraw.lld / dtc)
    print('dead time correction, maximum:', ((1/dtc).max() - 1) * 100, 'percent')
    return dfraw

def get_peaks():
    dfraw = load_data()
    t0 = datetime.datetime(2015,3,18,13,0)
    t_injection = [t0 + datetime.timedelta(days=ii) for ii in range(9)]
    ret = []
    for t in t_injection:
        dt = datetime.timedelta(hours=6)
        dfss = dfraw[t:t+dt].copy()
        t0 = dfss.index[0]
        dfss['seconds_since_injection'] = [(itm-t0).total_seconds() for itm in dfss.index]
        ret.append(dfss)
    return ret




if __name__ == "__main__":
    fdir='./figures/peak-fit-results'
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
    parameters['inj_source_strength'] = 21.2e3 * 1400 * 6 # approximate, check source
    parameters['inj_begin'] = 0.0     # start injecting at T=0
    parameters['inj_duration'] = 60.0 # 60 seconds source injection

    parameters['t_delay'] = 90.0

    parameters['rs'] = 0.95
    parameters['recoil_prob'] = 0.025

    #load obs data
    peaklist = get_peaks()

    if False:
        # test the theoretical model

        dt_output = 60
        tmax = 6*60*60
        npts = tmax/dt_output

        dfmod = tm.detector_model_wrapper(timestep=dt_output,
                                  initial_state=np.zeros(tm.N_state),
                                  external_radon_conc=np.ones(npts) * onemBq / 100.0,
                                  internal_airt_history=np.ones(npts)*(273.15+20),
                                  parameters=parameters)


    chains_list = []

    parameters.pop('transform_radon_timeseries')
    parameters.pop('total_efficiency')
    parameters.pop('total_efficiency_frac_error')


    count = 0
    for dfss in peaklist:
        count += 1
        import emcee_deconvolve_tm
        import theoretical_model as tm

        # adjust Q based on 2h mean vent captor measurement
        u_centre = dfss.inflow.values[:120].mean()
        pipe_area = np.pi * (25e-3)**2
        u_mean = 0.80 * u_centre
        Q_pipe = u_mean * pipe_area
        parameters['Q'] = Q_pipe

        # adjust Q_external based on external flow sensor
        parameters['Q_external'] = dfss.exflow.values[:120].mean() / 60.0 / 1000.0

        # set V_tank to measured volume
        parameters['V_tank'] = 679.0 / 1000.0

        # set t_delay to a more tightly constrained value which works ok for all days
        parameters['t_delay'] = 37.0



        variable_parameter_names = 'Q_external', 'Q', 'V_delay', 'V_tank', 't_delay', 'eff', 'lamp', 'rs'
        parameters['variable_parameter_lower_bounds'] = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0, 0.0, 0.5])
        parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1.0])
        variable_parameters_mu_prior = np.array(
                                [parameters[k] for k in variable_parameter_names])
        variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.01,
                                                 parameters['Q']*0.05,
                                                 parameters['V_delay']*0.002,
                                                 parameters['V_tank']*0.002,
                                                 2.0,
                                                 10.0,
                                                 1/100.0,
                                                 0.1])

        ## Try to encourage the model to fit for a change in recoil_prob
        #variable_parameter_names = 't_delay', 'eff', 'lamp', 'rs'
        #parameters['variable_parameter_lower_bounds'] = np.array([-np.inf, 0.0, 0.0, 0.5])
        #parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, np.inf, 1.0])
        #variable_parameters_mu_prior = np.array(
        #                        [parameters[k] for k in variable_parameter_names])
        #variable_parameters_sigma_prior = np.array([
        #                                         60.0,
        #                                         10.0,
        #                                         1/100.0,
        #                                         0.1])




        # how does the initial guess look?
        t = dfss.seconds_since_injection.values
        Y = tm.detector_model_wrapper(np.round(np.diff(t)[0]),
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
             radon_conc=onemBq * 10 * np.ones(len(dfss)), # or, could be []
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


        # calculate net eff and a to c ratio for each row in the chain
        net_effs = []
        a_to_c = []
        for ii in range(len(df_params_chain)):
            pm = {}
            pm.update(parameters)
            for k in df_params_chain.columns:
                pm[k] = df_params_chain.iloc[ii][k]
            ne = tm.calc_detector_efficiency(pm)
            ac = tm.calc_detector_activity_a_to_c_ratio(pm)
            net_effs.append(ne)
            a_to_c.append(ac)
        df_params_chain['net_eff'] = net_effs
        df_params_chain['a_to_c'] = a_to_c
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

    for k in list(variable_parameter_names) + ['net_eff','a_to_c']:
        fig, ax = plt.subplots()
        df_params_summary.boxplot(k, by='day_number', ax=ax)


    chains_by_day = df_params_summary.groupby('day_number')

    for xvar in 'airt', 'relhum', 'inflow':
        for k in list(variable_parameter_names) + ['net_eff', 'a_to_c']:
            data = []
            positions = []
            for day_number, dfchain in chains_by_day:
                dfss = peaklist[day_number-1]
                data.append(dfchain[k].values)
                positions.append(dfss[xvar][:120].mean())
            fig, ax = plt.subplots()
            ax.boxplot(data,  positions=positions, showfliers=False)
            ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.set_xlabel(xvar)
            ax.set_ylabel(k)
    plt.show()

    #
    # ... draw the plot I want for the paper, matching style of other figures
    #
    def fancy_ylabel(ax,s, pos='left', hoffset=0):
        textsize=mpl.rcParams['font.size']
        if pos == 'left':
            ann = ax.annotate(s,xy=(0,1), xycoords='axes fraction',
                        textcoords='offset points', xytext=(-28+hoffset, 10),
                        fontsize=textsize)
        elif pos == 'right':
            ann = ax.annotate(s,xy=(1,1), xycoords='axes fraction',
                        textcoords='offset points', xytext=(28+hoffset, 10),
                        fontsize=textsize,
                        horizontalalignment='right')
        return ann

    xvar = 'relhum'
    yvar = 'a_to_c'
    data = []
    positions = []
    for day_number, dfchain in chains_by_day:
        dfss = peaklist[day_number-1]
        data.append(dfchain[k].values)
        positions.append(dfss[xvar][:60].mean())
    fig, ax = figure_template('acp-1')
    ax.boxplot(data,  positions=positions,
                showfliers=False, showcaps=False,
                boxprops=dict(color="#0072B2"),
                medianprops=dict(color="#0072B2"))
    ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Relative humidity (%)')
    fancy_ylabel(ax, 'Polonium-218 counts / Polonium-214 counts')

    savefig(fig, 'bayes-fit-to-peaks-relh-versus-po-count-ratio-boxplot', fdir='./figures/')

    # error bar plot version
    yplt = [np.mean(itm) for itm in data]
    yerr = [np.std(itm) for itm in data]
    fig, ax = figure_template('acp-1')
    ax.errorbar(x=positions, y=yplt, yerr=yerr, color="#0072B2", linestyle='', marker='.')
    ax.set_xlabel('Relative humidity (%)')
    fancy_ylabel(ax, 'Polonium-218 counts / Polonium-214 counts')
    ax.set_yticks([0.9, 0.95, 1.0])
    savefig(fig, 'bayes-fit-to-peaks-relh-versus-po-count-ratio', fdir='./figures/')

    if False:
        # seaborn-style plots
        import seaborn as sns
        sns.set_style('ticks')
        sns.set_context('paper')

        #sns.boxplot(x="day_number", y="rs", hue="a_to_c", data=df_params_summary, palette="PRGn")

        for xvar in 'airt', 'relhum', 'inflow':
            for k in variable_parameter_names + ['net_eff', 'a_to_c']:
                data = []
                positions = []
                for day_number, dfchain in chains_by_day:
                    dfss = peaklist[day_number-1]
                    data.append(dfchain[k].values)
                    positions.append(dfss[xvar][:120].mean())
                positions = np.array(positions)
                fig, ax = figure_template('acp-1')
                sns.boxplot(x=positions, y=data, showfliers=False, ax=ax,
                            palette='Paired', widths=0.05*(positions.max()-positions.min()))
                ax.xaxis.set_major_locator(mpl.ticker.AutoLocator())
                ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                ax.set_xlabel(xvar)
                ax.set_ylabel(k)


    plt.show()
