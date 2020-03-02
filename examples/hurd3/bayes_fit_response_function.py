#!/usr/bin/env python
# coding: utf-8

"""
Work out the response function of the detector from calibration peaks

TODO: refactor

"""




from __future__ import (absolute_import, division,
                        print_function)

#import warnings
#warnings.simplefilter('error')

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

import rddeconv.util

import rddeconv.fast_model

import rddeconv.theoretical_model as tm
import emcee_deconvolve_tm

lamrn = 2.1001405267111005e-06
lama = 0.0037876895112565318
lamb = 0.00043106167945270227
lamc = 0.00058052527685087548

def dilate_bool(b, passes=1):
    """
    dilate a boolan mask so that output[n] is true if 
        input[n-1] || input[n] || input[n+1] 
    is true
    """
    out = b.copy()
    out[:-1] = np.logical_or(out[:-1], b[1:])
    out[1:] = np.logical_or(out[1:], b[:-1])
    if passes > 1:
        out = dilate_bool(out, passes-1)
    return out

def get_calibration_spikes(site='ko', return_all_data=False):

    if site == 'ko':
        df = util.load_kopri()
        # get data from july 10th onwards (five spikes)
        df = df[datetime.datetime(2013,7,10):]
        source_strength = 9.9e3 #Radium activity, Bq
    elif site == 'jb':
        df = util.load_jangbobo()
        df = df[datetime.datetime(2015,12,30):]
        # serial number 231
        source_strength = 9.669e3
    elif site == 'ri':
        df = util.load_richmond()
        source_strength = 118.19 #Radium activity, Bq
    elif site in ['ca','cb']:
        height = dict(ca=20, cb=200)[site]
        df = util.load_cabauw(height)
        source_strength = 22.343e3
        ## data from June 2013 (bad calibration until proior to that)
        df = df[datetime.datetime(2013,6,1):]
        # data from all of 2013
        #df = df[datetime.datetime(2013,1,1):]
    else:
        assert(False)

    # add inlet radon concentration (flag==2)
    # worked backwards from Scott's spreadsheet, should be checked
    # Stations/09_KOPRI/Rn_Data/02_Calibrated/KO_Internal_DB_2013_v12.xlsx
    # sheetname : Cal
    # TODO: talk to Scott, especially for the 1.07 factor!
    # source_strength = 1000*2.52*9.9/20.0 / 1.07
    # ---> talked to Scott.  Source Strength is 9.9e3 Bq (Radium)
    C = source_strength * lamrn / (df.exflow / (1000. * 60.) )
    # works out at about 25 Bq/m3
    if 'flag' in df.columns:
        df['cal_radon_conc'] = (df.flag==2).astype(np.int) * C
    else:
        df['cal_radon_conc'] = C

    # air temperature: convert to K
    df.airt += 273.15

    # time for each cal starting
    if site == 'ri':
        # can't detect calibrations automatically
        trefs = [datetime.datetime(2011, 3, 16, 8, 30),
                 datetime.datetime(2011, 12, 16, 9, 30) ]
        # for testing, just use one
        trefs = trefs[-1:]
    else:
        trefs = df.ix[((df.flag.diff() == 2) & (df.flag == 2))].index.to_pydatetime()
    dfl = []

    dates_to_exclude_ko = set( [datetime.date(2014,2,6),
                             datetime.date(2014,3,1)] )
    if site=='ko':
        for t in trefs:
            dfss = df.ix[t-datetime.timedelta(minutes=60):
                        t+datetime.timedelta(hours=12)].copy()
            # adjust the cal to match injection (flag is longer)
            # (just from looking for best match with obs)
            dfss.cal_radon_conc.iloc[2] = 0.0
            dfss.cal_radon_conc.iloc[6*2+1:] = 0.0

            if not dfss.lld[0] > 100 and ( not
                   dfss.index.to_pydatetime()[0].date() in dates_to_exclude_ko):
                dfl.append(dfss)
    elif site=='jb':
        for t in trefs:
            dfss = df.ix[t-datetime.timedelta(hours=4):
                        t+datetime.timedelta(hours=12)].copy()
            # adjust the cal to match injection (flag is longer)
            # (just from looking for best match with obs)
        #    dfss.cal_radon_conc.iloc[2] = 0.0
            dfss.cal_radon_conc.iloc[9*2:] = 0.0
            dfl.append(dfss)
    elif site == 'ri':
        for t in trefs:
            dfss = df.ix[t-datetime.timedelta(minutes=60):
                        t+datetime.timedelta(hours=12)].copy()
            # adjust the cal to match injection (flag is longer)
            # (just from looking for best match with obs)
            dfss.cal_radon_conc.iloc[:10] = 0.0
            # 5h injection, 5 minute counts
            dfss.cal_radon_conc.iloc[5*10+10+1:] = 0.0
            dfl.append(dfss)
    else:

        bad_data = (df.exflow < 90)
        bad_data = dilate_bool(bad_data, passes=6)
        df['bad_data_flag'] = bad_data
        for t in trefs:
            dfss = df.ix[t-datetime.timedelta(minutes=150):
                        t+datetime.timedelta(hours=14)].copy()
            # adjust the cal to match injection (flag is longer)
            # (just from looking for best match with obs)
            dfss.cal_radon_conc.iloc[5] = 0.0
            dfss.cal_radon_conc.iloc[10*2+6:] = 0.0
            # the flag is too short in the earlier records
            dfss.cal_radon_conc.iloc[17] = C[dfss.iloc[15].name]

            # all time steps must be equal
            dt = np.diff(dfss.index.to_pydatetime())
            peak_ok = np.alltrue(dt==dt[0])

            if peak_ok:
                dfl.append(dfss)

    if return_all_data:
        ret = [df, dfl]
    else:
        ret = dfl
    return ret



if __name__ == "__main__":


    site = 'jb' # 'ko', 'ca', or 'cb', or 'ri'

    import sys
    #print(sys.argv)
    if len(sys.argv) > 1:
        site = sys.argv[1]
        print('processing site', site)

    if site == 'ko':
        expected_net_eff = 0.360
        expected_background = 10./1800.0 # detector only (TODO:check)
        source_strength = 9.669e3 #Radium activity, Bq
        cal_begin = 2 * 1800.0
        cal_duration = 6 * 3600
    elif site == 'jb':
        expected_net_eff = 0.312
        expected_background = 26.6/1800.0 # detector only
        source_strength = 9.9e3 #Radium activity, Bq
        cal_begin = 8 * 1800.0
        cal_duration = 5 * 3600
    elif site == 'ri':
        # see data files in Radon_Public
        expected_net_eff = 0.319
        expected_background = 500.0/3600 # in December, 500 counts per hour
        cal_begin = 3600.0 # I'm keeping one hour prior to the cal
        cal_duration = 5*3600.0 # 5 h injection
        source_strength = 118.19e3
    else:
        expected_net_eff = 0.325
        expected_background = 50./1800. # detector only
        source_strength = 22.343e3
        cal_begin = 5 * 1800. # depends on code in get_calibration_spikes
        cal_duration = 10 * 3600. # sometimes 10h, sometimes 6h-need a check later

    from matplotlib.backends.backend_pdf import PdfPages


    with PdfPages('multipage_cal_peaks_site-{}.pdf'.format(site)) as pdfpages:


        dfl = get_calibration_spikes(site=site)

        #dfl = dfl[-5:] # for testing

        # show the spikes
        for itm in dfl:
            f, ax = plt.subplots()
            itm.lld.plot(ax=ax)
            ax.set_title(itm.index.to_pydatetime()[0].date().strftime('%b %Y'))

        chains_list = []
        # do the fit
        # dfl = dfl[0:2] # for testing
        for dfss in dfl:
            dfss = dfss.copy()
            # parameters
            parameters = {}
            parameters.update(util.standard_parameters)
            parameters['Q_external'] = dfss.exflow.mean() / 1000.0 / 60.0
            parameters['V_delay'] = 0.0
            parameters['V_tank'] = 1.500
            if site == 'jb':
                # 1200L detector
                parameters['V_tank'] = 1.2
            parameters['interpolation_mode'] = 1
            parameters['lamp'] /= 2.0 # bigger tank, lower plateout (but dunno for sure)
            parameters['t_delay'] = 60.0
            parameters['detector_background_cps'] = expected_background


            parameters.pop('transform_radon_timeseries')
            parameters.pop('total_efficiency')
            parameters.pop('total_efficiency_frac_error')

            # calibration parameters
            parameters['cal_source_strength'] = source_strength
            parameters['cal_begin'] = cal_begin
            if not site == 'ri':
                parameters['cal_duration'] = (dfss.cal_radon_conc > 0).sum() * 1800.0
            else:
                parameters['cal_duration'] = cal_duration

            # check net sensitivity/efficiency
            ne = tm.calc_detector_efficiency(parameters)
            # set eff so that net eff equals expectation
            parameters['eff'] = parameters['eff'] * expected_net_eff / ne

            # times relative to injection
            times = dfss.index.to_pydatetime()
            tzero = times[0]
            t = np.array([ (itm-tzero).total_seconds() for itm in times])
            tres = t[1] - t[0]
            # priors on parameters
            variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
            parameters['variable_parameter_lower_bounds'] = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
            parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, 2.0, np.inf, np.inf, np.inf])
            variable_parameters_mu_prior = np.array(
                                    [parameters[k] for k in variable_parameter_names])
            variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                                     parameters['Q']*0.02,
                                                     0.05,
                                                     1/100.0,
                                                     60.,
                                                     0.2])

            if site == 'jb':
                # assume smooth variation in background radon concentration
                # typically this is set to 1.25
                parameters['expected_change_std'] = 1.02
                # make the proiors less restrictive - try harder to fit the obs
                variable_parameters_sigma_prior = np.array([
                    parameters['Q_external'] * 0.02,
                    parameters['Q']*0.02,
                    0.1,
                    1/10.0,
                    120.,
                    0.5])


            if site=='ri':
                # Injection was not precisely 5 h - all manual
                variable_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff', 'cal_duration'
                parameters['variable_parameter_lower_bounds'] = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, 0.0, 3600*4])
                parameters['variable_parameter_upper_bounds'] = np.array([np.inf, np.inf, 2.0, np.inf, np.inf, np.inf, 3600*6])
                variable_parameters_mu_prior = np.array(
                                        [parameters[k] for k in variable_parameter_names])
                variable_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                                         parameters['Q']*0.02,
                                                         0.05,
                                                         1/100.0,
                                                         60.,
                                                         0.2,
                                                         10*60.0])
            # how does the initial guess look?
            Y = tm.detector_model_wrapper(np.diff(t)[0],
                        initial_state=np.array([0.,0,0,0,0]),
                        external_radon_conc=dfss.cal_radon_conc.values/lamrn*0.0,
                        internal_airt_history=dfss.airt.values,
                        parameters=parameters,
                        interpolation_mode=1,
                        return_full_state=False)
            dfss['lldmod'] = np.r_[0.0, Y.Acc_counts.diff().values[1:]] + \
                             expected_background*tres
            f, ax = plt.subplots()
            dfss[['lld','lldmod']].plot(ax=ax)
            ax.set_title(dfss.index.to_pydatetime()[0].date().strftime('%b %Y'))

            plt.show()

            # fit to obs

            if True:
                fit_ret = emcee_deconvolve_tm.fit_parameters_to_obs(t, observed_counts=dfss.lld.values,
                     radon_conc=[], #dfss.cal_radon_conc.values/lamrn*0.0,
                     internal_airt_history=dfss.airt.values,
                     parameters=parameters,
                     variable_parameter_names = variable_parameter_names,
                     variable_parameters_mu_prior = variable_parameters_mu_prior,
                     variable_parameters_sigma_prior = variable_parameters_sigma_prior,
                     iterations=2000,
                     thin=10,
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
                dfss['deconv_radon_conc'] = deconv_radon_conc
                dfss['conc_p25'],dfss['conc_p50'],dfss['conc_p75'] = \
                     scoreatpercentile(radon_conc_chain, [25,50,75], axis=0)

                dfss['lldmod_opt'] = np.r_[np.NaN,
                   emcee_deconvolve_tm.detector_model_specialised(popt, p_mod)]
                #from MAP solution
                dfss['lldmod_map'] = np.r_[np.NaN,
                   emcee_deconvolve_tm.detector_model_specialised(pmap, p_mod)]
                # without calibration peak
                p_mod['cal_source_strength'] = 0.0
                dfss['lldmod_nocal_opt'] = np.r_[np.NaN,
                   emcee_deconvolve_tm.detector_model_specialised(popt, p_mod)]

                f, axl = plt.subplots(nrows=2, figsize=[6,7])
                ax = axl[0]
                dfss[['lld','lldmod_opt', 'lldmod_map', 'lldmod_nocal_opt']].plot(ax=ax)
                ax.set_title(dfss.index.to_pydatetime()[0].date().strftime('%b %Y')
                        + ', cal='+str(np.mean(net_effs)) + '; cal_map='+str(net_effs[0]))
                # plot of deconvolved radon concentration
                ax = axl[1]
                (dfss[['deconv_radon_conc', 'conc_p25', 'conc_p50','conc_p75']]*lamrn).plot(ax=ax)
                f.tight_layout()

                pdfpages.savefig(f, bbox_inches='tight')
                plt.show()

        times_list = [itm.index.to_pydatetime()[0] for itm in dfl]
        labels_list = [itm.strftime('%b\n%Y') for itm in times_list]

        nchains = len(chains_list)
        f,ax = plt.subplots(figsize=[nchains*0.6,4])
        ne_list = [dfi.net_eff.values for dfi in chains_list]
        ne_map_list = [dfi.net_eff.iloc[0] for dfi in chains_list]
        ax.boxplot(ne_list, labels=labels_list)
        ax.plot(np.arange(nchains)+1, ne_map_list, 'ro')
        maxval = max([dfi.net_eff.max() for dfi in chains_list])
        if maxval > 0.8:
            ax.set_ylim([0,0.4])

        pdfpages.savefig(f, bbox_inches='tight')

        # na-fill zero values in cal_radon_conc
        for itm in dfl:
            itm[itm.cal_radon_conc == 0] = np.NaN

        df_summary = pd.concat([itm.mean() for itm in dfl], axis=1).T
        f, ax = plt.subplots(figsize=[3,12])
        cols = ['exflow','gm','inflow','lld','uld','tankp','airt','relhum',
                'press', 'cal_radon_conc']
        df_summary[cols].plot(subplots=True, ax=ax)
        pdfpages.savefig(f, bbox_inches='tight')

        # relative humidity effect on calibration
        f,ax = plt.subplots()
        ax.boxplot(ne_list, positions=df_summary.relhum.values, manage_xticks=False)
        ax.plot(df_summary.relhum.values, ne_map_list, 'o')
        ax.set_xlabel('relative humidity')
        ax.set_label('calibration coefficient')

        pdfpages.savefig(f, bbox_inches='tight')

        f, ax = plt.subplots()
        rs_list = [dfi.rs.values for dfi in chains_list]
        rs_map_list = [dfi.rs.iloc[0] for dfi in chains_list]
        ax.boxplot(rs_list, labels=labels_list)
        ax.plot(np.arange(nchains)+1, rs_map_list, 'o')




        df = pd.concat(chains_list)
        f, ax = plt.subplots()
        ax.plot(df.net_eff, df.rs, '.', alpha=1/100.)
        ax.set_xlabel('net_eff')
        ax.set_ylabel('rs')

        f, ax = plt.subplots()
        pd.tools.plotting.scatter_matrix(df, ax=ax, alpha=1/100.)

    # run the model forwards (using last cal)

    # 1. step change
    tres = 30*60

    params_saved = {}
    params_saved.update(parameters)
    # update parameters from obs
    for k in df.columns:
        if k in parameters.keys():
            print('updating',k,'from',parameters[k],'to',df[k].mean())
            parameters[k] = df[k].mean()

    parameters['t_delay'] = 0.0

    for tres in 30*60, 60*60:
        rn_conc = np.zeros(15)
        rn_conc[1:] = 1.0/lamrn # 1Bq/m3
        airt = np.zeros(rn_conc.shape)+273.15+20.0
        ss_counts = tm.calc_detector_efficiency(parameters) * tres
        df_model=tm.detector_model_wrapper(timestep=tres,
                                  initial_state=np.array([0.,0,0,0,0]),
                                  external_radon_conc=rn_conc,
                                  internal_airt_history=airt,
                                  parameters=parameters,
                                  interpolation_mode=0,
                                  return_full_state=False)

        counts = df_model.Acc_counts.diff()
        relcounts = counts/ss_counts
        relcounts.name = "normalised_counts"
        relcounts.index.name = "interval_end_minutes"
        df_results = relcounts.to_frame()
        df_results['corr_factor'] = 1/df_results['normalised_counts']
        print(df_results)
        df_results.to_excel('correction_factor_'+str(tres/60)+'min.xlsx')

    # 2. pulse
    tres = 60.0
    rn_conc = np.zeros(5*60)
    rn_conc[1] = 1.0/lamrn # 1Bq/m3
    airt = np.zeros(rn_conc.shape)+273.15+20.0
    ss_counts = tm.calc_detector_efficiency(parameters) * tres
    df_model=tm.detector_model_wrapper(timestep=tres,
                              initial_state=np.array([0.,0,0,0,0]),
                              external_radon_conc=rn_conc,
                              internal_airt_history=airt,
                              parameters=parameters,
                              interpolation_mode=0,
                              return_full_state=False)

    counts_no_delay = df_model.Acc_counts.diff()
    parameters['V_delay'] = 200.*2/1000.0 # two 200L barrels
    df_model=tm.detector_model_wrapper(timestep=tres,
                              initial_state=np.array([0.,0,0,0,0]),
                              external_radon_conc=rn_conc,
                              internal_airt_history=airt,
                              parameters=parameters,
                              interpolation_mode=0,
                              return_full_state=False)

    counts_with_delay = df_model.Acc_counts.diff()

    # 750L detector
    parameters.update(util.standard_parameters)
    df_model=tm.detector_model_wrapper(timestep=tres,
                              initial_state=np.array([0.,0,0,0,0]),
                              external_radon_conc=rn_conc,
                              internal_airt_history=airt,
                              parameters=parameters,
                              interpolation_mode=0,
                              return_full_state=False)

    counts_750L = df_model.Acc_counts.diff()

    df_results = pd.DataFrame(data=dict(fifteenhundred_litre=counts_no_delay,
                                    fifteenhundred_litre_nodelay=counts_with_delay,
                                    sevenfifty_litre=counts_750L))


    plt.show()
