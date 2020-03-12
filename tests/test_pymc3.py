#%%
import numpy as np
import pymc3 as pm
from scipy.stats import distributions
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime

import sys

sys.path.append(".")
sys.path.append("..")

from rddeconv.forward_model import convolve_radon_timeseries_numpy
from rddeconv.pymc3_deconvolve import fit_model_to_obs
from rddeconv.pymc3_deconvolve import trace_as_xarray
from rddeconv.pymc3_deconvolve import stats_from_xarray
from rddeconv.pymc3_deconvolve import result_plot
from rddeconv.deconvolve import FigureManager
import rddeconv

test_fit_calibration = True
real_data_use_cal = True
real_data_fit_cal = True


sp = {
    "Q": 0.0122,
    "rs": 0.95,
    "lamp": 0.005555555555555556,
    "eff": 0.17815,
    "Q_external": 0.0006666666666666666,
    "V_delay": 0.2,
    "V_tank": 0.7,
    "t_delay": 60.0,
    "expected_change_std": 1.25,
    "total_efficiency": 0.154,
    "total_efficiency_frac_error": 0.025,
    "cal_source_strength": 20e3,
    "cal_begin": 3600 * 5,
    "cal_begin_sigma": 0,
    "cal_duration": 3600 * 10,
    "cal_duration_sigma": 0,
    "recoil_prob": 0.025,
    # "background_count_rate": 1/60.0
}

sp_goulburn = {
    "Q": 0.0122,
    "rs": 0.9,
    "lamp": 0.005555555555555556,
    "eff": 0.14539,
    "Q_external": 0.0006666666666666666,
    "V_delay": 0.2,
    "V_tank": 0.7,
    "expected_change_std": 1.25,
    "total_efficiency": 0.154,
}

if real_data_fit_cal:
    sp_goulburn.update(
        {
            "expected_change_std": 1.1,
            "cal_source_strength": 100e3,
            "cal_begin": 3600 * 4,
            "cal_begin_sigma": 3600,
            "cal_duration": 3600 * 24,
            "cal_duration_sigma": 3600,
        }
    )

if test_fit_calibration:
    sp["expected_change_std"] = 1.1
else:
    sp["cal_source_strength"] = 0.0

if real_data_use_cal and not real_data_fit_cal:
    # set to zero to avoid ringing in calibration curve
    sp_goulburn["V_delay"] = 0.0
    del sp_goulburn["expected_change_std"]


def build_test_data(sp):
    tmax = 48 * 3600
    dt = 60 * 15
    dt = 60 * 30
    t = np.arange(dt, tmax + 1, dt)
    #
    radon = np.ones(t.shape) * 10
    if not test_fit_calibration:
        radon[(t > 3600 * 5) & (t <= 3600 * (5 + 3))] += 100
    radon_0 = 10
    params, df = convolve_radon_timeseries_numpy(t, radon, radon_0, detector_params=sp)

    return t, radon, df.rate.values


def load_test_data(with_cal=True):
    """
    Load data from real observations
    """
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA_DIR = os.path.join(
        PROJECT_DIR, "examples", "goulburn-deconvolution", "raw-data"
    )
    fname_raw_data = os.path.join(
        RAW_DATA_DIR, "Goulburn_Nov_2011_Internal_DB_v01_raw.csv"
    )

    df = rddeconv.load_standard_csv(fname_raw_data)
    # drop problematic first value (lld=1)
    df = df.dropna(subset=["lld"])
    df["lld"].iloc[0] = np.NaN
    df = df.dropna(subset=["lld"])

    # drop the bad data at the end of the record
    df = df.loc[: datetime.datetime(2011, 11, 10, 12)]

    if with_cal:
        # there's a calibration at the start of the data set
        df = df.head(70)
    else:
        # near the end, we're just working with a single night
        df = df.tail(48)
    counts = df.lld - df.uld
    time = (df.index - df.index[0]).total_seconds()
    time += time[1]
    return time.values, counts.values, df


#%%
do_real_data_case = False
if do_real_data_case:
    print("TESTING WITH REAL OBSERVATIONS")
    figure_manager = FigureManager("./figs", "test_with_obs")
    time_obs, counts_obs, df = load_test_data(with_cal=real_data_use_cal)
    fit_result = fit_model_to_obs(
        time_obs,
        counts_obs,
        detector_params=sp_goulburn,
        Nsamples=1000,
        figure_manager=figure_manager,
    )
    trace_obs = fit_result["trace"]
    ds_trace_obs = trace_as_xarray(df.index.values, trace_obs)
    ds_summary_obs = stats_from_xarray(ds_trace_obs)

    result_plot(ds_summary_obs)


#%%
do_synthetic_data_case = True
if do_synthetic_data_case:
    print("TESTING WITH SIMULATED OBSERVATIONS")
    figure_manager = FigureManager("./figs", "test_with_sim")
    time, true_radon, E_counts = build_test_data(sp)
    counts = distributions.poisson.rvs(E_counts)

    fit_result = fit_model_to_obs(
        time, counts, detector_params=sp, Nsamples=2000, figure_manager=figure_manager
    )
    trace = fit_result["trace"]

    ds_trace = trace_as_xarray(time, trace)
    ds_summary = stats_from_xarray(ds_trace)

    fig, ax = result_plot(ds_summary)

    # TODO: more plots
    # arviz.plot_pair(ds_trace, var_names=[''])

    summary = pm.summary(trace)
    fig, ax = plt.subplots()
    summary[["mean", "hpd_3%", "hpd_97%"]].plot(ax=ax)

    fig, ax = plt.subplots()
    t = ds_summary.time
    ax.plot(t, ds_summary.mcmc_mean.values, label="mean")
    ax.fill_between(
        t, ds_summary.mcmc_16pc, ds_summary.mcmc_84pc, alpha=0.5, label="HPD "
    )


plt.show()
