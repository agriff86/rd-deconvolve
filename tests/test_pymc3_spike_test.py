#%%
# NOTE: this is not currently working because the experimental data is from
# injection upstream of the external delay volume, whereas the detector model
# has injection downstream of the external delay

# this model can be *really* big because of the one-minute time resolution

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


sp = {
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


spike_params = {
    # note: the 1.3 is a fudge factor
    "cal_source_strength": 100e3 * 24 * 60 * 1.3,
    "cal_begin": 13300,
    "cal_begin_sigma": 60 * 5,
    "cal_duration": 60,
    "cal_duration_sigma": 0,
    "cal_injection_upstream_of_delay": True,
}

sp.update(spike_params)


def load_radon(fname):
    """load raw radon data in csv format"""

    def parse_hhmm_string(s):
        return datetime.datetime.strptime(s, "%H:%M").time()

    df = pd.read_csv(fname)
    df.columns = [itm.strip().lower() for itm in df.columns]
    df["time"] = df.time.apply(parse_hhmm_string)
    df.columns = [itm.strip().lower() for itm in df.columns]
    time = [
        datetime.datetime.combine(
            datetime.date(
                int(itm[1]["year"]), int(itm[1]["month"]), int(itm[1]["dom"])
            ),
            itm[1]["time"],
        )
        for itm in df.iterrows()
    ]

    df.index = time
    return df


def get_raw_data():
    PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA_DIR = os.path.join(PROJECT_DIR, "test_data")
    fnames = [
        os.path.join(RAW_DATA_DIR, "data-controlled-test-2/T1Mar15e.CSV"),
        os.path.join(RAW_DATA_DIR, "data-controlled-test-2/T1Apr15e.CSV"),
    ]
    dfobs = [load_radon(itm) for itm in fnames]
    dfobs = pd.concat(dfobs)
    return dfobs


def load_test_data(spike_idx=1):
    """
    Load data in right format for test
    """
    df = get_raw_data()

    dfl_spikes = [df.iloc[ii * 24 * 60 : (ii + 1) * 24 * 60] for ii in range(10)]
    df = dfl_spikes[spike_idx]

    counts = df.lld - df.uld
    time = (df.index - df.index[0]).total_seconds()
    time += time[1]

    return time.values, counts.values, df


#%%
do_spike_test_case = True
spike_idx = 1
if do_spike_test_case:
    print("TESTING BY FITTING TO SPIKE TEST")
    figure_manager = FigureManager("./figs", "test_with_spike")
    time_obs, counts_obs, df = load_test_data(spike_idx)

    background_estimate = df.lld.values[: 3 * 60].mean() / 60.0
    # trim size of data set
    df = df.iloc[3 * 60 + 30 : 7 * 60]
    sp["cal_begin"] -= (3 * 60 + 30) * 60

    # df = df.resample('5min').sum()

    counts_obs = df.lld - df.uld
    time_obs = (df.index - df.index[0]).total_seconds().values
    time_obs += time_obs[1]

    sp["background_rate"] = background_estimate
    known_radon = np.zeros_like(counts_obs, dtype=float)

    # prior
    used_params, df_sim = convolve_radon_timeseries_numpy(
        time_obs, known_radon, radon_0=0.001, detector_params=sp
    )
    counts_est = df_sim.rate.values

    # perturb params, for comparison
    sp2 = {}
    sp2.update(sp)
    # sp2['rs'] = 0.1
    sp2["Q_external"] = sp["Q_external"] * 0.5
    used_params, df_sim = convolve_radon_timeseries_numpy(
        time_obs, known_radon, radon_0=0.001, detector_params=sp2
    )
    counts_est2 = df_sim.rate.values
    sp3 = {}
    sp3.update(sp)
    # sp3['rs'] = 2.0
    sp3["Q_external"] = sp["Q_external"] * 2
    used_params, df_sim = convolve_radon_timeseries_numpy(
        time_obs, known_radon, radon_0=0.001, detector_params=sp3
    )
    counts_est3 = df_sim.rate.values

    fig, ax = plt.subplots()
    ax.plot(time_obs, counts_obs, label="obs")
    ax.plot(time_obs, counts_est, label="prior")
    ax.plot(time_obs, counts_est2, label="prior, Q_external * 0.5")
    ax.plot(time_obs, counts_est3, label="prior, Q_external * 2")

    ax.legend()

    fit_result = fit_model_to_obs(
        time_obs,
        counts_obs,
        detector_params=sp,
        known_radon=known_radon,
        Nsamples=1000,
        figure_manager=figure_manager,
    )
    trace_obs = fit_result["trace"]
    ds_trace_obs = trace_as_xarray(df.index.values, trace_obs)
    ds_summary_obs = stats_from_xarray(ds_trace_obs)

    result_plot(ds_summary_obs)

    ds_trace_obs.to_netcdf(f"trace-spike-test-{spike_idx}.nc")


plt.show()


# %%
