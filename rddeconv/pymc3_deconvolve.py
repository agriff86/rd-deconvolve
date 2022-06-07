"""
Deconvolution with pymc3 as MCMC sampler and forward model based on the
analytical solution of the detector model's governing equations
"""

# Set flags for theano
# ref: http://deeplearning.net/software/theano_versions/dev/faq.html#faster-theano-function-compilation
# ideas for compiling more quickly
# ref: https://github.com/Theano/Theano/issues/4494
import os

theano_flags = ["cycle_detection=fast,optimizer_excluding=inplace_elemwise_optimizer"]
# theano_flags = ["mode=FAST_COMPILE"]
os.environ["THEANO_FLAGS"] = ",".join(theano_flags)


# theano sometimes breaks recursion limits
# ref: https://github.com/Theano/Theano/issues/689
import sys

sys.setrecursionlimit(50000)

import numpy as np
import theano
import theano.tensor as tt
import datetime
import xarray as xr
import arviz

# theano.compile.DebugMode = True
# theano.config.mode = 'FAST_COMPILE'

from .forward_model import convolve_radon_timeseries, convolve_radon_timeseries_numpy
from scipy.stats import distributions

import pymc3 as pm

import matplotlib as mpl
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)


# timing helper
from contextlib import contextmanager
import time as timelib

lamrn = 2.100140526711101e-06

class DeconvolutionError(Exception):
    pass


@contextmanager
def timing(description: str) -> None:
    start = timelib.time()
    yield
    elapsed_time = timelib.time() - start
    logger.info(f"{description} took {round(elapsed_time, 1)} seconds")


def fit_model_to_obs(
    time: np.ndarray,
    counts: np.ndarray,
    detector_params: dict,
    known_radon: np.ndarray = None,
    Nsamples: int = 1000,
    Nchains: int = 4,
    njobs: int = 4,
    figure_manager=None,
):
    if njobs is None:
        njobs = 1
    with timing("Constructing model"):
        model = construct_model(
            time=time,
            counts=counts,
            detector_params=detector_params,
            known_radon=known_radon,
        )
    try:
        with timing(f"MCMC diagnostics"):
            logger.debug(f"Test point logp:\n{model.check_test_point()}")
            with model:
                map_estimate = pm.find_MAP(progressbar=(njobs > 1))
            logger.debug("MAP estimate:")
            for k in map_estimate.keys():
                if not k in ["logradon", "radon", "E_simulated_counts"]:
                    if k in detector_params:
                        logger.debug(
                            f"{k}: {map_estimate[k]} (normalised by prior: {map_estimate[k]/detector_params[k]})"
                        )
                    else:
                        logger.debug(f"{k}: {map_estimate[k]}")
                if k == "radon":
                    radon_str = " ".join(
                        [f"{float(itm):.3}" for itm in map_estimate[k]]
                    )
                    if len(radon_str) > 80:
                        radon_str = radon_str[:80] + "..."
                    logger.debug(f"{k}: {radon_str}")
            # This isn't doing what I want it to - disable for now
            # logger.debug(
            #    f"MAP estimate logp:\n{model.check_test_point(test_point=map_estimate)}"
            # )
            if True:
                # TODO: produce this plot and save to a file
                # perhaps via a 'figure manager' class which
                # knows where to save the figure, and keeps track
                # of a figure counter
                # plot MAP estimate
                fig, ax = plt.subplots()
                delta_t = time[1] - time[0]

                ax.plot(
                    time / 3600.0,
                    counts / delta_t / detector_params["total_efficiency"],
                    label="scaled counts",
                )
                ax.plot(
                    time / 3600.0,
                    map_estimate["E_simulated_counts"]
                    / delta_t
                    / detector_params["total_efficiency"],
                    label="scaled counts (simulated)",
                )
                ax.plot(time / 3600, map_estimate["radon"], label="reconstructed radon")
                ax.legend()

                if figure_manager is not None:
                    figure_manager.save_figure(fig, "map")

                # plt.show()

        with timing(f"MCMC sampling {Nchains} chains of {Nsamples} points each"):
            with model:
                # assume we're running under dask if njobs == 1 and we need to
                # hide the progressbar
                progressbar = njobs > 1
                # default target_accept is 0.8.  Increase (if needed) to make
                # sampling more robust
                # default number of tuning steps is tune=500
                ntune = min(500, Nsamples)
                trace = pm.sample(
                    Nsamples,
                    chains=Nchains,
                    progressbar=progressbar,
                    cores=njobs,
                    # target_accept=0.9,
                    tune=ntune,
                )

    except pm.parallel_sampling.ParallelSamplingError as pm_exception:
        logger.exception(pm_exception)
        logger.error("Checking test point, initial logp, and logp gradient evaluation")
        logger.error(f"Test point: {model.test_point}")
        logger.error(f"Test point logp: {model.check_test_point()}")
        f = model.logp_dlogp_function()
        f.set_extra_values({})
        y = f(f.dict_to_array(model.test_point))
        logger.error(f"logp gradient: {y}")
        raise
    ret = {}
    ret["trace"] = trace
    ret["map_estimate"] = map_estimate
    return ret


def trace_as_xarray(index_time, trace, vn="deconvolved_radon"):
    """Convert pymc3 trace to xarray, dropping the less-useful bits
    
    Parameters
    ----------
    index_time : [type]
        Data to use for the time index
    trace : [type]
        pymc3 trace from inference
    """
    ds = arviz.from_pymc3(trace).posterior
    # cleanup
    ds = ds.rename_dims({"radon_dim_0": "time"})
    ds = ds.rename_vars({"radon_dim_0": "time"})
    vars_to_drop = [
        "logradon_dim_0",
        "logradon",
    #    "E_simulated_counts",
        "E_simulated_counts_dim_0",
    ]
    for k in vars_to_drop:
        if k in ds:
            ds = ds.drop(k)
    # create time variable - possibly with placeholder values
    if index_time is None:
        Ntime = len(ds.time)
        dt = datetime.timedelta(minutes=30)
        index_time = [datetime.datetime(1900, 1, 1) + ii * dt for ii in range(Ntime)]
    else:
        index_time = np.array(index_time)

    timevar = xr.DataArray(index_time, dims=["time"])
    timevar.attrs["long_name"] = "Time at end of averaging period"
    ds["time"] = timevar

    return ds


def stats_from_xarray(ds):
    # analagous to a +/- one sigma interval
    # variable name root
    k = 'radon_deconv'
    summary_data = []
    mid = ds.stack(z=("chain", "draw")).radon.mean(dim="z")
    mid.name = k
    summary_data.append(mid)
    sd = ds.stack(z=("chain", "draw")).radon.std(dim="z")
    sd.name = k+'_sd'
    summary_data.append(sd)

    low_16pc, high_84pc = arviz.hpd(
        ds.stack(z=("chain", "draw")).radon.T, credible_interval=0.68
    ).T
    low_16pc = xr.DataArray(low_16pc, dims=("time",))
    low_16pc.name = k+"_16pc"
    high_84pc = xr.DataArray(high_84pc, dims=("time",))
    high_84pc.name = k+"_84pc"
    summary_data.append(low_16pc)
    summary_data.append(high_84pc)

    low_3pc, high_97pc = arviz.hpd(
        ds.stack(z=("chain", "draw")).radon.T, credible_interval=0.94
    ).T
    low_3pc = xr.DataArray(low_3pc, dims=("time",))
    low_3pc.name = k+"_3pc"
    high_97pc = xr.DataArray(high_97pc, dims=("time",))
    high_97pc.name = k+"_97pc"
    summary_data.append(low_3pc)
    summary_data.append(high_97pc)

    ds_ret = xr.merge(summary_data)
    try:
        ds_ret['overlap_flag'] = ds['overlap_flag']
    except KeyError:
        pass

    return ds_ret


def result_plot(ds, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    t = ds.time
    ax.plot(t, ds.mcmc_mean.values, label="mean")
    ax.fill_between(t, ds.mcmc_16pc, ds.mcmc_84pc, alpha=0.3, label="HPD 68%")
    ax.fill_between(t, ds.mcmc_3pc, ds.mcmc_97pc, alpha=0.6, label="HPD 94%")
    return fig, ax


def summarise_result(index_time, trace, vn="deconvolved_radon"):
    pm_summary = pm.summary(trace)
    # TODO: extract only relevant information
    radon_summary = pm.summary(trace, var_names=["radon"])
    cols_to_keep = ["mean", "sd", "hpd_3%", "hpd_97%", "ess_mean", "r_hat"]
    rename_dict = {}
    for k in cols_to_keep:
        rename_dict[k] = f"{k}_{vn}"
    rename_dict["mean"] = vn
    radon_trace = trace["radon"].copy()

    radon_summary = radon_summary[cols_to_keep].rename(columns=rename_dict)
    radon_summary.index = index_time
    radon_summary.index.name = "time"
    radon_summary["interval_end_time"] = radon_summary.index.values
    dt = radon_summary["interval_end_time"].diff().values[1]
    radon_summary["interval_start_time"] = radon_summary["interval_end_time"] - dt

    # 1d parameters that are not transformed
    inference_vars = [
        k for k in trace.varnames if len(trace[k].shape) == 1 and not k.endswith("__")
    ]
    print(inference_vars)

    inference_vars_summary = pm.summary(trace, var_names=inference_vars)
    inference_vars_summary = inference_vars_summary[cols_to_keep].rename(
        columns=rename_dict
    )

    return pm_summary


def construct_model(
    time: np.ndarray,
    counts: np.ndarray,
    detector_params: dict,
    known_radon: np.ndarray = None,
) -> pm.Model:

    # shorter variable name, as we use this a lot
    sp = detector_params
    # derived variables and flags
    Npts = len(counts)
    # note - theano doesn't handle boolean masks, hence the .nonzero()
    # ref: https://stackoverflow.com/questions/37425401/theano-tensor-slicing-how-to-use-boolean-to-slice
    flag_valid = np.isfinite(counts).nonzero()
    Npts_valid = len(counts[flag_valid])
    delta_t = time[1] - time[0]

    if time[0] == 0:
        logger.error(
            "Time should not start at zero because timestamps should label the end of each counting interval"
        )

    radon_conc_known = known_radon is not None
    smooth_radon_timeseries = "expected_change_std" in sp
    if not "cal_source_strength" in sp and not "cal_duration" in sp:
        simulate_calibration = False
    else:
        simulate_calibration = (sp["cal_source_strength"] > 0) and (
            sp["cal_duration"] > 0
        )

    if radon_conc_known:
        logger.info(
            "Ambient radon concentration is known - configuring model to fit other parameters"
        )
    else:
        logger.info("Configuring model to perform deconvolution")
        logger.info(
            f"Smoothing radon concentration timeseries: {smooth_radon_timeseries}"
        )

    cal_injection_upstream_of_delay = detector_params.get(
        "cal_injection_upstream_of_delay", False
    )
    if simulate_calibration:
        inj_loc = ["downstream", "upstream"][int(cal_injection_upstream_of_delay)]
        logger.info(
            f"Simulating calibration in model with injection {inj_loc} of external delay volume"
        )

    if not np.allclose(np.diff(time), delta_t):
        raise ValueError("time must have uniform spacing")

    try:
        background_count_rate = sp["background_rate"]
    except KeyError:
        logger.warning("Background count rate not specified, assuming zero")
        logger.warning("(Set 'background_rate' in cps to include background counts.)")
        background_count_rate = 0.0

    logger.info(
        f"Timeseries is {Npts} points long with {np.isnan(counts).sum()} NaN values"
    )
    if Npts - np.isnan(counts).sum() < 1:
        raise DeconvolutionError('No valid data for deconvolution')

    logger.debug(f"Timestep is {delta_t/60} minutes")
    # Priors --
    # we'll express these with a lognormal distribution
    # but the sigma gets provided in terms of sigma/mu
    # for example, a 10% error (1-sigma range between mu/1.1 to mu*1.1)
    # implies a shape parameter of log(1.1)

    total_efficiency = sp["total_efficiency"]
    # this is not presently in use
    total_efficiency_frac_error = 1.05

    Q_external_frac_error = 1.025
    rs = sp["rs"]
    rs_frac_error = 1.025
    rn0_guess = counts[np.isfinite(counts)][0] / sp["total_efficiency"] / delta_t
    # place a floor value on rn0
    rn0_guess = max(rn0_guess, 1 / sp["total_efficiency"] / delta_t)
    rn0_frac_error = 1.25
    # radon - reference value (for scaling initial guess)
    rn_ref = (counts[np.isfinite(counts)].mean() / delta_t - background_count_rate) / sp["total_efficiency"]

    # are we simulating a calibration?
    if simulate_calibration:
        # if we're calibrating, we don't know total eff very well
        total_efficiency_frac_error = 2.0
        # adjust rn_ref for calibration injection
        cal_increment = sp["cal_source_strength"] / sp["Q_external"] * lamrn
        rn_ref -= cal_increment * sp["cal_duration"] / (time[-1] - time[0])

    # guard against rn_ref < 0
    if rn_ref <= 0:
        # don't subtract off the background
        rn_ref = (counts[np.isfinite(counts)].mean() / delta_t ) / sp[
            "total_efficiency"
        ]

    logger.debug(
        f"Radon reference value is {rn_ref}, radon at t=0 (prior estimate) is {rn0_guess}"
    )

    radon_detector_model = pm.Model()

    with radon_detector_model:
        if radon_conc_known:
            radon = pm.Constant("radon", known_radon.astype(float), shape=(Npts,))
        else:
            if smooth_radon_timeseries:
                # radon timeseries (with smoothing prior)
                logradon = pm.GaussianRandomWalk(
                    "logradon",
                    shape=(Npts,),
                    mu=0,
                    sigma=np.log(sp["expected_change_std"]),
                )
                radon = pm.Deterministic("radon", rn_ref * pm.math.exp(logradon))
            else:
                # otherwise, just put radon somewhere close to its mean value
                # fix sigma and then use the relationship that, for
                # lognormal, E[x] = exp(mu + 1/2 sigma**2)
                # => mu = log(E[x]) - 1/2 sigma**2
                sigma_rn = np.log(2.0)
                mu_rn = np.log(rn_ref) - 0.5 * sigma_rn ** 2
                radon = pm.Lognormal("radon", mu=mu_rn, sigma=sigma_rn, shape=(Npts,))

        # detector parameter priors
        # note: assuming that sigma << mu for Lognormal distributions
        #       (alternative could be to use truncated normal)
        rs = pm.distributions.Lognormal(
            "rs", mu=np.log(rs), sigma=np.log(rs_frac_error)
        )
        rn0 = pm.distributions.Lognormal(
            "rn0",
            mu=np.log(rn0_guess) - 0.5 * np.log(rn0_frac_error) ** 2,
            sigma=np.log(rn0_frac_error),
        )

        if Q_external_frac_error > 0:
            Q_external = pm.distributions.Lognormal(
                "Q_external",
                mu=np.log(sp["Q_external"]),
                sigma=np.log(Q_external_frac_error),
            )
        else:
            # Q_external is a constant
            Q_external = sp["Q_external"]

        # params taken from "sp" take fixed values
        detector_params = {
            "Q": sp["Q"],
            "rs": rs,
            "lamp": sp["lamp"],
            "Q_external": Q_external,
            "V_delay": sp["V_delay"],
            "V_tank": sp["V_tank"],
            "total_efficiency": tt.as_tensor_variable(sp["total_efficiency"]),
            "num_delay_volumes": sp["num_delay_volumes"],
        }

        # if we are fitting to a calibration then the total efficiency is unknown
        if simulate_calibration:
            total_efficiency = pm.distributions.Lognormal(
                "total_efficiency",
                mu=np.log(sp["total_efficiency"]),
                sigma=np.log(total_efficiency_frac_error),
            )
            # RVs for cal
            if sp["cal_begin_sigma"] > 0:
                cal_begin = pm.Bound(pm.Normal, lower=0, upper=max(time))(
                    "cal_begin", mu=sp["cal_begin"], sigma=sp["cal_begin_sigma"]
                )
            else:
                cal_begin = tt.as_tensor_variable(sp["cal_begin"])
            if sp["cal_duration_sigma"] > 0:
                cal_duration = pm.Bound(pm.Normal, lower=0)(
                    "cal_duration",
                    mu=sp["cal_duration"],
                    sigma=sp["cal_duration_sigma"],
                )
            else:
                cal_duration = tt.as_tensor_variable(sp["cal_duration"])
            detector_params["cal_begin"] = cal_begin
            detector_params["cal_duration"] = cal_duration
            detector_params["total_efficiency"] = total_efficiency
            detector_params["cal_source_strength"] = sp["cal_source_strength"]

        # expected value of "simulated counts"
        E_simulated_counts = pm.Deterministic(
            "E_simulated_counts",
            convolve_radon_timeseries(
                t=time,
                radon=radon,
                radon_0=rn0,
                Npts=Npts,
                delta_t=delta_t,
                detector_params=detector_params,
                simulate_calibration=simulate_calibration,
                cal_injection_upstream_of_delay=cal_injection_upstream_of_delay,
            )
            + background_count_rate * delta_t,
        )
        pm.distributions.Poisson(
            "counts", mu=E_simulated_counts[flag_valid], observed=counts[flag_valid], shape=(Npts_valid,)
        )

    return radon_detector_model
