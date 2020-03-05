"""
Forward model of the two-filter radon detector
"""


from . import generated_functions_with_cse as gf

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# import rddeconv.util as rdutil
# rdutil.standard_parameters

import theano.tensor as tt
import theano

# import rddeconv.theoretical_model as tm

lamr = 2.100140526711101e-06

def calc_eff_and_recoil_prob(
    Q, rs, lamp, Q_external, V_delay, V_tank, total_efficiency, **kwargs
):
    """Calculate efficiency and recoil probability assuming that they are 
        linked (as described in paper) and with efficiency set by the net
        detector efficiency
    
    Arguments:
        Q {float} -- internal flow rate (m3/s)
        rs {float} -- screen efficiency (0-1)
        lamp {float} -- plateout constant (/s)
        Q_external {float} -- external flow rate (m3/s)
        V_delay {float} -- delay volume
        V_tank {float} -- tank volume (m3)
        total_efficiency {float} -- efficiency of the entire system
    
    Returns:
        (float,float) -- alpha detection efficiency (eff), recoil probability
    """
    recoil_prob = 0.5 * (1 - rs)
    eff = 1.0
    ssc = gf.steady_state_count_rate(rs, recoil_prob, V_tank, Q, lamp, eff)
    eff = eff * total_efficiency / ssc
    return eff, recoil_prob


def convolve_radon_timeseries_numpy(t, radon, radon_0, detector_params):
    # we want to zero out response after a threshold time
    # (avoids round-off error related problems, and may reduce computation)
    threshold_sec = 60000
    threshold_idx = np.argmax(t > threshold_sec)

    all_keys = [
        "Q",
        "rs",
        "lamp",
        "eff",
        "Q_external",
        "V_delay",
        "V_tank",
        "t_delay",
        "interpolation_mode",
        "expected_change_std",
        "total_efficiency",
        "total_efficiency_frac_error",
        "transform_radon_timeseries",
        "cal_source_strength",
        "cal_begin",
        "cal_duration",
        "recoil_prob",
    ]
    all_keys = [
        "Q",
        "rs",
        "lamp",
        #     'eff',
        "Q_external",
        "V_delay",
        "V_tank",
        # 't_delay',
        #     'interpolation_mode',
        #     'expected_change_std',
        "total_efficiency",
        #     'total_efficiency_frac_error',
        #     'transform_radon_timeseries',
        "cal_source_strength",
        "cal_begin",
        "cal_duration"
        #     'recoil_prob'
    ]

    params = {k: detector_params[k] for k in all_keys}
    # derive eff and recoil prob from other parameters
    params["eff"], params["recoil_prob"] = calc_eff_and_recoil_prob(**params)
    #
    N_af, N_bf = gf.calc_na_nb_factors(**params)
    #
    # ... initial radon concentration, slow decay with time
    #
    a0, b0, c0 = gf.num_filter_atoms_steady_state(**params)
    a0 *= radon_0
    b0 *= radon_0
    c0 *= radon_0
    N_d0 = radon_0
    N_r0 = N_d0
    tc1 = gf.tc_integral_filter_activity(
        t=t, N_d0=N_d0, N_r0=N_r0, a0=a0, b0=b0, c0=c0, N_af=N_af, N_bf=N_bf, **params
    )
    tc1[threshold_idx:] = tc1[threshold_idx]
    rate1 = tc1.copy()
    rate1[1:] = np.diff(tc1)
    #
    # ... response to radon signal
    #
    delta_t = t[1] - t[0]
    R = gf.tc_integral_square_wave(
        t=t, t0=0, delta_t=delta_t, N_af=N_af, N_bf=N_bf, N_e=1.0, **params
    )
    # R = gf.tc_integral_delta(t=t, t0=0, delta_t=delta_t, N_af=N_af, N_bf=N_bf, Ncal=1.0, **params)
    R[threshold_idx:] = R[threshold_idx]
    tc2 = np.zeros(tc1.shape)
    N = len(radon)
    # write the convolution as a loop, for copying into stan
    use_loop = False
    if use_loop:
        for ii in range(N):
            tc2[ii:] += R[0 : N - ii] * radon[ii]
    else:
        # alternative to the loop - faster, but the loop
        # isn't a major bottleneck
        tc2 = np.convolve(radon, R)[:N]
    rate2 = tc2.copy()
    rate2[1:] = np.diff(tc2)

    #
    # ... response to calibration
    #
    # TODO: calc Ncal from source strength
    Ncal = params['cal_source_strength'] / params['Q_external'] * lamr
    tc3 = gf.tc_integral_calibration(t=t, N_af=N_af, N_bf=N_bf, Ncal=Ncal, **params)
    rate3 = tc3.copy()
    rate3[1:] = np.diff(tc3)

    # clip to zero where necessary
    rate3[
        (t - params["cal_begin"] < 0)
        | (t > params["cal_begin"] + params["cal_duration"] + threshold_sec)
    ] = 0.0

    tc = tc1 + tc2 + tc3
    rate = rate1 + rate2 + rate3

    df = pd.DataFrame(
        data={
            "tc1": tc1,
            "rate1": rate1,
            "tc2": tc2,
            "rate2": rate2,
            "tc3": tc3,
            "rate3": rate3,
            "R": R,
            "tc": tc,
            "rate": rate,
            "radon": radon,
        },
        index=t,
    )

    return params, df


def theano_diff(x):
    """Specialised version of diff to use with theano
    
    Returns y = [x0, x1-x0, x2-x1, ...]

    Arguments:
        x {tt.vector} -- theano vector to diff
    """
    diff_expr0 = x[1:] - x[:-1]
    diff_expr = tt.concatenate([x[0].reshape((1,)), diff_expr0])
    return diff_expr


#%%
def convolve_radon_timeseries(
    t, radon, radon_0, Npts, delta_t, detector_params, simulate_calibration=False
):
    # we want to zero out response after a threshold time
    # (avoids round-off error related problems, and may reduce computation)
    do_threshold = True
    threshold_sec = 3600 * 7  # 60000
    threshold_idx = int(threshold_sec / delta_t)

    if not do_threshold:
        threshold_idx = Npts

    tt.specify_shape(radon, (Npts,))

    required_keys = [
        "Q",
        "rs",
        "lamp",
        #     'eff',
        "Q_external",
        "V_delay",
        "V_tank",
        # 't_delay',
        #     'interpolation_mode',
        #     'expected_change_std',
        "total_efficiency",
        #     'total_efficiency_frac_error',
        #     'transform_radon_timeseries',
        #     'recoil_prob'
    ]
    cal_keys = [
        "cal_source_strength",
        "cal_begin",
        "cal_duration"

    ]

    params = {k: detector_params[k] for k in required_keys}
    if simulate_calibration:
        for k in cal_keys:
            params[k] = detector_params[k]

    # derive eff and recoil prob from other parameters
    params["eff"], params["recoil_prob"] = calc_eff_and_recoil_prob(**params)
    #
    N_af, N_bf = gf.calc_na_nb_factors(**params)
    # # delay time
    # t = t - params['t_delay']
    #
    # ... initial radon concentration, slow decay with time
    #
    a0, b0, c0 = gf.num_filter_atoms_steady_state(**params)
    a0 *= radon_0
    b0 *= radon_0
    c0 *= radon_0
    N_d0 = radon_0
    N_r0 = N_d0
    tc1_short = gf.tc_integral_filter_activity(
        t=t[:threshold_idx],
        N_d0=N_d0,
        N_r0=N_r0,
        a0=a0,
        b0=b0,
        c0=c0,
        N_af=N_af,
        N_bf=N_bf,
        **params
    )
    rate1_short = theano_diff(tc1_short)
    rate1 = tt.zeros(shape=(Npts,))
    rate1 = tt.set_subtensor(rate1[:threshold_idx], rate1_short)

    #
    # ... response to radon signal
    #
    R_short = gf.tc_integral_square_wave(
        t=t[:threshold_idx],
        t0=0,
        delta_t=delta_t,
        N_af=N_af,
        N_bf=N_bf,
        N_e=1.0,
        **params
    )
    R_short_rate = theano_diff(R_short)
    # R = gf.tc_integral_delta(t=t, t0=0, delta_t=delta_t, N_af=N_af, N_bf=N_bf, Ncal=1.0, **params)
    R = tt.zeros(shape=(Npts,))
    R = tt.set_subtensor(R[:threshold_idx], R_short_rate)

    rate2 = tt.zeros((Npts,))
    # write the convolution as a loop, for copying into stan
    for ii in range(Npts):
        rate2 = tt.inc_subtensor(rate2[ii:], R[0 : Npts - ii] * radon[ii])

    rate = rate1 + rate2

    #
    # ... response to calibration
    #
    if simulate_calibration:
        # restrict computation to a short period around calibration
        Ncal = params['cal_source_strength'] / params['Q_external'] * lamr

        tc3 = gf.tc_integral_calibration(
            t=t, N_af=N_af, N_bf=N_bf, Ncal=Ncal, **params
        )
        rate3 = theano_diff(tc3)
        # sometimes the calculations fail long lag times - 
        # at these times the rate should be close to zero, so 
        # here we just set it to zero
        # rate3 = tt.set_subtensor(rate3[tt.isnan(rate3).nonzero()], 0.0)
        # --- No, trying a more numerically stable version of the generated
        # functions instead.
        rate = rate + rate3

    return rate
