"""
Test the forward model of the detector
"""

#%%

import theano

theano.config.mode = "FAST_COMPILE"

import os
import sys
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_DIR)


from rddeconv.forward_model import *

lambda_rn = 2.100140526711101e-06

# standard parameters
sp = {
    "Q": 0.0122,
    "rs": 0.95,
    "lamp": 0.005555555555555556,
    "eff": 0.17815,
    "Q_external": 0.0006666666666666666,
    "V_delay": 0.2,
    "V_tank": 0.7,
    "t_delay": 60.0,
    "interpolation_mode": 1,
    "expected_change_std": 1.1,
    "total_efficiency": 0.154,
    "total_efficiency_frac_error": 0.025,
    "transform_radon_timeseries": True,
    "cal_source_strength": 10
    / lambda_rn
    * 0.0006666666666666666,  # defined to give us 10 Bq/m3 during cal
    "cal_begin": 3600 * 4,
    "cal_duration": 3600 * 5,
    "recoil_prob": 0.025000000000000022,
}

# this config was giving me problems
sp2 = {
    "Q": 0.0122,
    "rs": 0.8499656524756104,
    "lamp": 0.005555555555555556,
    "eff": 0.14539,
    "Q_external": 0.0009244574392942429,
    "V_delay": 0.2,
    "V_tank": 0.75,
    "interpolation_mode": 1,
    "expected_change_std": 1.25,
    "total_efficiency": 0.154,
}

# %%
gf.steady_state_count_rate(**sp)

# "real" radon timeseries

# time, in seconds, 15 min intervals
tmax = 48 * 3600
dt = 60 * 15
t = np.arange(dt, tmax + 1, dt)
#
radon = np.ones(t.shape)
radon[(t > 3600 * 4) & (t <= 3600 * (5 + 4))] = 10 + 1
# %%
plt.plot(t / 3600, radon)

# %%
radon_0 = 1
params, df = convolve_radon_timeseries_numpy(t, radon, radon_0, detector_params=sp)
# %%
df[["rate", "rate1", "rate2", "rate3"]].plot()

# %%
df[["tc1", "tc2", "tc3"]].plot()


# %%
theano_var_names = [
    "Q",
    "rs",
    "lamp",
    #     'eff',
    "Q_external",
    "V_delay",
    "V_tank",
    #     't_delay',
    #     'interpolation_mode',
    #     'expected_change_std',
    "total_efficiency",
    #     'total_efficiency_frac_error',
    #     'transform_radon_timeseries',
    #     'recoil_prob'
]
# %%
sp_theano = {k: tt.dscalar(k) for k in theano_var_names}

# %%
sp_theano

# %%

# theano.config.optimizer = 'fast_compile'

print("testing theano function")

s_radon_0 = tt.dscalar("radon_0")
s_radon = tt.dvector("radon")
s_time = tt.constant(t)
Npts = len(t)
delta_t = t[1] - t[0]
s_expr = convolve_radon_timeseries(
    s_time, s_radon, s_radon_0, Npts, delta_t, detector_params=sp_theano
)


func_args = [sp_theano[k] for k in theano_var_names]
func_args = [s_radon, s_radon_0] + func_args
convolve_radon_timeseries_theano = theano.function(func_args, s_expr)

# %%
numeric_args = {k: sp[k] for k in theano_var_names}
numeric_args["radon"] = radon
numeric_args["radon_0"] = radon_0
counts_theano = convolve_radon_timeseries_theano(**numeric_args)

#%%

counts_theano = convolve_radon_timeseries_theano(**numeric_args)


#%%
plt.figure()
plt.plot(t, counts_theano)
plt.figure()

#
# ... test version of the model with explicit calibration
#

# define required symbolic variables
cal_variable_names = ["cal_source_strength", "cal_begin", "cal_duration"]
sp_theano_cal = {k: tt.dscalar(k) for k in cal_variable_names}
sp_theano_with_cal = {}
sp_theano_with_cal.update(sp_theano)
sp_theano_with_cal.update(sp_theano_cal)

# build symbolic graph
s_expr2 = convolve_radon_timeseries(
    s_time,
    s_radon,
    s_radon_0,
    Npts,
    delta_t,
    detector_params=sp_theano_with_cal,
    simulate_calibration=True,
)

# compile symbolic graph
func_args2 = [sp_theano_with_cal[k] for k in theano_var_names + cal_variable_names]
func_args2 = [s_radon, s_radon_0] + func_args2
convolve_radon_timeseries_theano2 = theano.function(func_args2, s_expr2)

# evaluate function on actual data
numeric_args2 = {k: sp[k] for k in theano_var_names + cal_variable_names}
radon_const = radon.copy()
radon_const[:] = radon_const[0]
numeric_args2["radon"] = radon_const
numeric_args2["radon_0"] = radon_0
counts_theano2 = convolve_radon_timeseries_theano2(**numeric_args2)

# numpy-based model with calibration
radon_cal = radon.copy()
radon_cal[:] = radon[0]
params, df = convolve_radon_timeseries_numpy(t, radon, radon_0, detector_params=sp)
counts_numpy = df.rate.values


# plot results from both runs
plt.figure()
plt.plot(t, counts_theano, label="model without explicit calibration")
plt.plot(t, counts_theano2, label="model with explicit calibration")
plt.plot(t, counts_numpy, label="numpy-based model")
plt.legend()

# check model output with random input


rel_counts = (counts_theano - counts_theano[-1]).sum() / (
    counts_theano2 - counts_theano2[-1]
).sum()
print(f"counts with radon timeseries / counts with explicit cal: {rel_counts}")

# experiment with radon_0
# evaluate function on actual data
fig, ax = plt.subplots()

for radon_0ii in [0, radon_0 / 2, radon_0, radon_0 * 2]:
    numeric_args2 = {k: sp[k] for k in theano_var_names + cal_variable_names}
    radon_const = radon.copy()
    radon_const[:] = radon_const[0]
    numeric_args2["radon"] = radon_const
    numeric_args2["radon_0"] = radon_0ii
    counts_theano2 = convolve_radon_timeseries_theano2(**numeric_args2)
    ax.plot(t, counts_theano2, label=f"radon0={radon_0ii}")

ax.legend()

plt.show()
