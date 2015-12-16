#!/usr/bin/env python
# coding: utf-8

"""
Parameterised model of the 750l radon detector based on W&Z's 1996 paper
"""


from __future__ import (absolute_import, division,
                        print_function)


import glob
import datetime
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.stats import poisson, norm, lognorm

import emcee


import fast_model

# constants for radon decay and daughers
radon_chain_half_life = np.array([3.82*24*3600, #Rn-222 (3.82 d)
                               3.05*60,      #Po-218 (3.05 min)
                               26.8*60,      #Pb-214 (26.8 min)
                               19.9*60       #Bi-214 (19.9 min)
                               ])
radon_chain_num_alpha = np.array([1, 1, 0, 1])
radon_chain_name = [ 'Rn-222', 'Po-218', 'Pb-214', 'Bi-214']
radon_chain_lambda = np.log(2.0)/radon_chain_half_life

lamrn = radon_chain_lambda[0]
lama = radon_chain_lambda[1]
lamb = radon_chain_lambda[2]
lamc = radon_chain_lambda[3]

# number of state variables in the detector model
N_state = 5

def tank_concentration_rate_of_change(Y, t, lamp, Nrn):
    """
    Rate of change of species as they travel through the tank.

    Assumes plug flow.  Implements Eq. A1--A3 in W&Z 1996

    Parameters
    ----------
    Y : np.array
        state vector: Na, Nb, Nc

    TODO:

    Returns
    -------
    dYdt : np.array
        rate of change of state vector
    """
    #unpack state vector
    Na, Nb, Nc = Y
    dNadt = Nrn*lamrn - Na*(lama+lamp)
    dNbdt = Na*lama - Nb*(lamb+lamp)
    dNcdt = Nb*lamb - Nc*(lamc+lamp)
    return np.array([dNadt, dNbdt, dNcdt])


def calc_NaNbNc_num(Q, V_tank, lamp, Nrn):
    Y0 = np.zeros(3)
    tt = V_tank/Q
    t = np.linspace(0,tt,500)
    parameters = lamp, Nrn
    soln = odeint(tank_concentration_rate_of_change, Y0, t, args=parameters)
    return t, soln  #for testing


from numba import jit
from numpy import exp
@jit
def calc_NaNbNc(t, Nrn, lamp):
    """
    Compute concentrations of radon daughters at time t

    Assumes plug flow.  Based on analytical solution of Eq. A1--A3 in W&Z 1996

    Parameters
    ----------
    t : scalar or ndarray
        time in seconds

    TODO:

    Returns
    -------
    Na,Nb,Nc : scalars or ndarrays
        Number oncentration of radon-222 daughters.  Suffixes a,b,c mean:
        218Po, 214Pb, 214Bi.
    """
    # these expressions were generated by sage
    Na = Nrn*lamrn/(lama+lamp)-Nrn*lamrn*exp(-t*(lama+lamp))/(lama+lamp)

    Nb = (Nrn*lama*lamrn/(lama*lamb+lamp**2+lamp*(lama+lamb))-
            Nrn*lama*lamrn*exp(-t*(lamb+lamp))/
            (lama*lamb-lamb**2+lamp*(lama-lamb))+Nrn*lama*lamrn*
            exp(-t*(lama+lamp))/(lama**2-lama*lamb+lamp*(lama-lamb)))

    Nc = (Nrn*lama*lamb*lamrn/(lama*lamb*lamc+lamp**3+lamp**2*(lama+lamb+lamc)
            +lamp*(lama*lamb+lamc*(lama+lamb)))-
            Nrn*lama*lamb*lamrn*exp(-t*(lamc+lamp))/
            (lama*lamb*lamc+lamc**3-lamc**2*(lama+lamb)+lamp*(lama*lamb+lamc**2-
            lamc*(lama+lamb)))+Nrn*lama*lamb*lamrn*exp(-t*(lamb+lamp))/
            (lama*lamb**2-lamb**3-lamc*(lama*lamb-lamb**2)+lamp*(lama*lamb-
            lamb**2-lamc*(lama-lamb)))-Nrn*lama*lamb*lamrn*exp(-t*(lama+lamp))/
            (lama**3-lama**2*lamb-lamc*(lama**2-lama*lamb)+lamp*(lama**2-lama*
            lamb-lamc*(lama-lamb))))
    return Na, Nb, Nc

def calc_steady_state(Nrn, Q, rs, lamp,
                      V_tank, recoil_prob):
    """
    Compute the steady-state solution for the state variable Y

    Parameters
    ----------
     :


    Returns
    -------
    Yss : ndarray [Nrnd, Nrn, Fa, Fb, Fc]

    """
    # transit time assuming plug flow in the tank
    tt = V_tank / Q
    Na, Nb, Nc = calc_NaNbNc(tt, Nrn, lamp)
    # expressions based on these lines from detector_state_rate_of_change
    # dFadt = Q*rs*Na - Fa*lama
    # dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    # dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb
    Fa = Na*Q*rs/lama
    Fb = (Q*rs*Nb + Fa*lama * (1.0-recoil_prob)) / lamb
    Fc = (Q*rs*Nc + Fb*lamb) / lamc
    Yss = np.array( (Nrn, Nrn, Fa, Fb, Fc) )
    return Yss


def test_sage_NaNbNc():
    Q = 800.0 / 60.0 / 1000.0
    lamp = np.log(2.0)/120.0
    V_tank = 750.0 / 1000.0
    Nrn = 1.0
    t, numerical_soln = calc_NaNbNc_num(Q, V_tank, lamp, Nrn)
    analytical_soln = calc_NaNbNc(t, Nrn, lamp)
    analytical_soln = np.array(analytical_soln).T
    f, ax = plt.subplots()
    ax.plot(t, numerical_soln)
    ax.plot(t, analytical_soln)
    f, ax = plt.subplots()
    ax.plot(t, analytical_soln - numerical_soln)
    return t, numerical_soln, analytical_soln

def screen_penetration(U0, D, solid_fraction, wire_diameter, screen_thickness, n=1):
    """
    Compute the screen penetration fraction

    Parameters
    ----------
    U0 : float, m/s
        Freestream velocity

    D : float
        Particle diffusion coefficient

    solid_fraction : float, 0-1
        Screen solid fraction

    wire_diameter : float, m
        Diameter of individual wire in screen

    screen_thickness : float, m
        Screen thickness, typically twice wire diameter

    n : integer, default 1
        number of screens in array

    Returns
    -------
    P : float
        Penetration fraction of screen

    References
    ----------
    Main reference:
    Cheng, Y. S. and Yeh, H. C.: Theory of a screen-type diffusion battery,
    Journal of Aerosol Science, 11(3), 313–320,
    doi:10.1016/0021-8502(80)90105-6, 1980.

    Implementation based on:
    Heim, M., Mullins, B. J., Wild, M., Meyer, J. and Kasper, G.: Filtration
    Efficiency of Aerosol Particles Below 20 Nanometers, Aerosol Science and
    Technology, 39(8), 782–789, doi:10.1080/02786820500227373, 2005.
    """
    # Heim eq [4]
    Peclet_No = wire_diameter*U0/D
    # Heim eq [3]: diffusional collection efficiency
    nu_D = 2.7*Peclet_No**(-2.0/3.0)
    # Heim eq[2]: screen parameter
    S = 4 * solid_fraction * screen_thickness / (
                                np.pi * (1 - solid_fraction) * wire_diameter)
    # Heim eq [1]
    P = np.exp(-n*S*nu_D)
    return P

def Nrn_ext_spike(t):
    """
    Default external radon boundary condition

    1 Bq/m3 radon between t=0 and t=60 seconds, 0 otherwise

    Parameters
    ----------
    t : real
        time (seconds)

    Returns
    -------
    Nrn : real
        Radon concentration at detector inlet (atoms/m3)
    """
    if t<60.0 and t >=0.0:
        return 1.0 / lamrn
    else:
        return 0.0

def Nrn_ext_const(t):
    """
    Constant external radon boundary condition

    1 Bq/m3 radon

    Parameters
    ----------
    t : real
        time (seconds, not used)

    Returns
    -------
    Nrn : real
        Radon concentration at detector inlet (atoms/m3)
    """
    return 1.0 / lamrn

def Nrn_ext_stepwise_constant(t, tres, bc_values):
    """
    A stepwise-constant boundary condition

    This performs nearest-neighbour lookup between prescibed values.  Out-of-
    range values are extrapolated from the endpoints.

    Parameters
    ----------
    t : real
        time (seconds)

    tres : real
        time (seconds) between bc_values

    bc_values : ndarray
        radon concentration array.  bc_values[ii] is the radon concentration
        over the interval (ii, tres*ii]

    Returns
    -------
    Nrn : real
        Radon concentration at detector inlet (atoms/m3)
    """
    idx = max(int(np.ceil(t/tres)) - 1, 0)
    idx = min(idx, len(bc_values) - 1)
    return bc_values[idx]

def detector_state_rate_of_change(Y, t,
                                  Q, rs, lamp, eff, Q_external,
                                  V_delay, V_tank, recoil_prob,
                                  Nrn_ext=Nrn_ext_spike):
    """
    Compute the rate of change of the detector state (dY/dt)

    Parameters
    ----------
    Y : ndarray [Nrnd, Nrn, Fa, Fb, Fc]
        Current state vector made up of:
         - Nrnd radon concentration in delay tank
         - Nrn radon concentration in detector
         - Fa 218Po on filter
         - Fb 214Pb on filter
         - Fc 214Bi on filter

    t : real
        Current time (s)

    Q : real
        Internal loop flow rate (m3/s)

    rs : real (0-1)
        Dimensionless screen retention factor

    lamp : real
        Plate-out time constant (1/s)

    eff : real
        Counting efficiency (probability of an alpha decay on screen resulting
        in a photomultiplier tube count)

    Q_external : real
        External loop flow rate (m3/s)

    V_delay : real
        Volume of delay tank upstream of detector (m3)

    V_tank : real
        Volume of the detector internal volumn (m3)

    recoil_prob : real (0-1)
        Probability of 214Pb being lost after alpha decay of 218Po

    Nrn_ext : function (default: Nrn_ext_spike)
        A function which returns the external radon concentration as a function
        of time

    Returns
    -------
    dY/dt : ndarray
        Detector state rate of change
    """
    # unpack state vector
    Nrnd, Nrn, Fa, Fb, Fc = Y
    # effect of delay and tank volumes
    dNrnddt = Q_external / V_delay * (Nrn_ext(t) - Nrnd) - Nrnd*lamrn
    dNrndt = Q_external / V_tank * (Nrnd - Nrn) - Nrn*lamrn
    # Na, Nb, Nc from steady-state in tank
    # transit time assuming plug flow in the tank
    tt = V_tank / Q
    Na, Nb, Nc = calc_NaNbNc(tt, Nrn, lamp)
    # compute rate of change of each state variable
    dFadt = Q*rs*Na - Fa*lama
    dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb
    return np.array([dNrnddt, dNrndt, dFadt, dFbdt, dFcdt])



def detector_model(t, Y0 = np.zeros(5),
                      Q = 800.0 / 60.0 / 1000.0,
                      rs = 0.7,
                      lamp = np.log(2.0)/120.0,
                      eff = 0.33, # Whittlestone's paper
                      Q_external = 40.0 / 60.0 / 1000.0,
                      V_delay = 200.0 / 1000.0,
                      V_tank = 750.0 / 1000.0,
                      t_delay = 60.0,
                      recoil_prob = 0.02,
                      Nrn_ext=Nrn_ext_spike,
                      return_full_state=False,
                      **ignored_kwargs):

    parameters = (Q, rs, lamp, eff, Q_external,
                  V_delay, V_tank, recoil_prob, Nrn_ext)

    from util import timewith
    # ensure functions have been compiled
    detector_state_rate_of_change(Y0, 0, *parameters)
    # with timewith("ODE integration") as timer:
    soln = odeint(detector_state_rate_of_change, Y0, t-t_delay,
                  args=parameters, hmax=30.0)

    df = pd.DataFrame(index=t/60.0, data=soln)
    df.columns = 'Nrnd,Nrn,Fa,Fb,Fc'.split(',')
    df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)
    if return_full_state:
        tt = V_tank / Q
        df['Nrn_ext'] = [Nrn_ext(itm) for itm in t]
        df['Na'], df['Nb'], df['Nc'] = calc_NaNbNc(tt, df['Nrn'].values, lamp)

    return df


def detector_model_wrapper(timestep, initial_state, external_radon_conc,
                           internal_airt_history,
                           parameters, interpolation_mode=1,
                           return_full_state=False):
    """
    TODO:
    """
    t = np.arange(0, timestep*len(external_radon_conc), timestep, dtype=np.float)
    params = fast_model.parameter_array_from_dict(parameters)
    soln = fast_model.detector_model(timestep, interpolation_mode,
                                  external_radon_conc,
                                  internal_airt_history,
                                  initial_state, params)
    df = pd.DataFrame(index=t/60.0, data=soln)
    df.columns = 'Nrnd,Nrn,Fa,Fb,Fc,Acc_counts'.split(',')
    eff = parameters['eff']
    df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)
    if return_full_state:
        #TODO
        assert(False)
    return df

def detector_model_observed_counts(timestep, initial_state, external_radon_conc,
                           internal_airt_history,
                           parameters, interpolation_mode=1):
    """just return the observed_counts timeseries"""
    params = fast_model.parameter_array_from_dict(parameters)
    soln = fast_model.detector_model(timestep, interpolation_mode,
                                  external_radon_conc, internal_airt_history,
                                  initial_state, params)
    return np.diff(soln[:,-1])


def test_detector_model(doplots=False):
    """
    run some tests, perturbing parameters in the detector model
    """
    Y0 = np.zeros(5)
    timestep = 60
    t  = np.arange(0, 3600*5, timestep)   # time grid
    #t  = np.arange(0, 3600*24, 60*5)   # time grid

    Q = 800.0 / 60.0 / 1000.0
    rs = 0.7
    lamp = np.log(2.0)/120.0
    eff = 0.33
    Q_external = 40.0 / 60.0 / 1000.0
    V_delay = 200.0 / 1000.0
    V_tank = 750.0 / 1000.0
    t_delay = 0 #30*60.
    t_delay = 60.0 * 10
    recoil_prob = 0.02

    parameters = dict(Q=Q, rs=rs, lamp=lamp, eff=eff, Q_external=Q_external,
                      V_delay=V_delay, V_tank=V_tank, recoil_prob=recoil_prob,
                      t_delay=t_delay)

    # perturb some things
    df = detector_model(t, Y0, **parameters)

    external_radon_conc = np.zeros(len(t))
    external_radon_conc[1] = 1./lamrn
    internal_airt_history = np.zeros(len(t)) + 300.0

    # parameters['t_delay'] -= 60.0

    df2 = detector_model_wrapper(timestep, Y0, external_radon_conc,
                                 internal_airt_history,
                                 interpolation_mode=0, parameters=parameters)


    dfp = detector_model(t, Q=Q*2)
    df['count rate Q*2'] = dfp['count rate']
    dfp = detector_model(t, rs=1.0)
    df['count rate rs=1'] = dfp['count rate']
    dfp = detector_model(t, lamp=np.log(2.0)/20.0)
    df['count rate plateout=20s'] = dfp['count rate']
    dfp = detector_model(t, Q_external=Q_external*1.1)
    df['count rate Q_external*1.1'] = dfp['count rate']
    dfp = detector_model(t, recoil_prob=0.5)
    df['count rate recoil_prob=0.5'] = dfp['count rate']

    if doplots:
        # compare python with C implementations
        f, ax = plt.subplots()
        df['count rate, c impl'] = df2['count rate']
        df[['count rate, c impl', 'count rate']].plot(ax=ax)
        # some other stuff
        if False:
            cols = [itm for itm in df.columns if 'count rate' in itm]
            df[cols].plot()
            norm = df['count rate'].max()
            for itm in cols:
                df[itm] = df[itm]/norm #df[itm].mean()
            df[cols].plot()

            for itm in cols:
                df[itm] /= df[itm].mean()
            df[cols].plot()



    return df


def test_steady_state():
    """
    test the steady-state solution obtained from Sage
    """
    Y0 = np.zeros(5)
    #t  = np.arange(0, 3600*5, 60)   # time grid
    t  = np.arange(0, 3600*24, 60*5)   # time grid
    t = np.linspace(0, 3600*24*30, 1000)

    parameters = dict(
    Q = 800.0 / 60.0 / 1000.0, # from Whittlestone's paper, L/min converted to m3/s
    rs = 0.7, # from Whittlestone's paper (screen retention)
    lamp = np.log(2.0)/120.0, # from Whittlestone's code (1994 tech report)
    eff = 0.33, # Whittlestone's paper
    Q_external = 40.0 / 60.0 / 1000.0,
    V_delay = 200.0 / 1000.0,
    V_tank = 750.0 / 1000.0,
    recoil_prob = 0.02,
    t_delay = 60.0)

    #for k,val in parameters.items():
    #    parameters[k]/=2.0

    df = detector_model(t, Nrn_ext = Nrn_ext_const,
                         return_full_state=True, **parameters)
    Nrn = Nrn_ext_const(1)
    ret = calc_steady_state(Nrn, Q=parameters['Q'], rs=parameters['rs'],
                            lamp=parameters['lamp'],
                            V_tank=parameters['V_tank'],
                            recoil_prob=parameters['recoil_prob'])
    df['Fa_ss'], df['Fb_ss'], df['Fc_ss'] = ret[2:]

    f, ax = plt.subplots()
    df[['Fa','Fb','Fc', 'Fa_ss', 'Fb_ss', 'Fc_ss']].plot(ax=ax)
    #df[['Fa', 'Fa_ss']].plot()

    return df


def calc_detector_efficiency(parameters):
    """
    Compute steady-state counting efficiency (counts per Bq/m3 of radon)
    """
    Y0 = calc_steady_state(Nrn=1.0/lamrn, Q=parameters['Q'], rs=parameters['rs'],
                        lamp=parameters['lamp'],
                        V_tank=parameters['V_tank'],
                        recoil_prob=parameters['recoil_prob'])
    Nrnd,Nrn,Fa,Fb,Fc = Y0
    steady_state_counts = parameters['eff']*(Fa*lama + Fc*lamc)
    steady_state_efficiency = steady_state_counts / 1.0
    return steady_state_efficiency

def calc_detector_activity_a_to_c_ratio(parameters):
    """
    Calculate the ratio of counts from Fa to counts from Fb at steady state
    """
    Y0 = calc_steady_state(Nrn=1.0/lamrn, Q=parameters['Q'], rs=parameters['rs'],
                        lamp=parameters['lamp'],
                        V_tank=parameters['V_tank'],
                        recoil_prob=parameters['recoil_prob'])
    Nrnd,Nrn,Fa,Fb,Fc = Y0
    Fa_counts = Fa*lama
    Fc_counts = Fc*lamc
    return Fa_counts/Fc_counts



# Below here should go into a new file
################################################################################
################################################################################




"""
def run_model_and_massage_into_data_frame(df, obs_column='lld'
#as a class?

parameters = [ inlet_radon_concentration[N],
               Q_external, Q, recoil_prob, rs
"""


def gen_initial_guess(observed_counts, one_sided_prf, reg='tv'):
    """
    an initial guess based on the RL deconvolution

    use emcee.utils.sample_ball to generate perturbed guesses for each walker
    """
    M = len(one_sided_prf)
    symmetric_prf = np.r_[np.zeros(M-1), one_sided_prf]
    Ndim = len(observed_counts) + M - 1
    # pad first to avoid end effects
    pad0 = np.ones(M)*observed_counts[0]
    pad1 = np.ones(M)*observed_counts[-1]
    observed_counts_padded = np.r_[pad0, observed_counts, pad1]
    initial_guess = util.deconvlucy1d(observed_counts_padded, symmetric_prf,
                                     iterations=1000, reg=reg)
    # exclude padding from return value
    initial_guess = initial_guess[M+1:-M]
    return initial_guess


def fit_parameters_to_obs(t, observed_counts, radon_conc=[], parameters=dict(),
                          walkers_per_dim=3, keep_burn_in_samples=False, thin=1,
                          iterations=100):
    """
    TODO: move this into its own file - it's too big to go here.
    """
    # make a local copy of the paramters dictionary
    parameters_ = parameters
    parameters = {}
    parameters.update(parameters_)

    radon_conc_is_known = (len(radon_conc) == len(t))
    parameters['observed_counts'] = observed_counts

    if radon_conc_is_known:
        print("Trying to adjust hyper parameters to match observations")
        parameters['radon_conc'] = radon_conc
    else:
        print("Trying to deconvolve observations")

    # TODO: put info about hyper-parameters into the parameters dict
    hyper_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay'
    nhyper = len(hyper_parameter_names)
    nstate = fast_model.N_state


    # TODO: these should be provided as arguments
    hyper_parameters_mu_prior = np.array(
                            [parameters[k] for k in hyper_parameter_names])
    hyper_parameters_sigma_prior = np.array([parameters['Q_external'] * 0.02,
                                             parameters['Q']*0.02,
                                             0.05,
                                             np.log(2),
                                             5.])
    parameters.update( dict(hyper_parameter_names=hyper_parameter_names,
                            nhyper=nhyper,
                            nstate=nstate,
                            hyper_parameters_mu_prior=hyper_parameters_mu_prior,
                            hyper_parameters_sigma_prior=
                                                hyper_parameters_sigma_prior))
    # Detector state at t=0, prior and initial guess
    Y0 = calc_steady_state(Nrn=1.0, Q=parameters['Q'], rs=parameters['rs'],
                        lamp=parameters['lamp'],
                        V_tank=parameters['V_tank'],
                        recoil_prob=parameters['recoil_prob'])
    Y0 = np.r_[Y0, 100.0]  # arbitrary value for Acc_counts- simplify algo later
    Nrnd,Nrn,Fa,Fb,Fc, Acc_counts = Y0
    expected_counts = parameters['eff']*(Fa*lama + Fc*lamc) * (t[1]-t[0])
    scale_factor = observed_counts[0] / expected_counts
    Y0_mu_prior = Y0 * scale_factor


    def unpack_parameters(p, model_parameters):
        """
        unpack paramters from vector and return dict
        """
        nhyper = model_parameters['nhyper']
        nstate = model_parameters['nstate']
        Y0 = p[:nstate]
        parameters = {'Y0':Y0}
        #hyper_parameter_names = 'Q_external', 'Q', 'rs', 'lamp', 't_delay'
        #nhyper = len(hyper_parameter_names)
        hyper_parameters = p[nstate:nhyper+nstate]
        radon_concentration_timeseries = p[nhyper+nstate:]
        parameters.update( zip(hyper_parameter_names, hyper_parameters) )
        return parameters, Y0, hyper_parameters, radon_concentration_timeseries

    def pack_parameters(Y0, hyper_parameters, radon_concentration_timeseries=[]):
        return np.r_[Y0, hyper_parameters, radon_concentration_timeseries]


    parameters['tres'] = t[1] - t[0]
    assert(np.allclose(np.diff(t), parameters['tres']))

    if not radon_conc_is_known:
        # generate initial guess by (1) working out the PSF, (2) RL deconv.
        # determine PSF for these parameters
        df = detector_model(t, Y0*0.0, **parameters)
        #TODO: taking the first 61 points is a hack
        #TODO: adding that small constant is a hack because RL deconv doesn't work
        #      when there's a zero in the one-sided prf (apparently)
        one_sided_prf = df['count rate'].values[:61] + 0.000048
        one_sided_prf = one_sided_prf / one_sided_prf.sum()

        radon_conc = gen_initial_guess(observed_counts,
                                                                 one_sided_prf)
    if radon_conc_is_known:
        p = pack_parameters(Y0_mu_prior, hyper_parameters_mu_prior, [])
    else:
        p = pack_parameters(Y0_mu_prior, hyper_parameters_mu_prior, radon_conc)

    p00 = p.copy()

    def detector_model_specialised(p, parameters):
        """
        Detector model, specialised for use with emcee
        """
        (varying_parameters, Y0, hyper_parameters,
            radon_concentration_timeseries) = unpack_parameters(p, parameters)

        #print(varying_parameters)

        parameters.update(varying_parameters)
        # link recoil probability to screen efficiency
        parameters['recoil_prob'] = 0.5*(1.0-parameters['rs'])
        N = len(radon_concentration_timeseries)
        if N==0:
            assert radon_conc_is_known
            radon_concentration_timeseries = parameters['radon_conc']
            N = len(radon_concentration_timeseries)

        if False:  # call python version of detector model

            #boundary condition
            tres = parameters['tres']
            def Nrn_ext(t):
                return Nrn_ext_stepwise_constant(t, tres,
                                                 radon_concentration_timeseries)
            # output times (TODO: make this higher resolution than observations
            # followed by downsampling)
            t = np.arange(0, tres*(N), tres)
            assert len(t) == len(radon_concentration_timeseries)
            # python version
            df = detector_model(t, Nrn_ext=Nrn_ext, **parameters)
            detector_count_rate = df['count rate']
        else:
            timestep = parameters['tres']
            external_radon_conc = radon_concentration_timeseries
            cr = detector_model_observed_counts(timestep, parameters['Y0'],
                                         external_radon_conc,
                                         parameters, interpolation_mode=0)
            detector_count_rate = cr
        return detector_count_rate


    def lnlike(p, parameters):
        observed_counts = parameters['observed_counts']
        Nobs = len(observed_counts)
        detector_count_rate = detector_model_specialised(p, parameters)
        if not len(detector_count_rate) == Nobs-1:
            print(len(detector_count_rate), Nobs)
            assert False
        #scale counts so that total number of counts is preserved (?)
        # detector_count_rate
        lp = poisson.logpmf(observed_counts[1:], detector_count_rate)
        lp = lp.sum()
        #f, ax = plt.subplots()
        #ax.plot(observed_counts)
        #ax.plot(detector_count_rate)
        #plt.show()
        return lp


    def lnprior_hyperparameters(p, parameters):
        """
        Prior constraints on hyper-parameters
        """
        (varying_parameters, Y0, hyper_parameters,
            radon_concentration_timeseries) = unpack_parameters(p, parameters)
        Q_external, Q, rs, lamp, t_delay = hyper_parameters
        idxlamp = 3
        if Q_external <= 0 or Q<=0 or rs <=0 or rs >=1 or lamp <=0:
            return -np.inf
        # normally-distributed priors
        idx = np.arange(len(hyper_parameters))
        idx = idx != idxlamp
        lp = norm.logpdf(hyper_parameters, hyper_parameters_mu_prior,
                                            hyper_parameters_sigma_prior)[idx].sum()
        # lamp has log-normal prior
        log_lamp_mu = np.log(hyper_parameters_mu_prior[idxlamp])
        log_lamp_sigma = hyper_parameters_sigma_prior[idxlamp]
        lp += norm.logpdf(np.log(lamp), log_lamp_mu, log_lamp_sigma)

        return lp

    def lnprior_Y0(Y0):
        """
        Prior on detector state at t=0
        """
        if Y0.min() <= 0.0:
            return -np.inf
        # note - for parameter definitions see
        # http://nbviewer.ipython.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/lognormal_distributions.ipynb
        sigma = np.log(2.0) # Standard deviation of log(X) - factor of two
        shape = sigma # Scipy's shape parameter
        scale = Y0_mu_prior # Scipy's scale parameter = np.exp( mean of log(X) )
        ret = lognorm.logpdf(Y0, shape, loc=0, scale=scale)
        ret = ret.sum()
        return ret

    def lnprior_difference(radon_concentration_timeseries):
        """
        log-normal prior on step-by-step changes in radon concentration
        """
        p = radon_concentration_timeseries
        # Parameters must all be > 0
        if p.min() <= 0:
            lp =  -np.inf
        else:
            dpdt = np.diff(np.log(p))
            mu = 0.0  # mean expected change - no change
            #sigma = np.log(2) #standard deviation - factor of two change
            sigma = np.log(1.5)
            #sigma = np.log(1.05) # much more smoothing
            lp = norm.logpdf(dpdt, mu, sigma).sum()
        return lp

    def lnprior_params(p, parameters):
        """
        comine priors
        """
        (varying_parameters, Y0, hyper_parameters, radon_concentration_timeseries
           ) = unpack_parameters(p, parameters)
        lp = 0.0
        if len(radon_concentration_timeseries) > 0:
            # radon concentrations are not known (deconvolution)
            lp += lnprior_difference(radon_concentration_timeseries)
        lp += lnprior_Y0(Y0)
        lp += lnprior_hyperparameters(p, parameters)
        return lp

    def lnprob(p, parameters):
        # print(len(p), p/p00)
        lp = lnprior_params(p, parameters)
        if np.isfinite(lp):
            lp += lnlike(p, parameters)
        return lp

    print(p)
    print(parameters)
    print(unpack_parameters(p, parameters)[0])
    # we should now be able to compute the liklihood of the initial location p
    print("P0 log-prob:",lnprob(p, parameters))
    # the function should return -np.inf for negative values in parameters
    # (with the exception of the delay time)
    #for ii in range(len(p)):
    #    pp = p.copy()
    #    pp[ii] *= -1
    #    print(ii, lnprob(pp, parameters))
    #assert(False)
    #print("check:", lnprob(np.r_[Y0_mu_prior, p[5:]], parameters))
    # it should be possible to optimise the a posteria probability
    if False:
        # but it's not possible to optimise, perhaps because the optimisation
        # algorithm fails
        def minus_lnprob(p,parameters):
            p = np.r_[Y0_mu_prior, p]
            lp = lnprob(p,parameters)
            print(p)
            print(lp)
            return - lp
        #check we can call this function
        #print('minus lnprob:', minus_lnprob(p[5:], parameters))
        from scipy.optimize import minimize
        ret = minimize(minus_lnprob, x0=p[5:], args=(parameters,), method='COBYLA',
                            options=dict(eps=p[5:]/1000.))
        print(ret)
        pmin = ret.x
        print("P_opt log-prob:", lnprob(pmin, parameters))

    # Number of walkers needs to be at least 2x number of dimensions
    Ndim = len(p)
    Nwalker = Ndim * walkers_per_dim
    Nwalker = max(Nwalker, 30) # don't run with less than 100 walkers
    # number of walkers must be even.
    # increment to the next multiple of 4 (for, maybe, easier load balancing)
    Nwalker += (4 - Nwalker % 4)
    p00 = p.copy()
    p0 = emcee.utils.sample_ball(p, std=p/100.0, size=Nwalker)

    # sampler
    sampler = emcee.EnsembleSampler(Nwalker,Ndim,lnprob,
                                    args=(parameters,),
                                    threads=1)
    # burn-in
    pos,prob,state = sampler.run_mcmc(p0, iterations,
                                    storechain=keep_burn_in_samples, thin=thin)

    # sample
    pos,prob,state = sampler.run_mcmc(pos, iterations, thin=thin)
    A = sampler.flatchain

    mean_est = A.mean(axis=0)
    low = np.percentile(A, 10.0, axis=0)
    high = np.percentile(A, 90.0, axis=0)

    return sampler, A, mean_est, low, high





def test_fit_to_obs():
    import util
    fname = 'data-controlled-test-2/T1Mar15e.CSV'
    dfobs = util.load_radon(fname)


    dom = 23   # 18--26 are spikes
    dom = 29
    for dom in [29]: #range(18,30): #this should be an argument

        t0 = datetime.datetime(2015,3,dom,11)
        dt = datetime.timedelta(hours=12)
        dt = datetime.timedelta(days=3)

        f, ax = plt.subplots()
        dfobs.lld[t0:t0+dt].plot()


        # work out the net counts and mean background
        t1 = datetime.datetime(2015,3,dom,13)
        t2 = datetime.datetime(2015,3,dom,20)

        dfss = dfobs.ix[t1-datetime.timedelta(hours=6):
                        t1+datetime.timedelta(hours=12)].copy()

        is_spike = dfss.lld.max() > 20000
        if is_spike:
            inj_minutes = 1
        else:
            inj_minutes = 60


        nhrs = 20-13
        total_count = dfobs.lld.cumsum()[t2] - dfobs.lld.cumsum()[t1]

        background_count_rate = dfobs.lld[t1-datetime.timedelta(hours=6):t1].mean()
        background_count = background_count_rate * nhrs * 60

        # we know that the data are one-minute averages, and that the injection goes
        # from 1300-1400 (for square wave) or 1300-1301 (for spike)
        injection_count_rate = (total_count - background_count) / inj_minutes

        parameters = dict(
            Q = 800.0 / 60.0 / 1000.0, # from Whittlestone's paper, L/min converted to m3/s
            rs = 0.7, # from Whittlestone's paper (screen retention)
            lamp = np.log(2.0)/120.0, # from Whittlestone's code (1994 tech report)
            eff = 0.33, # Whittlestone's paper
            Q_external = 40.0 / 60.0 / 1000.0,
            V_delay = 200.0 / 1000.0,
            V_tank = 750.0 / 1000.0,
            recoil_prob = 0.02,
            t_delay = 60.0)

        # tweak parameters for better match to obs
        parameters['Q_external'] *= 1.03
        parameters['rs'] = 0.92
        parameters['recoil_prob'] = 0.04
        parameters['t_delay'] = 1.0

        # extract time in seconds
        times = dfss.index.to_pydatetime()
        tzero = times[0]
        t = np.array([ (itm-tzero).total_seconds() for itm in times])
        tres = t[1] - t[0]

        # work out detector efficience for given parameters
        total_eff = calc_detector_efficiency(parameters)
        print("predicted detector counts per Bq/m3 radon:", total_eff)

        radon_conc_bq = np.r_[ np.ones(6*60+1)*background_count_rate,
                            np.ones(inj_minutes)*injection_count_rate,
                            np.ones(11*60+60-inj_minutes)*background_count_rate
                             ] / total_eff / tres
        dfss['radon_conc'] = radon_conc_bq / lamrn



        # run the model
        # to ensure that the initial guess isn't going totally off the mark
        Y0 = fast_model.calc_steady_state(dfss.radon_conc.values[0],
                                Q=parameters['Q'], rs=parameters['rs'],
                                lamp=parameters['lamp'],
                                V_tank=parameters['V_tank'],
                                recoil_prob=parameters['recoil_prob'],
                                eff=parameters['eff'])

        lldmod = detector_model_observed_counts(tres,
                                           Y0,
                                           dfss.radon_conc.values,
                                           parameters,
                                           interpolation_mode=0)

        dfss['lldmod'] = np.r_[np.NaN, lldmod]

        f, ax = plt.subplots()
        dfss[['lld','lldmod']].plot(ax=ax)


        f, ax = plt.subplots()
        tinj = t1
        dfss[['lld','lldmod']].ix[tinj:tinj+datetime.timedelta(minutes=120)].plot(ax=ax)

        if True:
            fit_ret = fit_parameters_to_obs(t, observed_counts=dfss.lld.values,
                                 radon_conc=dfss.radon_conc.values,
                                 parameters=parameters,
                                 iterations=100,
                                 keep_burn_in_samples=False)

            sampler, A, mean_est, low, high = fit_ret
            popt = A.mean(axis=0)
            # this is fragile...
            Y0 = popt[:fast_model.N_state]  # currently 6
            hyper_parameters = popt[fast_model.N_state:]
            Q_external, Q, rs, eff, t_delay = hyper_parameters
            parameters.update(dict(Q_external=Q_external, Q=Q,
                                   rs=rs, eff=eff, t_delay=t_delay))

            lldmodopt = detector_model_observed_counts(tres,
                                           Y0,
                                           dfss.radon_conc.values,
                                           parameters,
                                           interpolation_mode=0)

            dfss['lldmodopt'] = np.r_[np.NaN, lldmodopt]

            np.save(str(dom) + 'chain.npy', sampler.chain)

        else:
            fit_ret = None

    return dfss, fit_ret




if __name__ == "__main__":

    #t, numerical_soln, analyical_soln = test_sage_NaNbNc()

    df = test_detector_model(doplots=True)
    #df = test_steady_state()


    if True:
        dfss, fit_ret = test_fit_to_obs()
        plt.show()

        sampler, A, mean_est, low, high = fit_ret

        for ii in range(10):
            f,ax = plt.subplots()
            blah = ax.plot(sampler.chain[:,:,ii].T)

    plt.show()
