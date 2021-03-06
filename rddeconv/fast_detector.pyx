# distutils: language = c++
# distutils: sources = rddeconv/fast_detector_model.cpp

"""
references:
  http://docs.cython.org/src/tutorial/numpy.html
  http://docs.cython.org/src/userguide/memoryviews.html
  http://stackoverflow.com/questions/17855032/passing-and-returning-numpy-arrays-to-c-methods-via-cython
  """


from __future__ import division
import numpy as np


from libc.math cimport exp
from libc.math cimport log
from libc.math cimport tanh
#from libc.math cimport isfinite

import cython
cimport numpy as np
from cython.view cimport array as cvarray

cdef extern from "<cmath>" namespace "std":
    cdef bint isfinite(long double)

#this is the c (or c++) function
cdef extern from "fast_detector_model.hpp" nogil:
    const int NUM_STATE_VARIABLES
    const int NUM_PARAMETERS
    const double lamrn
    const double lama
    const double lamb
    const double lamc
    int integrate_radon_detector(int N_times,
                              double timestep,
                              int interpolation_mode,
                              double* external_radon_conc,
                              double* internal_airt_history,
                              double* initial_state,
                              double* state_history,
                              double* parameters)
    double linear_interpolation(double xi, int N, double timestep,
                                const double *y, const int mode)

# make these available to python
N_state = int(NUM_STATE_VARIABLES)
N_param = int(NUM_PARAMETERS)

def linear_interpolation_test(double xi,
                              double timestep,
                              np.ndarray[np.double_t, ndim=1] y,
                              int mode=1):
    cdef int N
    N = len(y)
    return linear_interpolation(xi, N, timestep, &y[0], mode)

def parameter_array_from_dict(parameter_dict):
    # this needs to be kept in sync with two_filter_detector's constructor
    # TODO: doc for parameters
    cdef np.ndarray[np.double_t, ndim=1,
                    mode="c"] parameter_array = np.zeros(NUM_PARAMETERS)
    parameter_array =  np.array((
                                 parameter_dict['Q'],
                                 parameter_dict['rs'],
                                 parameter_dict['lamp'],
                                 parameter_dict['eff'],
                                 parameter_dict['Q_external'],
                                 parameter_dict['V_delay'],
                                 parameter_dict.get('V_delay_2',0.0),
                                 parameter_dict['V_tank'],
                                 parameter_dict['t_delay'],
                                 parameter_dict['recoil_prob'],
                                 parameter_dict.get('cal_source_strength',0),
                                 parameter_dict.get('cal_begin',-9999),
                                 parameter_dict.get('cal_duration',0),
                                 parameter_dict.get('inj_source_strength',0),
                                 parameter_dict.get('inj_begin',-9999),
                                 parameter_dict.get('inj_duration',0)
                                 ), dtype=np.float)
    return parameter_array

@cython.cdivision(True)
def calc_NaNbNc(double t, double Nrn, double lamp):
    """
    Compute concentrations of radon daughters at time t

    Assumes plug flow.  Based on analytical solution of Eq. A1--A3 in W&Z 1996

    Parameters
    ----------
    t : double
        time in seconds

    TODO:

    Returns
    -------
    Na,Nb,Nc : double
        Concentration of radon-222 daughters
    """
    cdef double Na, Nb, Nc
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


# the above function, broken into parts
@cython.cdivision(True)
cdef double calc_Na(double t, double Nrn, double lamp):
    cdef double Na
    Na = Nrn*lamrn/(lama+lamp)-Nrn*lamrn*exp(-t*(lama+lamp))/(lama+lamp)
    return Na

@cython.cdivision(True)
cdef double calc_Nb(double t, double Nrn, double lamp):
    cdef double Nb
    Nb = (Nrn*lama*lamrn/(lama*lamb+lamp**2+lamp*(lama+lamb))-
            Nrn*lama*lamrn*exp(-t*(lamb+lamp))/
            (lama*lamb-lamb**2+lamp*(lama-lamb))+Nrn*lama*lamrn*
            exp(-t*(lama+lamp))/(lama**2-lama*lamb+lamp*(lama-lamb)))
    return Nb

@cython.cdivision(True)
cdef double calc_Nc(double t, double Nrn, double lamp):
    cdef double Nc
    Nc = (Nrn*lama*lamb*lamrn/(lama*lamb*lamc+lamp**3+lamp**2*(lama+lamb+lamc)
            +lamp*(lama*lamb+lamc*(lama+lamb)))-
            Nrn*lama*lamb*lamrn*exp(-t*(lamc+lamp))/
            (lama*lamb*lamc+lamc**3-lamc**2*(lama+lamb)+lamp*(lama*lamb+lamc**2-
            lamc*(lama+lamb)))+Nrn*lama*lamb*lamrn*exp(-t*(lamb+lamp))/
            (lama*lamb**2-lamb**3-lamc*(lama*lamb-lamb**2)+lamp*(lama*lamb-
            lamb**2-lamc*(lama-lamb)))-Nrn*lama*lamb*lamrn*exp(-t*(lama+lamp))/
            (lama**3-lama**2*lamb-lamc*(lama**2-lama*lamb)+lamp*(lama**2-lama*
            lamb-lamc*(lama-lamb))))
    return Nc

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_steady_state(double Nrn, double Q, double rs, double lamp,
                      double V_tank, double recoil_prob, double eff):
    """
    Compute the steady-state solution for the state variable Y

    Parameters
    ----------
     :


    Returns
    -------
    Yss : ndarray [Nrnd, Nrnd2, Nrn, Fa, Fb, Fc]

    """
    # transit time assuming plug flow in the tank
    cdef double tt = V_tank / Q
    cdef double Na, Nb, Nc, Fa, Fb, Fc

    Na = calc_Na(tt, Nrn, lamp)
    Nb = calc_Nb(tt, Nrn, lamp)
    Nc = calc_Nc(tt, Nrn, lamp)
    # expressions based on these lines from detector_state_rate_of_change
    # dFadt = Q*rs*Na - Fa*lama
    # dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    # dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb
    Fa = Na*Q*rs/lama
    Fb = (Q*rs*Nb + Fa*lama * (1.0-recoil_prob)) / lamb
    Fc = (Q*rs*Nc + Fb*lamb) / lamc
    cdef double Acc_counts = eff*(Fa*lama + Fc*lamc)
    cdef np.ndarray[np.double_t, ndim=1,
                mode="c"] Yss = np.empty(NUM_STATE_VARIABLES)
    Yss[0] = Nrn
    Yss[1] = Nrn
    Yss[2] = Nrn
    Yss[3] = Fa
    Yss[4] = Fb
    Yss[5] = Fc
    Yss[6] = Acc_counts
    return Yss



def detector_model(double timestep,
                   int interpolation_mode,
                   np.ndarray[np.double_t, ndim=1] external_radon_conc,
                   np.ndarray[np.double_t, ndim=1] internal_airt_history,
                   np.ndarray[np.double_t, ndim=1] initial_state,
                   np.ndarray[np.double_t, ndim=1] parameters
                   ):
    """
    Model of the two-filter radon detector


    Parameters
    ----------
    timestep : double
        Timestep, in seconds, for input and output arrays

    interpolation_mode : [0, 1]
        Mode to use for interpolating the external radon concentration
        0 means stepwise-constant, 1 means linear interpolation

    external_radon_conc : np.ndarray(N_times)
        External radon concentration timeseries

    internal_airt_history : np.ndarray(N_times)
        Internal air temperature timeseries

    initial_state : np.ndarray(N_state)
        Initial state of the radon detector

    parameters : np.ndarray(N_param)
        Model parameters

    Returns
    -------
    state_history : np.ndarray((N_times, NUM_STATE_VARIABLES))
        Model state as a function of time
    """

    external_radon_conc = np.ascontiguousarray(external_radon_conc)
    internal_airt_history = np.ascontiguousarray(internal_airt_history)
    initial_state = np.ascontiguousarray(initial_state)


    # allow the initial state to be missing the accumulated counts -
    # and set it to zero
    cdef np.ndarray[np.double_t, ndim=1,
                    mode="c"] new_is = np.empty(NUM_STATE_VARIABLES)
    if len(initial_state) == NUM_STATE_VARIABLES - 1:
        new_is[:-1] = initial_state
        new_is[-1] = 0.0
        initial_state = new_is

    cdef int N_times = len(external_radon_conc)
    
    # check inputs
    if not (len(internal_airt_history) == N_times):
        raise ValueError('The length of the air temperature array'
                                    ' is not equal to N_times')
    if not(NUM_STATE_VARIABLES == len(initial_state)):
        raise ValueError('The initial state has length {} but'
                                    ' needs length {}'.format(
                                    len(initial_state), NUM_STATE_VARIABLES))
    if not (NUM_PARAMETERS == len(parameters)):
        raise ValueError('The parameters array has length {} but'
                                    ' needs length {}'.format(len(parameters), 
                                                              NUM_PARAMETERS))
    if not(interpolation_mode == 0 or interpolation_mode == 1):
        raise ValueError('interpolation_mode has value {} but may'
                                    ' only equal 0 or 1'.format(
                                        interpolation_mode))

    output_dims = (N_times, NUM_STATE_VARIABLES)
    cdef np.ndarray[np.double_t, ndim=2,
                    mode="c"] state_history = np.zeros(output_dims)

    cdef int err

    with nogil:
        err = integrate_radon_detector(N_times,
                                   timestep,
                                   interpolation_mode,
                                   &external_radon_conc[0],
                                   &internal_airt_history[0],
                                   &initial_state[0],
                                   &state_history[0,0],
                                   &parameters[0])
    if err:
        raise(RuntimeError)

    return state_history


#
# ... utility functions for transforming the radon timeseries into an
# ... array parameters, which should hopefully be easier for the optimisation
# ... routines
#

cdef double logit(double p):
    return log(p) - log(1.0 - p)

cdef double inv_logit(double p):
    #return np.exp(p) / (1 + np.exp(p))
    # use the fact that 2*inv_logit(2*p)-1 == tanh(p)
    return (tanh(p/2.0)+1.0)/2.0

@cython.cdivision(True)
cdef double transform_constrained_to_unconstrained(double x, double a=0, double b=1):
    y = logit( (x-a)/(b-a) )
    return y

cdef double transform_unconstrained_to_constrained(double y, double a=0, double b=1):
    x = a + (b-a) * inv_logit(y)
    return x

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def transform_radon_concs(np.ndarray[np.double_t, ndim=1] radon_conc):
    cdef int N = radon_conc.shape[0]
    cdef int ii
    cdef np.ndarray[np.double_t, ndim=1,
                    mode="c"] p = np.empty(N)
    # was:
    # cdef double rnsum = radon_conc.sum()
    cdef double rnsum = 0.0
    for ii in range(N):
        rnsum += radon_conc[ii]
    cdef double tmp = 0.0
    p[0] = log(rnsum)
    cdef double acc = rnsum
    for ii in range(N - 1):
        tmp = radon_conc[ii] / acc
        p[ii+1] = transform_constrained_to_unconstrained(tmp)
        acc -= radon_conc[ii]
    #resid = radon_conc[len(radon_conc)] - acc
    return p

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int transform_radon_concs_inplace(np.ndarray[np.double_t, ndim=1] radon_conc):
    cdef int N = radon_conc.shape[0]
    cdef int ii
    cdef double rnsum = 0.0
    cdef double tmp = 0.0
    cdef double tmp_prev = 0.0
    for ii in range(N):
        rnsum += radon_conc[ii]
    cdef double acc = rnsum
    tmp_prev = log(rnsum)
    # this version of the loop uses a temporay value because we
    # need to read the [ii+1]'th location while writing to the [ii]'th
    for ii in range(N - 1):
        tmp = radon_conc[ii]
        radon_conc[ii] = tmp_prev
        tmp_prev = transform_constrained_to_unconstrained(tmp / acc)
        acc -= tmp
    radon_conc[N-1] = tmp_prev
    return 0

@cython.boundscheck(False)
cpdef inverse_transform_radon_concs(np.ndarray[np.double_t, ndim=1] p):
    cdef int N = p.shape[0]
    cdef np.ndarray[np.double_t, ndim=1,
                    mode="c"] radon_conc = np.empty(N)
    cdef double acc = exp(p[0])
    cdef int ii = 0
    cdef double tmp = 0.0
    cdef double rn = 0.0

    for ii in range(N - 1):
        tmp = transform_unconstrained_to_constrained(p[ii+1])
        rn = tmp * acc
        radon_conc[ii] = rn
        acc -= rn
    radon_conc[N-1] = acc
    return radon_conc

@cython.boundscheck(False)
cpdef int inverse_transform_radon_concs_inplace(np.ndarray[np.double_t, ndim=1] p):
    cdef int N = p.shape[0]
    cdef double acc = exp(p[0])
    cdef int ii = 0
    cdef double tmp = 0.0
    cdef double rn = 0.0

    for ii in range(N - 1):
        tmp = transform_unconstrained_to_constrained(p[ii+1])
        rn = tmp * acc
        p[ii] = rn
        acc -= rn
    p[N-1] = acc
    return 0

def transform_parameters(np.ndarray[np.double_t, ndim=1] p, parameters):

    cdef double lb
    cdef double ub

    cdef int nhyper = parameters['nhyper']
    cdef int nstate = parameters['nstate']
    lower_bounds = parameters['varying_parameters_lower_bound']
    upper_bounds = parameters['varying_parameters_upper_bound']
    # state variables: bounded by zero, take log
    state = p[:nstate]
    logstate = np.log(state)
    # TODO: make bounds etc configurable
    # 'Q_external', 'Q', 'rs', 'lamp', 't_delay', 'eff'
    for lb, ub in zip(lower_bounds, upper_bounds):
        if isfinite(lb) and isfinite(ub):
            pass
        elif isfinite(ub):
            pass
        elif isfinite(lb):
            assert False # not implemented
        else:
            # copy, no constraints on variable
            pass

    hyper = p[nstate:nstate+nhyper]
    Q_external, Q, rs, lamp, t_delay, eff = hyper
    Q_external = np.log(Q_external)
    Q = np.log(Q)
    rs = transform_constrained_to_unconstrained(rs)
    lamp = np.log(lamp)
    t_delay = t_delay
    eff = transform_constrained_to_unconstrained(eff)
    hyper = np.array( [Q_external, Q, rs, lamp, t_delay, eff] )
    #
    if len(p) > nstate+nhyper:
        radon_conc_p = transform_radon_concs(p[nstate+nhyper:])
    else:
        radon_conc_p = np.array([])

    x = np.r_[logstate, hyper, radon_conc_p]

    return x
