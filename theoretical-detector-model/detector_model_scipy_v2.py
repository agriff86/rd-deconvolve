#!/usr/bin/env python
# coding: utf-8

"""
Scipy model of the 750l radon detector
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

from scipy.integrate import odeint

#
# numerical solution - from
#      http://wiki.scipy.org/Cookbook/Zombie_Apocalypse_ODEINT
#     or
#      http://docs.sympy.org/dev/modules/mpmath/calculus/odes.html#mpmath.odefun

#
# ... define system of equations as vector
# ...   dY/dt = FY
#




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

Q = 800.0 / 60.0 / 1000.0 # from Whittlestone's paper, L/min converted to m3/s
rs = 0.7 # from Whittlestone's paper (screen retention)
lamp = np.log(2.0)/120.0 # from Whittlestone's code (1994 tech report)
eff = 0.33 # Whittlestone's paper

Q_external = 40.0 / 60.0 / 1000.0
V_delay = 200.0 / 1000.0
V_tank = 750.0 / 1000.0

recoil_prob = 0.02

# play with values
#lamp = 0.0


# boundary value
def Nrn_ext_spike(t):
    if t<60.0 and t >=0.0:
        return 1.0 / lamrn
    else:
        return 0.0

def Nrn_ext_const(t):
    # 1 Bq/m3
    return 1.0 / lamrn

Nrn_ext = Nrn_ext_spike



def tank_concentrations(Y, t, Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn):
    #unpack state vector
    Na, Nb, Nc = Y
    dNadt = Nrn*lamrn - Na*(lama+lamp)
    dNbdt = Na*lama - Nb*(lamb+lamp)
    dNcdt = Nb*lamb - Nc*(lamc+lamp)
    return np.array([dNadt, dNbdt, dNcdt])


def calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn):
    Y0 = np.zeros(3)
    tt = V_tank/Q
    t = np.linspace(0,tt,5)
    parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn
    soln = odeint(tank_concentrations, Y0, t, args=parameters)
    return soln[-1,:]
    #return t, soln  #for testing


def rate_of_change(Y, t, Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob,
                    Nrn_ext=Nrn_ext_spike):
    # unpack state vector
    Nrnd, Nrn, Fa, Fb, Fc = Y
    # effect of delay and tank volumes
    dNrnddt = Q_external / V_delay * (Nrn_ext(t) - Nrnd) - Nrnd*lamrn
    dNrndt = Q_external / V_tank * (Nrnd - Nrn) - Nrn*lamrn
    # Na, Nb, Nc from steady-state in tank
    Na, Nb, Nc = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn)
    # compute rate of change of each state variable
    dFadt = Q*rs*Na - Fa*lama
    dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb * (1.0-recoil_prob)
    return np.array([dNrnddt, dNrndt, dFadt, dFbdt, dFcdt])


def rate_of_change_opt(Y, t, Q, rs, lamp, eff, Q_external, V_delay, V_tank,
                       recoil_prob, Na_factor, Nb_factor, Nc_factor,
                       Nrn_ext=Nrn_ext_spike):
    # unpack state vector
    Nrnd, Nrn, Fa, Fb, Fc = Y
    # effect of delay and tank volumes
    dNrnddt = Q_external / V_delay * (Nrn_ext(t) - Nrnd) - Nrnd*lamrn
    dNrndt = Q_external / V_tank * (Nrnd - Nrn) - Nrn*lamrn
    # Na, Nb, Nc from steady-state in tank
    # Na, Nb, Nc = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn)
    Na = Na_factor * Nrn
    Nb = Nb_factor * Nrn
    Nc = Nc_factor * Nrn
    # compute rate of change of each state variable
    dFadt = Q*rs*Na - Fa*lama
    dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb
    return np.array([dNrnddt, dNrndt, dFadt, dFbdt, dFcdt])


#initial conditions
Y0 = np.zeros(5)
t  = np.arange(0, 3600*5, 60)   # time grid

parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob
soln = odeint(rate_of_change, Y0, t, args=parameters)


df = pd.DataFrame(index=t/60.0, data=soln)
df.columns = 'Nrnd,Nrn,Fa,Fb,Fc'.split(',')
df['Nrn_ext'] = [Nrn_ext(itm) for itm in t]
df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)

f, ax = plt.subplots()
df[['Nrn_ext','Nrnd','Nrn']].plot(ax=ax)


# add computed Na,Nb,Nc
df['Na'] = 0.
df['Nb'] = 0.
df['Nc'] = 0.
for ix, itm in df.iterrows():
    Nrn = itm['Nrn']
    Na,Nb,Nc = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn)
    df.Na[ix] = Na
    df.Nb[ix] = Nb
    df.Nc[ix] = Nc
#end of adding Na Nb Nc

f, ax = plt.subplots()
df[['Na','Nb','Nc']].plot(ax=ax)
f, ax = plt.subplots()
df[['Fa','Fb','Fc']].plot(ax=ax)
f, ax = plt.subplots()
df[['count rate']].plot(ax=ax)



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
                      Nrn_ext=Nrn_ext_spike):
    Na_factor, Nb_factor, Nc_factor = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn=1.0)
    parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Na_factor, Nb_factor, Nc_factor, Nrn_ext
    soln = odeint(rate_of_change_opt, Y0, t-t_delay, args=parameters, hmax=1.0)
    df = pd.DataFrame(index=t/60.0, data=soln)
    df.columns = 'Nrnd,Nrn,Fa,Fb,Fc'.split(',')
    df['Nrn_ext'] = [Nrn_ext(itm) for itm in t]
    df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)
    df['Na'] = 0.
    df['Nb'] = 0.
    df['Nc'] = 0.
    for ix, itm in df.iterrows():
        Nrn = itm['Nrn']
        Na,Nb,Nc = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn)
        df.Na[ix] = Na
        df.Nb[ix] = Nb
        df.Nc[ix] = Nc
    return df


# perturb some things
df = detector_model(t)
df_standard = df.copy()
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

cols = [itm for itm in df.columns if 'count rate' in itm]
df[cols].plot()
norm = df['count rate'].max()
for itm in cols:
    df[itm] = df[itm]/norm #df[itm].mean()
df[cols].plot()

for itm in cols:
    df[itm] /= df[itm].mean()
df[cols].plot()

df[['count rate plateout=20s', 'count rate']].plot()

df = detector_model(t)

# compare with experiment
import util
fnames = ['../data-controlled-test-2/T1Mar15e.CSV',
          '../data-controlled-test-2/T1Apr15e.CSV']

dfobs = [util.load_radon(itm) for itm in fnames]
dfobs = pd.concat(dfobs)

#
# ... how much do the calibration peaks vary?
#
# 1-min injections (expected to vary a bit)
ts0 = datetime.datetime(2015,3,18,13)
nspikes = 9
cp = datetime.timedelta(hours=6)
oneday = datetime.timedelta(days=1)

totcounts = [dfobs[ts0+ii*oneday:ts0+ii*oneday+cp].lld.sum() 
                for ii in range(nspikes)]
totcounts_spike = np.array(totcounts)

# one hour injections (from flushed source, less variation expected)
ti0 = datetime.datetime(2015,3,27,13)
ninj = 6
totcounts = [dfobs[ti0+ii*oneday:ti0+ii*oneday+cp].lld.sum() 
                for ii in range(ninj)]
totcounts_inj = np.array(totcounts)

f, ax = plt.subplots()
ax.plot(totcounts_spike/totcounts_spike.mean(), label='spikes')
ax.plot(totcounts_inj/totcounts_inj.mean(), label='1h injection')
ax.legend()


t0 = datetime.datetime(2015,3,17,13)
dt = datetime.timedelta(hours=5)
dt -= datetime.timedelta(minutes=1)

f, ax = plt.subplots()
for ii in range(6):
    dfrel = dfobs[t0:t0+dt]
    dfrel.index = (dfrel.index.values - dfrel.index.values[0]) / 1e9 / 60
    dfrel.lld.plot(ax=ax, label=str(ii))
    t0 += datetime.timedelta(days=1)
plt.legend()

# like prev, but normalise
t0 = datetime.datetime(2015,3,18,13)

f, ax = plt.subplots()
for ii in range(8):
    dfrel = dfobs[t0:t0+dt]
    dfrel.lld /= dfrel.lld.mean()
    dfrel.index = (dfrel.index.values - dfrel.index.values[0]) / 1e9 / 60
    dfrel.lld.plot(ax=ax, label=str(ii))
    t0 += datetime.timedelta(days=1)
plt.legend()



df['observed count rate'] = dfrel.lld.values
# normalise
df['observed count rate'] = df['observed count rate'] /               \
                df['observed count rate'].mean() * df['count rate'].mean()

# model with parameters from W&Z's "new version"
df_nv = detector_model(t, rs=0.987, recoil_prob=0,
                       lamp=np.log(2)/(6*60), #for a ~10% loss to plateout in 60sec
                        V_tank=730/1000.0)

df['"new version" count rate'] = df_nv['count rate'] / \
                        df_nv['count rate'].mean() * df['count rate'].mean()

df[['observed count rate', 'count rate', '"new version" count rate']].plot()




plt.show()

#
# ... try to optimise parameter values
#

def fit_to_obs(df):
    dfrel = df.copy()

    from scipy.optimize import minimize

    Na_factor, Nb_factor, Nc_factor = calc_NaNbNc(Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn=1.0)
    def minfunc(x):
        Y0 = np.zeros(5)
        t  = np.arange(0, 3600*5, 60)   # time grid
        Q_external_scale, t_delay, recoil_prob, lamp_scale = x
        # link screen efficiency to recoil_prob
        #rs = 1 - 2*recoil_prob * corr #corr was a fit parameter, seemed to be close to 1
        rs = 1 - 2*recoil_prob
        
        #recoil_prob = 0.02
        
        parameters = Q, rs, lamp*lamp_scale, eff, Q_external*Q_external_scale, V_delay, V_tank, recoil_prob, Na_factor, Nb_factor, Nc_factor, Nrn_ext

        soln = odeint(rate_of_change_opt, Y0, t-t_delay, args=tuple(parameters), hmax=30)
        dfs = pd.DataFrame(index=t/60.0, data=soln)
        dfs.columns = 'Nrnd,Nrn,Fa,Fb,Fc'.split(',')
        modelled_counts = eff*(dfs.Fa*lama + dfs.Fc*lamc)
        # normalise by experiment
        experiment =  df['observed count rate'].values
        modelled_counts = modelled_counts / modelled_counts.mean() * experiment.mean()
        residsq = (experiment-modelled_counts)**2
        #f,ax = plt.subplots(figsize=[4,2])
        #ax.plot(experiment)
        #ax.plot(modelled_counts)
        #ax.set_title(str(x))
        #plt.show()
        return residsq.mean()
    #f, ax = plt.subplots()
    #df['observed count rate'].plot()
    #plt.show()

    res = minimize(minfunc, np.array([1.0, 60.0, 0.1, 100]) ,
                        options=dict(maxiter=500), method='L-BFGS-B',
                        bounds=[(0.8, 1.2),
                                (0, 120),
                                (0, 0.0),
                                (0.01, 10000)])
    param_opt = res.x
    print(res)

    f, ax = plt.subplots()

    #df = detector_model(t, t_delay=param_opt[1], Q_external=param_opt[0]*Q_external, recoil_prob=param_opt[2], rs = 1 - 2*param_opt[2] * param_opt[3])
    df = detector_model(t, t_delay=param_opt[1], Q_external=param_opt[0]*Q_external, recoil_prob=param_opt[2], rs = 1 - 2*param_opt[2])

    df['observed count rate'] = dfrel['observed count rate']
    # normalise
    df['observed count rate'] = df['observed count rate'] /               \
                    df['observed count rate'].mean() * df['count rate'].mean()

    df[['observed count rate', 'count rate']].plot(ax=ax)
    ax.set_title('optimised parameters')
    f, ax = plt.subplots()
    (df['observed count rate']-df['count rate']).plot(ax=ax)
    ax.set_title('optimised parameters, residuals')
    return param_opt

t0 = datetime.datetime(2015,3,18,13)
results = []
for ii in range(8):
    dfrel = dfobs[t0:t0+dt]
    df['observed count rate'] = dfrel.lld.values
    param_opt = fit_to_obs(df)
    t0 += datetime.timedelta(days=1)
    plt.show()
    results.append(param_opt)


#extract the peak count rate from each
t0 = datetime.datetime(2015,3,18,13)
peak_counts = []
for ii in range(8):
    peak_counts.append(dfobs[t0:t0+dt].lld.max())
    t0 += datetime.timedelta(days=1)

df['observed count rate'] = dfrel.lld.values


f, ax = plt.subplots()
df = detector_model(t, rs=0.987, recoil_prob=0,
                       lamp=np.log(2)/(6*60), #for a ~10% loss to plateout in 60sec
                        Nrn_ext=Nrn_ext_const,
                        V_tank=730/1000.0)
df[['count rate']].plot(ax=ax, color='k')
dft = df.tail(1)
print("Screen alpha:", (dft.Fa*lama + dft.Fc*lamc).values[0] )
print("counts per Bq/m3 radon:", eff*(dft.Fa*lama + dft.Fc*lamc).values[0] )

for param_opt in results:
    df = detector_model(t, t_delay=param_opt[1],
                        Q_external=param_opt[0]*Q_external,
                        recoil_prob=param_opt[2],
                        Nrn_ext=Nrn_ext_const)

    df[['count rate']].plot(ax=ax)

ax.set_title('optimised parameters, 1Bq/m3 inflow')



plt.show()

if __name__ == "__main__":
    #code
    pass
