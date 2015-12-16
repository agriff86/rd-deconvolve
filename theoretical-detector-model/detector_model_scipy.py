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

def rate_of_change(Y, t, Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob,
                    Nrn_ext=Nrn_ext_spike):
    # unpack state vector
    Nrnd, Nrn, Na, Nb, Nc, Fa, Fb, Fc = Y
    # effect of delay and tank volumes
    dNrnddt = Q_external / V_delay * (Nrn_ext(t) - Nrnd) - Nrnd*lamrn
    dNrndt = Q_external / V_tank * (Nrnd - Nrn) - Nrn*lamrn
    # compute rate of change of each state variable
    dNadt = Nrn*lamrn - Na*(lama+lamp)  - Q*Na #- Na*Q_external/V_tank
    dNbdt = Na*lama - Nb*(lamb+lamp) - Q*Nb    #- Nb*Q_external/V_tank
    dNcdt = Nb*lamb - Nc*(lamc+lamp) - Q*Nc    #- Nc*Q_external/V_tank
    dFadt = Q*rs*Na - Fa*lama
    dFbdt = Q*rs*Nb - Fb*lamb + Fa*lama * (1.0-recoil_prob)
    dFcdt = Q*rs*Nc - Fc*lamc + Fb*lamb * (1.0-recoil_prob)
    return np.array([dNrnddt, dNrndt, dNadt, dNbdt, dNcdt, dFadt, dFbdt, dFcdt])

#initial conditions
Y0 = np.zeros(8)
t  = np.arange(0, 3600*5, 60)   # time grid

parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob
soln = odeint(rate_of_change, Y0, t, args=parameters)


df = pd.DataFrame(index=t/60.0, data=soln)
df.columns = 'Nrnd,Nrn,Na,Nb,Nc,Fa,Fb,Fc'.split(',')
df['Nrn_ext'] = [Nrn_ext(itm) for itm in t]
df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)

f, ax = plt.subplots()
df[['Nrn_ext','Nrnd','Nrn']].plot(ax=ax)


f, ax = plt.subplots()
df[['Na','Nb','Nc']].plot(ax=ax)
f, ax = plt.subplots()
df[['Fa','Fb','Fc']].plot(ax=ax)
f, ax = plt.subplots()
df[['count rate']].plot(ax=ax)


def detector_model(t, Y0 = np.zeros(8), 
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
    parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob, Nrn_ext
    soln = odeint(rate_of_change, Y0, t-t_delay, args=parameters, hmax=1.0)
    df = pd.DataFrame(index=t/60.0, data=soln)
    df.columns = 'Nrnd,Nrn,Na,Nb,Nc,Fa,Fb,Fc'.split(',')
    df['Nrn_ext'] = [Nrn_ext(itm) for itm in t]
    df['count rate'] = eff*(df.Fa*lama + df.Fc*lamc)
    return df


# perturb some things
df = detector_model(t)
dfp = detector_model(t, Q=Q*2)
df['count rate Q*2'] = dfp['count rate']
dfp = detector_model(t, rs=1.0)
df['count rate rs=1'] = dfp['count rate']
dfp = detector_model(t, lamp=np.log(2.0)/40.0)
df['count rate plateout=40s'] = dfp['count rate']
dfp = detector_model(t, Q_external=Q_external*2)
df['count rate Q_external*2'] = dfp['count rate']


cols = [itm for itm in df.columns if 'count rate' in itm]
df[cols].plot()
norm = df['count rate'].max()
for itm in cols:
    df[itm] = df[itm]/norm #df[itm].mean()
df[cols].plot()

for itm in cols:
    df[itm] /= df[itm].mean()
df[cols].plot()


df = detector_model(t)

# compare with experiment
import util
fname = '../data-controlled-test-2/T1Mar15e.CSV'
dfobs = util.load_radon(fname)
f, ax = plt.subplots()
t0 = datetime.datetime(2015,3,17,13)
dt = datetime.timedelta(hours=5)
dt -= datetime.timedelta(minutes=1)
for ii in range(6):
    dfrel = dfobs[t0:t0+dt]
    dfrel.index = (dfrel.index.values - dfrel.index.values[0]) / 1e9 / 60
    dfrel.lld.plot(ax=ax, label=str(ii))
    t0 += datetime.timedelta(days=1)
plt.legend()

df['observed count rate'] = dfrel.lld.values
# normalise
df['observed count rate'] = df['observed count rate'] /               \
                df['observed count rate'].mean() * df['count rate'].mean()

df[['observed count rate', 'count rate']].plot()

plt.show()

#
# ... try to optimise parameter values
#
from scipy.optimize import fmin

def minfunc(x):
    Y0 = np.zeros(8)
    t  = np.arange(0, 3600*5, 60)   # time grid
    recoil_prob, t_delay = x
    parameters = Q, rs, lamp, eff, Q_external, V_delay, V_tank, recoil_prob
    soln = odeint(rate_of_change, Y0, t-t_delay, args=tuple(parameters), hmax=1.0)
    dfs = pd.DataFrame(index=t/60.0, data=soln)
    dfs.columns = 'Nrnd,Nrn,Na,Nb,Nc,Fa,Fb,Fc'.split(',')
    modelled_counts = eff*(dfs.Fa*lama + dfs.Fc*lamc)
    # normalise by experiment
    experiment =  df['observed count rate'].values
    modelled_counts /= modelled_counts.mean()
    modelled_counts *= experiment.mean()
    residsq = (experiment-modelled_counts)**2
    #f,ax = plt.subplots(figsize=[4,2])
    #ax.plot(experiment)
    #ax.plot(modelled_counts)
    #ax.set_title(str(x))
    #plt.show()
    return residsq.sum()

param_opt = fmin(minfunc, np.array([recoil_prob, 60.0]) )

f, ax = plt.subplots()

df = detector_model(t, t_delay=param_opt[1], recoil_prob=param_opt[0])

df['observed count rate'] = dfrel.lld.values
# normalise
df['observed count rate'] = df['observed count rate'] /               \
                df['observed count rate'].mean() * df['count rate'].mean()

df[['observed count rate', 'count rate']].plot(ax=ax)
ax.set_title('optimised parameters')

df = detector_model(t, t_delay=param_opt[1], recoil_prob=param_opt[0], Nrn_ext=Nrn_ext_const)
f, ax = plt.subplots()
df[['count rate']].plot(ax=ax)
ax.set_title('optimised parameters, 1Bq/m3 inflow')

plt.show()

if __name__ == "__main__":
    #code
    pass
