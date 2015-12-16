#!/usr/bin/env python
# coding: utf-8

"""
Sympy model of the 750l radon detector
"""

# These two imports are intended to allow the code to run on both 3 and 2.7
#ref: http://python-future.org/quickstart.html
from __future__ import (absolute_import, division,
                        print_function)

import glob
import datetime
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sympy

sympy.init_printing(use_unicode=True)

#from micromet.map import create_map
#from micromet.plot import figure_template, fancy_ylabel

#
# ... define symbols
#
Na, Nb, Nc = sympy.symbols('Na Nb Nc', cls=sympy.Function)
Nrn = sympy.symbols('Nrn')
lama, lamb, lamc, lamrn, lamp = sympy.symbols('lama lamb lamc lamrn lamp')
Fa, Fb, Fc = sympy.symbols('Fa Fb Fc')
Q, e, rs, t = sympy.symbols('Q e rs t')

#
# ... define equations
#
eqs = (
sympy.Eq( Na(t).diff(t) , Nrn*lamrn - Na(t)*(lama+lamp) ),
sympy.Eq( Nb(t).diff(t) , Na(t)*lama - Nb(t)*(lamb+lamp) ),
sympy.Eq( Nc(t).diff(t) , Nb(t)*lamb - Nc(t)*(lamc+lamp) )
)

#
# ... solve DE and find values of constants C1,C2,C3
#
C1,C2,C3 = sympy.symbols('C1 C2 C3')

#
# ... I reckon this should work, but it doesn't
#

#soln = sympy.dsolve(eqs)
#
#C1 = sympy.solve(soln[0], C1)[0].subs(t,0).subs(Na(0),0)
#C2 = sympy.solve(soln[1], C2).subs(t,0).subs(Nb(0),0)
#C3 = sympy.solve(soln[2], C3).subs(t,0).subs(Nc(0),0)


eq0soln = sympy.dsolve(eqs[0])
C1_val = sympy.solve(eq0soln, C1)[0].subs(t,0).subs(Na(0),0)
eq0s = eq0soln.subs(C1, C1_val)


assert(False)

eqs2 = (
sympy.Eq( Fa(t).diff(t) , Q*rs*Na(t) - Fa(t)*lama ),
sympy.Eq( Fb(t).diff(t) , Q*rs*Nb(t) - Fb(t)*lamb + Fa(t)*lama),
sympy.Eq( Fc(t).diff(t) , Q*rs*Nc(t) - Fc(t)*lamc + Fb(t)*lamb),
)


#
# ... numerical values
#
radon_chain_half_life = np.array([3.82*24*3600, #Rn-222 (3.82 d)
                               3.05*60,      #Po-218 (3.05 min)
                               26.8*60,      #Pb-214 (26.8 min)
                               19.9*60       #Bi-214 (19.9 min)
                               ])
radon_chain_num_alpha = np.array([1, 1, 0, 1])
radon_chain_name = [ 'Rn-222', 'Po-218', 'Pb-214', 'Bi-214']
radon_chain_lambda = np.log(2.0)/radon_chain_half_life

lamrn_n = radon_chain_lambda[0]
lama_n = radon_chain_lambda[1]
lamb_n = radon_chain_lambda[2]
lamc_n = radon_chain_lambda[3]

Q = 800.0 / 60.0 / 1000.0 # from Whittlestone's paper, L/min converted to m3/s
rs = 0.7 # from Whittlestone's paper (screen retention)
lamp_n = np.log(2.0)/120.0 # from Whittlestone's code (1994 tech report)
eff = 0.33 # Whittlestone's paper

Q_external = 40.0 / 60.0 / 1000.0
V_delay = 200.0 / 1000.0
V_tank = 750.0 / 1000.0
#transit time in tank
tt = V_tank / Q

#
# ... treat flow through tank as laminar (eqs 1,2,3)
#
C1, Nrn0 = sympy.symbols('C1,Nrn0')
# make Nrn(t) a constant
eq0 = eqs[0].subs(Nrn(t), Nrn0)
eq0_soln = sympy.dsolve(eqs[0])

substitutions = [(lamrn, lamrn_n), (lama, lama_n), (lamb, lamb_n),
                  (lamp, lamp_n), (C1,0), (Nrn(t), Nrn0)]

eq0_soln.subs(substitutions).doit()
