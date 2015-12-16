"""
measure speed of various things (copy-and-paste into ipython)
"""

%matplotlib inline
from fast_model.fast_detector import calc_steady_state, calc_NaNbNc
import theoretical_model as tm
import numpy as np
Q = 800.0 / 60.0 / 1000.0
rs = 0.7
lamp = np.log(2.0)/120.0
eff = 0.33 # Whittlestone's paper
Q_external = 40.0 / 60.0 / 1000.0
V_delay = 200.0 / 1000.0
V_tank = 750.0 / 1000.0
t_delay = 60.0
recoil_prob = 0.02
Nrn = 1/tm.lamrn
p1 = (Nrn, Q, rs, lamp, V_tank, recoil_prob)
p2 = (Nrn, Q, rs, lamp, V_tank, recoil_prob, eff)


%timeit tm.calc_steady_state(*p1)
%timeit calc_steady_state(*p2)

