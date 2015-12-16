#!/usr/bin/env python

import fast_detector
import numpy as np

#
# ... test the detector model
#

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

params = fast_detector.parameter_array_from_dict(parameters)
print(params)

Ntimes = 150
Nstate = fast_detector.N_state
x0 = np.zeros(Nstate)
rn = np.zeros(Ntimes)
rn[0] = 1.0
timestep = 60.0
interpolation_mode = 0
x_history = fast_detector.detector_model(timestep, interpolation_mode, rn, x0, params)
print(x_history.shape)
#print(x_history)

#
# ... test the linear interpolation methods
#
print(fast_detector.linear_interpolation_test(2.5, 1, np.array([-1, 0,1.,2.0])))

timestep = 60
x = np.arange(0, 10*timestep, timestep, dtype=np.float)
y = x*2
y = np.zeros(len(x))
y[1] = 300
xi = np.linspace(-10, 300, 1000)
yi = [fast_detector.linear_interpolation_test(itm, timestep, y) for itm in xi]
yinn = [fast_detector.linear_interpolation_test(itm, timestep, y, mode=0) for itm in xi]
yi = np.array(yi)
yinn = np.array(yinn)


import matplotlib as mpl
import matplotlib.pyplot as plt
plt.plot(x,y,'o')
plt.plot(xi,yi)
plt.plot(xi,yinn)
plt.gca().set_ylim([-100, 600])
plt.show()

