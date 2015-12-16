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

import theoretical_model as tm

# filter parameters - 635 steel mesh
# http://www.twpinc.com/tcpdf/pdf/create.php?prod_id=4184
Mesh_info = """Mesh
635 x 635 per inch

Wire Diameter
0.0008 inches
0.0203 mm

Opening Size
0.0008 inches
0.02 mm

Overall Thickness
0.0016 inches
0.04064 mm

Weight
0.14 lb/sq foot
0.068 kg/sq meter

Opening Area
24%
Percentage

Weave Type
Twill"""

wire_diameter = 0.02e-3
screen_thickness = wire_diameter*2.0
area_density = 0.68e3 # mass per unit area, kg/m2
rho_ss = 8e6 # from google search - 8 g/cm3
solid_fraction = area_density/rho_ss/screen_thickness

#expecting a solid fraction of about 0.3
#this would mean an area density of
expected_area_density = 0.3 * screen_thickness * rho_ss

# compute expected density from thread count and diameter
threadlen = 1.0 # m
threads = 1000/25.4 * 635 * 2
volume = np.pi * (wire_diameter/2.0)**2 * threads * threadlen
solid_fraction_computed = volume / screen_thickness
area_density_computed = volume*rho_ss
# ok, just set solid fraction
solid_fraction = 0.39


# From Holub, R. and Knutson, E.: Measuring polonium-218 diffusion-coefficient spectra using multiple wire screens, in Radon and its decay products: Occurrence, properties, and health effects., 1987.  (also see Frey 1981, in Science, for breakdown by netural/charged)
D = 0.08 / 100 / 100

# To convert from D to particle diameter, look at Eq 11:
# Porstendörfer, J.: Radon: Measurements related to dose, Environment International, 22, Supplement 1, 563–583, doi:10.1016/S0160-4120(96)00158-4, 1996.




# internal flow rate
Q = 0.0122 # specification

# internal flow rate from vent captor
# u_mean = 0.80 * u_centre (according to scott's measurements)
# pipe diameter = 50 mm
u_centre = 7.1  # measured: 7.14 +/- 0.36
pipe_area = np.pi * (25e-3)**2
u_mean = 0.80 * u_centre
Q_pipe = u_mean * pipe_area



# cross-sectional area of the exposed part of the screen
screen_area = 0.02625
U0 = Q/screen_area


def estimate_screen_penetration(Q):
    """
    screen penetration estimate, based on various assumptions about detector
    etc.
    """
    wire_diameter = 0.02e-3
    screen_thickness = wire_diameter*2.0
    area_density = 0.68e3 # mass per unit area, kg/m2
    rho_ss = 8e6 # from google search - 8 g/cm3
    solid_fraction = area_density/rho_ss/screen_thickness

    #expecting a solid fraction of about 0.3
    #this would mean an area density of
    expected_area_density = 0.3 * screen_thickness * rho_ss

    # compute expected density from thread count and diameter
    threadlen = 1.0 # m
    threads = 1000/25.4 * 635 * 2
    volume = np.pi * (wire_diameter/2.0)**2 * threads * threadlen
    solid_fraction_computed = volume / screen_thickness
    area_density_computed = volume*rho_ss
    # ok, just set solid fraction
    solid_fraction = 0.39


    screen_area = 0.02625
    U0 = Q/screen_area
    D = 0.08 / 100 / 100
    p0_neutral = tm.screen_penetration(U0, D, solid_fraction, wire_diameter, screen_thickness)
    p0_positive = tm.screen_penetration(U0, 0.03e-4, solid_fraction, wire_diameter, screen_thickness)

    # assume 50% charged Particles
    return (p0_neutral + p0_positive)/2.0


p0_neutral = tm.screen_penetration(U0, D, solid_fraction, wire_diameter, screen_thickness)
p0_positive = tm.screen_penetration(U0, 0.03e-4, solid_fraction, wire_diameter, screen_thickness)

relative_change = (1-p0_neutral)/(1-p0_positive)

U = np.linspace(0.1, 1, 1000)
P = tm.screen_penetration(U, D, solid_fraction, wire_diameter, screen_thickness)
C = 1-P

f, ax = plt.subplots()
ax.plot(U,C, label='D=0.08 cm2/s')
D = 0.03 / 100 / 100
P = tm.screen_penetration(U, D, solid_fraction, wire_diameter, screen_thickness)
C = 1-P
ax.plot(U,C, label='D=0.03 cm2/s')

ax.set_xlabel('Flow velocity (m/s)')
ax.set_ylabel('Screen capture fraction')
ax.axvline(U0)
ax.legend()

eff_positive = 1-tm.screen_penetration(U0, 0.03e-4, solid_fraction, wire_diameter, screen_thickness)
eff_neutral = 1-tm.screen_penetration(U0, 0.08e-4, solid_fraction, wire_diameter, screen_thickness)

print('at U0, d=0.03 cm2/s [positive clusters], screen cap eff is', eff_positive)
print('at U0, d=0.08 cm2/s [neutral clusters], screen cap eff is', eff_neutral)
print('min eff, 88% positive clusters is', eff_positive*0.88 + eff_neutral*(1-0.88))

plt.show()
