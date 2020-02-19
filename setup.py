# -*- coding: utf-8 -*-
#
# note -- for use in-place use
# python setup.py build_ext --inplace
from __future__ import print_function


import os
import sys

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
# Add the current directory to the module search path.
sys.path.insert(0, os.path.abspath('.'))

## Constants
CODE_DIRECTORY = 'rddeconv'

## Handle boost libraries installed in anaconda environment
BOOST_INCLUDE_DIRS = []
try:
    CONDA_PREFIX = os.environ['CONDA_PREFIX']
    BOOST_INCLUDE_DIRS.append( os.path.join(CONDA_PREFIX, 'include'))
except KeyError:
    pass


setup(
    name = "rddeconv",
    ext_modules = cythonize(os.path.join(CODE_DIRECTORY, '*.pyx')),
    include_dirs=[np.get_include()] + BOOST_INCLUDE_DIRS,
    packages = [CODE_DIRECTORY],
)
