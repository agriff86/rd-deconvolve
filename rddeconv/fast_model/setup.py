# build this with: python setup.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

boost_include_dirs = ['/home/agf/sw/boost_1_57_0/']

setup(
    name = "fast_detector",
    ext_modules = cythonize('*.pyx'),
    include_dirs=[np.get_include()] + boost_include_dirs
)

# ... __init__.py changes where setup puts the .so file
import os
os.rename('fast_model/fast_detector.so', 'fast_detector.so')
