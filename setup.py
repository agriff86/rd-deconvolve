# -*- coding: utf-8 -*-
#
# note -- for use in-place use
# python setup.py build_ext --inplace
# ref: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

import os
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

## Constants
CODE_DIRECTORY = "rddeconv"

## Handle boost libraries installed in anaconda environment
## TODO: perhaps copy the required headers into the project?  It's about 30Mb of code though.
BOOST_INCLUDE_DIRS = []
try:
    CONDA_PREFIX = os.environ["CONDA_PREFIX"]
    BOOST_INCLUDE_DIRS.append(os.path.join(CONDA_PREFIX, "include"))
except KeyError:
    # assume that the boost headers have been installed by the user and are located in a sensible place
    pass

include_dirs = [np.get_include()] + BOOST_INCLUDE_DIRS

extensions = [
    Extension(
        ".".join([CODE_DIRECTORY, "fast_detector"]),
        sources=[os.path.join(CODE_DIRECTORY, "fast_detector.pyx")],
        include_dirs=include_dirs,
    )
]

# single-source versioning https://packaging.python.org/guides/single-sourcing-package-version/
metadata_file = os.path.join(CODE_DIRECTORY, "metadata.py")
version = None
for line in open(metadata_file, "rt"):
    if "=" in line:
        k, v = line.split("=")
        if k.strip() == "version":
            version = v.strip().strip("'")

assert version is not None

setup(
    name="rddeconv",
    version=version,
    description="Deconvolution routines for ANSTO radon detectors",
    author="Alan Griffiths",
    author_email="alan.griffiths@ansto.gov.au",
    license="LICENSE.txt",
    ext_modules=cythonize(extensions),
    packages=[CODE_DIRECTORY],
    zip_safe=False,
    install_requires=[
        "numpy",
        "pandas", # >= 1.0
        "pymc3",
        "emcee", # >= 3.0
        "logzero",
        "theano",
        "scipy",
        "joblib",
    ],
)
