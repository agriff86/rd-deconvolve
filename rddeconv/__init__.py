# -*- coding: utf-8 -*-
"""Correct (deconvolve) the output from two-filter radon detectors"""

from rddeconv import metadata

__version__ = metadata.version
__author__ = metadata.authors[0]
__license__ = metadata.license
__copyright__ = metadata.copyright


from .deconvolve import deconvolve_dataframe
from .deconvolve import deconvolve_dataframe_in_chunks
import pymc3
import theano

# this is a helpful constant
from .theoretical_model import lamrn

from .util import standard_parameters_700L
from .util import standard_parameters_1500L
from .util import load_standard_csv
