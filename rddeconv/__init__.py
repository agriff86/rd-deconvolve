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
import logging
import logzero
import theano


# log_format = '%(color)s[%(levelname)1.1s %(process)d %(asctime)s %(module)s:%(lineno)d]%(end_color)s %(message)s'
log_format = (
    "%(color)s[%(levelname)1.1s PID=%(process)d %(asctime)s]%(end_color)s %(message)s"
)

formatter = logzero.LogFormatter(fmt=log_format)
logzero.setup_default_logger(formatter=formatter)

# setup the pymc3 logger to use the same settings as the main one
pymc3_logger = logzero.setup_logger(
    "pymc3", disableStderrLogger=True, formatter=formatter
)

# setup the theano logger to silence INFO messages (and use otherwise similar settings)

#theano_logger = logzero.setup_logger(
#    "theano", disableStderrLogger=True, formatter=formatter
#)
#theano_logger.setLevel(logging.WARNING)

# silence compilelock messages
_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.WARNING)

# this is a helpful constant
from .theoretical_model import lamrn

from .util import standard_parameters_700L
from .util import standard_parameters_1500L
from .util import load_standard_csv
