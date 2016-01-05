# run this using py.test from top-level directory
# ref: https://pytest.org/latest/getting-started.html
from __future__ import print_function, division
from theoretical_model import test_detector_model
import fast_detector
import numpy as np

REL_TOLERENCE = 1e-4

def test_c_vs_python_implementation():
    df = test_detector_model()
    relative_error = (df['count rate, c impl'] -
                      df['count rate']).abs().max() / df['count rate'].max()
    print('C vs Python relative error:', relative_error)
    assert relative_error < REL_TOLERENCE

def test_inplace_transform():
    N = 500
    nstate=0
    nhyper=0
    p = np.random.random_sample(N)
    orig = p[nstate+nhyper:].copy()
    fast_detector.transform_radon_concs_inplace(p[nstate+nhyper:])
    assert np.allclose(fast_detector.inverse_transform_radon_concs(p), orig)

def test_transform_round_trip():
    N = 500
    nstate=0
    nhyper=0
    p = np.random.random_sample(N)
    orig = p[nstate+nhyper:].copy()
    fast_detector.transform_radon_concs_inplace(p[nstate+nhyper:])
    assert np.allclose(p, fast_detector.transform_radon_concs(orig))

def test_inplace_transform_round_trip():
    N = 500
    nstate=0
    nhyper=0
    p = np.random.random_sample(N)
    orig = p[nstate+nhyper:].copy()
    fast_detector.transform_radon_concs_inplace(p[nstate+nhyper:])
    fast_detector.inverse_transform_radon_concs_inplace(p[nstate+nhyper:])
    assert np.allclose(p, orig)


def test_non_inplace_transform():
    N = 500
    nstate=0
    nhyper=0
    p = np.random.random_sample(N)
    orig = p[nstate+nhyper:].copy()
    p = fast_detector.transform_radon_concs(p[nstate+nhyper:])
    assert np.allclose(fast_detector.inverse_transform_radon_concs(p), orig)
