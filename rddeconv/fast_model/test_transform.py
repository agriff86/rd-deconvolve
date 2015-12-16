import fast_detector
import numpy as np


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



this_stuff_is_useful_in_an_ipython_session = """
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import fast_model

nhyper=0
nstate = 0
N = 500
p = (np.random.random_sample(N) + 0.01)*1000
orig = p[nstate+nhyper:].copy()
fast_model.fast_detector.transform_radon_concs_inplace(p[nstate+nhyper:])
plt.plot(p)
plt.plot(fast_model.fast_detector.transform_radon_concs(orig))
"""
