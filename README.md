Radon Detector Deconvolution (`rd-deconvolve`)
==============================================

This can be used to correct the output from a dual-flow-loop two-filter radon
detector for the slow time response (~45 minutes) of the detector. It works
by deconvolving the detector response from the measured signal.


Instructions
============

This is a Python library which has grown from experimentation.  It does not have a well-planned or stable API.  The recommended way to use it is to 
1. clone the repository, 
2. setup a Python environment using conda, 
3. install `rd-deconvolve` into the environment, and
4. run or adapt one of the examples in the `examples` subdirectory

Running an example
------------------

These steps have been tested on Ubuntu 18.04.

1. install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)
2. Clone this git repository: 
```sh
git clone https://github.com/agriff86/rd-deconvolve.git
```
3. build the environment
```sh
cd rd-deconvolve
./build_conda_env
```
4. activate the new environment
```sh
conda activate ./env
```
5. compile the fast detector model
```sh
python setup.py build_ext --inplace
```
6. run an example
```sh
cd examples/generic-1500l-deconvolution
# reads and pre-processes raw data
python clean_data.py
# outputs deconvolved data to ./data-processed
python run_deconv.py
```

Algorithm
==========
The deconvolution routine, based on [emcee](https://emcee.readthedocs.io/en/stable/user/sampler/) and the [Boost ODE integrator](https://www.boost.org/doc/libs/1_75_0/libs/numeric/odeint/doc/html/index.html), is described in [this paper](https://doi.org/10.5194/amt-9-2689-2016).  Since the paper was released, we have also experimented with a backend using [PyMC3](https://docs.pymc.io/).


Licenses
========

This library is release under the MIT/X11 license.


Authors
=======

* Alan Griffiths
