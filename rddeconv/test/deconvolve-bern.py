#!/usr/bin/env python
# coding: utf-8

"""
Deconvolve bern radon obs
"""

# These two imports are intended to allow the code to run on both 3 and 2.7
#ref: http://python-future.org/quickstart.html
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import glob
import datetime
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import pygrib

#from micromet.map import create_map
#from micromet.plot import figure_template, fancy_ylabel

import util

def load_radon(fname, sheet_name):
    """
    load radon data from excel spreadsheet
    """
    df = pd.read_excel(fname, sheet_name, parse_cols="A,D,E,K")
    
    df.columns = [itm.lower() for itm in df.columns]
    
    if False:
        
        #hod is HH:MM, but sometimes gets interpreted as a timestamp
        def fixtime(itm):
            if hasattr(itm,'time'):
                itm = itm.time()
            return itm
        df.hod = [fixtime(itm) for itm in df.hod]


        time = [ datetime.datetime.combine(
                    datetime.datetime(int(itm[1]['year']),
                                      int(itm[1]['mon']),
                                      int(itm[1]['dom'])),
                    itm[1]['hod'])
                                      for itm in df.iterrows() ]
    else:
        #dates in these files are a mess.  Just generate our own time axis
        Ntime = len(df)
        y,m,d = df.iloc[0][['year','mon','dom']]
        t0 = datetime.datetime(int(y), int(m), int(d))
        time = [ t0 + datetime.timedelta(minutes=30)*ii for ii in range(Ntime)]

    df.index = time
    df['hod'] = df.index.time
    
    return df

FILL_VALUE = 10
# load point response function (prf)
one_sided_prf = pd.read_csv('one_sided_prf.csv', index_col=0).prf.values
prf30, prf30_symmetric = util.prf30_from_prf6(one_sided_prf)

for fname in sorted(glob.glob('data-bern/Bern_????_Internal_Rn_DB_v01.xlsx')):
    df = load_radon(fname, 'Raw')
    
    df.lld[df.lld<0] = FILL_VALUE
    
    df['lld_deconvolve'] = util.deconvlucy1d(df.lld.values, prf30_symmetric, 
                                              reg='tv', iterations=1000,
                                              lambda_reg=0.008)
    
    fname_save = os.path.splitext(fname)[0] + '_raw_deconvolve.xlsx'
    df.to_excel(fname_save)
    

if __name__ == "__main__":
    #code
    pass
