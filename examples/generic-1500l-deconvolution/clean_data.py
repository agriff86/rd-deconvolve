"""
Deconvolution for a 1500L radon detector

Step 1: clean up the raw data files

This script reads from ./data-raw and writes a pickle file to ./data-intermediate

For plots to appear, run inside a Jupyter session (cell markers are delimited with #%%)
or add a call to plt.show() at the end.
"""

#%%
import glob
import numpy as np
import pandas as pd
import datetime
import os
import sys
import glob

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import rddeconv
except ImportError:
    # assume we're running from within source tree but don't want to install
    sys.path.append(PROJECT_DIR)
    import rddeconv

import rddeconv
from rddeconv.util import load_standard_csv
from rddeconv.emcee_deconvolve_tm import emcee_deconvolve_tm


ddir0 = os.path.join(EXAMPLE_DIR,'data-raw')
ddir1 = os.path.join(EXAMPLE_DIR,'data-intermediate')
ddir2 = os.path.join(EXAMPLE_DIR,'data-processed')

#%%
for ddir in ddir1,ddir2:
    if not os.path.exists(ddir):
        print('Creating directory: {}'.format(ddir))
        os.mkdir(ddir)

#%%
def _load_single_file(fname):
    """Load a single file in the usual csv format 

    Parameters
    ----------
    fname : string
        name of file to read
    """
    df = pd.read_csv(fname)
    # remove extra whitespace from column names, and convert to lower case
    df.columns = [itm.strip().lower() for itm in df.columns]
    df['time'] = [datetime.datetime.strptime(itm, '%H:%M').time() for itm in df.time]
    time = [ datetime.datetime.combine(datetime.date(int(itm[1]['year']),
                                                         int(itm[1]['month']),
                                                         int(itm[1]['dom'])),
                                           itm[1]['time']) for itm in df.iterrows()]
    df.index = time
    #clean up negative values (negative values for lld probably indicate missing data)
    df.loc[df.lld<0, 'lld'] = np.NaN
    return df


def load_data(fname_glob):
    """Load raw radon data files

    Parameters
    ----------
    fname_glob : str
        File name or file name glob (e.g. "raw-data/*.CSV")
    """
    fnames = sorted(glob.glob(fname_glob))
    df = pd.concat([_load_single_file(fn) for fn in fnames]).sort_index()
    return df

fname_glob = os.path.join(ddir0, '*.CSV')
print("Reading: {}".format(fname_glob))
df = load_standard_csv(fname_glob)

# Missing values represented with -9999
df[df==-9999] = np.NaN

#%%

#df = df.drop_duplicates().sort_index()
# want to drop entire rows if the index is duplicated
df = df.loc[np.logical_not(df.index.duplicated()), :]
df = df.sort_index()
df["lld_raw"] = df.lld.copy()
# input data should already be at 30-min, but sometimes files have duplicated
# rows
df = df.resample("30min").asfreq()

# %%
# some cleanup
df["lld_raw"] = df.lld.copy() # keep the original data
df = df.resample("30min").asfreq()

df.loc[df.uld > 10, "lld"] = np.NaN
#%%

# typically, we would add the 'cal' band 'bg' columns to the input data
# because of the amount of user judgement required to define them.
# For this example, just use constants.

df['cal'] = 0.188   # (Detector cps) / (Bq/m3 inside detector)
df['bg'] = 160.     # Counts per half hour

df.head()

# %%
df[['inflow']].plot()
# %%

# add time-varying detector parameters, in correct (SI) units
# ["background_rate", "Q", "Q_external", "total_efficiency"]
# total efficiency is (counts_per_second) / (Bq/m3)

# preliminary cal and bg are from spreadsheet
df["background_rate"] = df['bg'] / (60*30)
df["Q_external"] = df.exflow / 1000 / 60.0
df["total_efficiency"] = df['cal']
# internal flow, convert from velocity (m/sec) to volumetric flow rate m3/sec
# the inflow parameter is in units of m/sec, pipe diameter is 100mm
pipe_area = np.pi * (100 / 1000 / 2.0) ** 2
df["Q"] = df.inflow * pipe_area

#%%
df[["lld"]].plot()

# %%
df[["background_rate", "Q", "Q_external", "total_efficiency", 'airt','relhum','tankp']].plot(subplots=True)

#%%
fname_out = os.path.join(ddir1,'deconvolution_input_data.csv')
print("Writing: {}".format(fname_out))
df.to_csv(fname_out)


# %%
