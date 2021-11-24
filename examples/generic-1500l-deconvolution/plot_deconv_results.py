
#%%
import glob
import numpy as np
import pandas as pd
import datetime
import os
import logzero
import sys
import xarray as xr
import matplotlib.pyplot as plt
import subprocess

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import rddeconv
except ImportError:
    # assume we're running from within source tree but don't want to install
    sys.path.append(PROJECT_DIR)
    import rddeconv


import yaml

from rddeconv.util import load_standard_csv
from rddeconv.emcee_deconvolve_tm import emcee_deconvolve_tm

from rddeconv.pymc3_deconvolve import stats_from_xarray
#%%

ddir0 = './data-raw'
ddir1 = './data-intermediate'
ddir2 = './data-processed'

df_input = pd.read_csv(os.path.join(ddir1, 'deconvolution_input_data.csv'), index_col=0, parse_dates=True)
fn_emcee = f'{ddir2}/emcee-deconv-results.csv'
if os.path.exists(fn_emcee):
    df_emcee = pd.read_csv(fn_emcee, index_col=0, parse_dates=True)
else:
    df_emcee = None
fnames = sorted(glob.glob(os.path.join(ddir1, 'deconvolution_result_chunk*.nc')))


def emcee_resample(df, interval='1h'):
    sample_cols = [itm for itm in df.columns if itm.startswith('lldsample_')]
    dfr = df.resample(interval, closed='right', label='right').mean()

    # re-calculate percentiles for lld -- note this can't be done!!
    # ... because the emcee samples have already been shifted to the _av ones
    #dfpc = dfr[sample_cols].quantile(q=[.03,.1,.16,.5,.84,.9,.97], axis=1).T
    #dfpc.columns = ['lld_3pc', 'lld_10pc', 'lld_16pc', 'lld_50pc', 'lld_84pc', 'lld_90pc','lld_97pc']
    #for k in dfpc.columns:
    #    dfr[k] = dfpc[k]

    # re-calculate percentiles for lldav
    #df_av = (df[sample_cols] + df[sample_cols].shift(1))/2
    ## - no - the lld_samples are already shifted!!!
    # check - 
    lldav_mean_check = df.mean(axis=1)
    (dfr.lldav_mean - lldav_mean_check).max()
    dfr.lldav_mean.head()
    lldav_mean_check.head()

    df = df.resample(interval, closed='right', label='right').mean()
    dfpc_av = df.quantile(q=[.03,.1,.16,.5,.84,.9,.97], axis=1).T
    dfpc_av.columns = ['lldav_3pc', 'lldav_10pc', 'lldav_16pc', 'lldav_50pc', 'lldav_84pc', 'lldav_90pc','lldav_97pc']
    for k in dfpc_av.columns:
        dfr[k] = dfpc_av[k]
    # drop the sample columns
    dfr = dfr.drop(columns=sample_cols)
    return dfr

df_emcee_30min = emcee_resample(df_emcee, interval='30min')
df_emcee_1h = emcee_resample(df_emcee, interval='1h')

#%%
def summarise_pymc3(fnames):
    """
    summarise output from pymc3 deconvolution backend
    """
    def preproc(ds0):
        # get rid of the overlap
        ds0 = ds0.where(ds0.overlap_flag==0, drop=True)
        # and remove the chunk_id field
        ds0 = ds0.squeeze('chunk_id')
        # E simulated counts too hard to handle (at least for now)
        try:
            # TODO: handle chunk_id properly
            ds0 = ds0.drop_vars(['E_simulated_counts','overlap_flag','chunk_id'])
        except ValueError:
            print('E_simulated_counts missing?')
            pass
        return ds0

    ds = xr.open_mfdataset(fnames, preprocess=preproc, combine='by_coords', concat_dim='time')
    ds = ds.load()
    ds_stats = stats_from_xarray(ds)
    dsr = ds.resample({'time':'1h'}, closed='right', label='right')
    ds1h = dsr.mean()
    ds_stats_1h = stats_from_xarray(ds1h)

    return ds_stats, ds_stats_1h

def add_time_vars(ds):
    """
    Add extra time variables (t0,t1) assuming that existing time corresponds to the end of sampling period
    """
    # handle data gap between t[0] and t[1]..
    delta_t = np.min(np.diff(ds.time.values))
    ds['t1'] = ds.time
    ds['t0'] = ds.time - delta_t
    return ds


def summarise_emcee(df_emcee):

    try:
        dfss = df_emcee[['lldav_mean','lldav_16pc', 'lldav_84pc']].copy()*rddeconv.lamrn
    except KeyError:
        dfss = df_emcee[['lldav_mean','lldav_p16', 'lldav_p84']].copy()*rddeconv.lamrn

    dfss.index.name = 'time'
    dfss.columns = ['radon_deconv_emcee', 'radon_deconv_emcee_16pc', 'radon_deconv_emcee_84pc']
    dfss = xr.Dataset(dfss)

    return dfss

def add_metadata(ds):
    metadata = [ ['time', 'Time', None],
             ['t0','Counting interval start', None],
             ['t1','Counting interval stop', None],
             ['radon_stp', 'Radon concentration at STP', 'Bq/m3 at STP; 20 degC 1000 hPa'],
             ['radon', 'Radon concentration inside detector', 'Bq/m3'],
             ['radon_uncertainty', 'One-sigma fractional uncertainty in radon concentration', ''],
             ['exflow', 'External loop flow rate', 'L/min'],
             ['gm', 'Number of gas meter impeller cycles', '/30-min'],
             ['inflow', 'Flow velocity at centre of internal 50mm ID pipe', 'm/s'],
             ['hv', 'High voltage supply to PMT tube', 'V'],
             ['lld', 'Scintillation counts', '/30-min'],
             ['lld_raw', 'Scintillation counts, as reported by detector', '/30-min'],
             ['uld', 'Noise counts', '/30-min'],
             ['tankp', 'Tank differential pressure, raw output from sensor', 'mV'],
             ['temp', 'Temperature of the data logger', 'degC'],
             ['airt', 'Air temperature measured inside the main tank', 'degC'],
             ['relhum', 'Relative humidity measured inside the main tank', '%'],
             ['press', 'Air pressure measured inside the main tank', 'hPa'],
             ['batt', 'Logger internal battery voltage', 'V'],
             ['comments', 'Comments recorded by operator', ''],
             ['flag', 'Status flag', '0 - normal, 1 - background, 2 - calibration',],
             ['lld_meas', 'Scintillation counts during sampling', '/30-min'],
             ['sensitivity', 'Detector sensitivity to radon', 'cps/(Bq/m3)'],
             ['sensitivity_uncertainty', 'One-sigma fractional uncertainty in sensitivity',''],
             ['background', 'Detector background counts', '/30-min'],
             ['background_uncertainty', 'One-sigma fractional uncertainty in background', ''],
             ['Q', 'Internal loop volumetric flow','m3/s'],
             ['Q_external', 'External loop volumetric flow', 'm3/s'],
             ['total_efficiency', 'Detector counts per ambient Bq/m3', '1/s / (1/s/m3)'],
             ['background_rate', 'Detector background count rate', '1/s'],
             ['radon_deconv', 'Radon concentration inside detector with response time correction applied', 'Bq/m3'],
             ['radon_deconv_sd', 'RT corrected radon uncertainty (one sigma)', 'Bq/m3'],
             ['radon_deconv_16pc', 'RT corrected radon uncertainty (16th percentile)', 'Bq/m3'],
             ['radon_deconv_84pc', 'RT corrected radon uncertainty (84th percentile)', 'Bq/m3'],
             ['radon_deconv_3pc', 'RT corrected radon uncertainty (3rd percentile)', 'Bq/m3'],
             ['radon_deconv_97pc', 'RT corrected radon uncertainty (97th percentile)', 'Bq/m3'],
             ['radon_deconv_emcee', 'Radon concentration inside detector with response time correction applied, emcee method', 'Bq/m3'],
             ['radon_deconv_emcee_sd', 'RT corrected radon uncertainty, emcee method (one sigma)', 'Bq/m3'],
             ['radon_deconv_emcee_16pc', 'RT corrected radon uncertainty, emcee method (16th percentile)', 'Bq/m3'],
             ['radon_deconv_emcee_84pc', 'RT corrected radon uncertainty, emcee method (84th percentile)', 'Bq/m3'],
             ['radon_deconv_emcee_3pc', 'RT corrected radon uncertainty, emcee method (3rd percentile)', 'Bq/m3'],
             ['radon_deconv_emcee_97pc', 'RT corrected radon uncertainty, emcee method (97th percentile)', 'Bq/m3'],
             ['radon_deconv_emcee_10pc', 'RT corrected radon uncertainty, emcee method (10th percentile)', 'Bq/m3'],
             ['radon_deconv_emcee_90pc', 'RT corrected radon uncertainty, emcee method (90th percentile)', 'Bq/m3'],
             ]

    for k,desc,units in metadata:
        if k in ds:
            ds[k].attrs['long_name'] = desc
            if units is not None:
                ds[k].attrs['units'] = units

    for k in ['time','t0','t1']:
        ds[k].attrs['timezone'] = 'not specified'

    return ds

def write_output(ds, fn_root='deconv_result'):
    # save netCDF data
    ##nc_fn = os.path.join(ddir2, 'deconvolution_result.nc')
    nc_fn = fn_root + '.nc'
    ds.to_netcdf(nc_fn)

    # make a csv copy of the data
    ##csv_fn = os.path.join(ddir2, 'deconvolution_result.csv')
    csv_fn = fn_root + '.csv'
    ds.to_dataframe().to_csv(csv_fn)

    # write metadata as json
    ##json_fn = os.path.join(ddir2, 'deconvolution_result_metadata.json')
    json_fn = fn_root + '.json'
    if os.path.exists(json_fn):
        os.unlink(json_fn)
    cmd = f'ncks --json -m {nc_fn}'
    json_string = str(subprocess.check_output(cmd.split()), encoding=sys.getdefaultencoding())
    with open(json_fn, 'wt') as fd:
        fd.write(json_string)

    # create a zip file from metadata
    ##zip_fn = os.path.join(ddir2, 'deconvolution_result.zip')
    zip_fn = fn_root + '.zip'
    if os.path.exists(zip_fn):
        os.unlink(zip_fn)
    cmd = f'zip {zip_fn} {json_fn} {csv_fn} {nc_fn}'
    subprocess.check_call(cmd.split())


# add a radon columns to input data
delta_t = 30*60
df_input['radon'] = ((df_input['lld'] - df_input['uld'])/delta_t - df_input['background_rate']) / df_input['total_efficiency']

if len(fnames) > 0:
    print(f"{len(fnames)} pymc3 output files found")
    ds30min, ds1h = summarise_pymc3(fnames)
else:
    ds30min = xr.Dataset()
    ds1h = xr.Dataset()

ds_emcee = summarise_emcee(df_emcee_30min)
ds_emcee_1h = summarise_emcee(df_emcee_1h)
df_input.index.name = 'time'
drop_columns = ['year', 'doy', 'month', 'dom', 'time','spare','lld_raw']
dfi = df_input.drop(columns=drop_columns, errors='ignore')

dfr = dfi.resample('1h', closed='right', label='right')
sum_cols = ['lld','uld']
dfi1h = dfr.mean()
for k in sum_cols:
    dfi1h[k] = dfr[k].sum()

dsj = xr.merge([ds30min, ds_emcee, xr.Dataset(dfi)])
dsj1h = xr.merge([ds1h, ds_emcee_1h, xr.Dataset(dfi1h)])

#%%
def showplot(dsj):
    fig, ax = plt.subplots(figsize=[12,4])
    pltcols = [itm for itm in ['radon','radon_deconv', 'radon_deconv_emcee'] if itm in dsj.columns]
    have_emcee =  'radon_deconv_emcee' in dsj.columns
    have_pymc3 = 'radon_deconv' in dsj.columns
    dsj[pltcols].plot(ax=ax)
    
    if have_pymc3:
        ax.fill_between(dsj.index.values, dsj.radon_deconv_16pc.values, dsj.radon_deconv_84pc.values, color='grey', zorder=0, alpha=0.5)
    if have_emcee:
        ax.fill_between(dsj.index.values, dsj.radon_deconv_emcee_16pc.values, dsj.radon_deconv_emcee_84pc.values, color='grey', zorder=0, alpha=0.5)
    ax.set_title('30-min resolution')
    ax.set_xlabel('')
    ax.set_ylabel('Radon activity concentration inside detector (Bq/m3)')
    return fig, ax

#%%
fig, ax = showplot(dsj.to_dataframe()['2020-11-17':'2020-11-20'])
ax.set_title('Response time correction at 30-min resolution (preliminary)')
fig.savefig(ddir2+'/response-time-correction-example1-30min.png',dpi=300)
# %%
fig, ax = showplot(dsj1h.to_dataframe()['2020-11-17':'2020-11-20'])
ax.set_title('Response time correction at 60-min resolution (preliminary)')
fig.savefig(ddir2 + '/response-time-correction-example1-60min.png',dpi=300)
#%%
fig, ax = showplot(dsj.to_dataframe()['2020-12-17':'2020-12-22'])
ax.set_title('Response time correction at 30-min resolution (preliminary)')
fig.savefig(ddir2+'/response-time-correction-example2-30min.png',dpi=300)
# %%
fig, ax = showplot(dsj1h.to_dataframe()['2020-12-17':'2020-12-22'])
ax.set_title('Response time correction at 60-min resolution (preliminary)')
fig.savefig(ddir2 + '/response-time-correction-example2-60min.png',dpi=300)


#%%

dsj = add_time_vars(dsj)
dsj = add_metadata(dsj)
dsj1h = add_time_vars(dsj1h)
dsj1h = add_metadata(dsj1h)

write_output(dsj, ddir2+'/Generic_1500L_detector_RT_corrected_radon_30min_preliminary')
write_output(dsj1h, ddir2+'/Generic_1500L_detector_radon_60min_preliminary')

