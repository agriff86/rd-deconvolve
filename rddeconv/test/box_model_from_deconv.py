#!/usr/bin/env python
# coding: utf-8

"""
    box_model_from_deconv
    ~~~~~~~~~~~~~~~~~~~~~

    Compute a box model derived mixing height based on deconvolved radon

    Box model suff copied from radon_box_model_analysis.py in directory
    2015-09-romania-ceilometer/box-model

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
from micromet.plot import figure_template, fancy_ylabel

import theoretical_model as tm

from radon.pbl_estimate import radon_timeseries_to_mixlength, hours_since_sunset, hours_since_sunrise, index_by_night

fname_data = 'data-processed/tm_deconvolution_glb.pkl'
fdir='./figures'
mpl.rcParams['font.sans-serif'] = ['Source Sans Pro']

# color palette from seaborn
seaborn_palettes = dict(
    deep=["#4C72B0", "#55A868", "#C44E52",
          "#8172B2", "#CCB974", "#64B5CD"],
    muted=["#4878CF", "#6ACC65", "#D65F5F",
           "#B47CC7", "#C4AD66", "#77BEDB"],
    pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
            "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    bright=["#003FFF", "#03ED3A", "#E8000B",
            "#8A2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#017517", "#8C0900",
          "#7600A1", "#B8860B", "#006374"],
    colorblind=["#0072B2", "#D55E00", "#009E73",
                "#CC79A7", "#F0E442", "#56B4E9"],
)
mpl.rcParams["axes.color_cycle"] = list(seaborn_palettes['colorblind'])

def fancy_ylabel(ax,s, pos='left', hoffset=0):
    textsize=mpl.rcParams['font.size']
    if pos == 'left':
        ann = ax.annotate(s,xy=(0,1), xycoords='axes fraction',
                    textcoords='offset points', xytext=(-28+hoffset, 10),
                    fontsize=textsize)
    elif pos == 'right':
        ann = ax.annotate(s,xy=(1,1), xycoords='axes fraction',
                    textcoords='offset points', xytext=(28+hoffset, 10),
                    fontsize=textsize,
                    horizontalalignment='right')
    return ann


def newfig():
    fig, ax = figure_template('acp-1')
    return fig, ax

def savefig(fig, name, fdir=fdir):
    fig.savefig(os.path.join(fdir, name+'.pdf'), transparent=True, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fdir, name+'.png'), transparent=False, dpi=300, bbox_inches='tight')


def run_boxmodel(t, df, colname='rn'):
    #estimated flux
    F = 1e-3
    #number of hours past sunrise to extend the analysis
    steps_past_sunrise = 10
    #number of hours before sunset to extend the analysis
    steps_before_sunset = 2
    #maximum allowed PBL
    HMIX_MAX = 3000.0
    #measurement site
    lon,lat = 150.748, -33.618
    lat, lon = 44.350756, 26.044006

    lat, lon = -34.887641, 149.695796 #goulburn field site

    #timezone in hours, i.e. 10 means UTC+10
    tzhr = 10

    #assume radon obs timestamp is for the end of the hour
    obstime = np.array(t) - datetime.timedelta(hours=tzhr) - datetime.timedelta(minutes=30)
    obsrn = df[colname].values

    t_rn_mix, rn_mix, rn_mix_acc,blah1, blah2 = radon_timeseries_to_mixlength(obstime,obsrn,lon,lat,
                                    F,steps_past_sunrise,
                                    steps_before_sunset=steps_before_sunset,
                                    tz=0,tolerate_errors=True,
                                    return_debug=True,
                                    hmax=HMIX_MAX,
                                    regular_grid=False)


    ##repeat analysis using the difference between observed radon and
    ##interpolated background
    #obsrn = df['diff'].values/1000.0
    #t_diff_mix, diff_mix, diff_mix_acc,blah1, blah2 = radon_timeseries_to_mixlength(obstime,obsrn,lon,lat,
    #                                F,steps_past_sunrise,
    #                                steps_before_sunset=steps_before_sunset,
    #                                tz=0,tolerate_errors=True,
    #                                return_debug=True,
    #                                hmax=HMIX_MAX,
    #                                regular_grid=False)

    #add an hours-since-sunset variable
    hss = hours_since_sunset(np.array(t) - datetime.timedelta(hours=tzhr), lon, lat)
    df['hours_since_sunset'] = hss
    df['rounded_hours_since_sunset'] = np.round(hss)

    #add an hours-since-sunrise variable
    hsr = hours_since_sunrise(np.array(t) - datetime.timedelta(hours=tzhr), lon, lat)
    df['hours_since_sunrise'] = hsr
    df['rounded_hours_since_sunrise'] = np.round(hsr)

    #note: convert back to local time
    dfnew = pd.DataFrame(index=np.array(t_rn_mix) + datetime.timedelta(hours=tzhr),
                          data=dict(rn_h_e=rn_mix,
                                    rn_h_a=rn_mix_acc
                                    ))

    dfret = df.join(dfnew)

    ##add the mixing heights based on 'diff'
    #dfnew = pd.DataFrame(index=np.array(t_diff_mix) + datetime.timedelta(hours=tzhr),
    #                      data=dict(diff_h_e=diff_mix,
    #                                diff_h_a=diff_mix_acc
    #                                ))
    #dfret = dfret.join(dfnew)

    #add a column for the night number (1--however many)
    night_number = index_by_night(obstime, lon,lat,tz=0)
    dfret['night_number'] = pd.Series(index=t, data=night_number)

    return dfret

df = pd.read_pickle(fname_data)
flux_mBq_m2_s = 19.2 # see calculation in manuscript

# which lag is best??  1Hr is the answer.
nlag = 4
lcorr = np.zeros(nlag)
for ii in range(nlag):
    lcorr[ii] = df[['lld_mean']].corrwith(df.lld_scaled.shift(-ii))
fig, ax = plt.subplots()
ax.plot(lcorr)
ax.set_xlabel('lag (number of points)')
ax.set_ylabel('correlation between deonvolved time series and lagged time series')

# add a lagged column (30 min)
df['lld_scaled_shifted'] = df.lld_scaled.shift(-2)



columns_to_process = [itm for itm in df.columns if itm.startswith('lldsample_')]
hmix_list = []
for col in columns_to_process:
    dfproc = run_boxmodel(df.index.to_pydatetime(), df, colname=col)
    hmix = dfproc.rn_h_e / tm.lamrn * flux_mBq_m2_s
    hmix_list.append(hmix)

# scaled detector output
dfproc = run_boxmodel(df.index.to_pydatetime(), df, colname='lld_scaled')
hmix_raw_output = dfproc.rn_h_e / tm.lamrn * flux_mBq_m2_s

#scaled+shifted detector output
dfproc = run_boxmodel(df.index.to_pydatetime(), df, colname='lld_scaled_shifted')
hmix_raw_shifted_output = dfproc.rn_h_e / tm.lamrn * flux_mBq_m2_s


A = np.vstack(hmix_list)
mean_est = A.mean(axis=0)
p10 = np.percentile(A, 10.0, axis=0)
p16 = np.percentile(A, 16.0, axis=0)
p84 = np.percentile(A, 84.0, axis=0)
p90 = np.percentile(A, 90.0, axis=0)



dfsummary = pd.DataFrame(index=hmix.index, data=dict(hmix_mean=mean_est,
                                                     hmix_p10=p10,
                                                     hmix_p16=p16,
                                                     hmix_p84=p84,
                                                     hmix_p90=p90,
                                                     hmix_scaled=hmix_raw_output,
                                                     hmix_scaled_shifted=hmix_raw_shifted_output))


def time_series_plot(df, extra_plots=[], column_name='hmix', ysc=1,
                     do_flight_data=True, ax=None,
                     draw_legend=True):
    """
    plot dfsummary (above), based on the function in draw_figures.py
    """
    #res = np.diff(df.index.to_pydatetime())[0].total_seconds()

    # make a custom legend handler to get the error range into the legend
    # ref: http://matplotlib.org/users/legend_guide.html#proxy-legend-handles
    class HardWiredHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = mpl.patches.Rectangle([x0, y0], width, height,
                                       facecolor='k', alpha=0.2,
                                       edgecolor='none',
                                       transform=handlebox.get_transform())
            line = mpl.lines.Line2D([x0, x0+width],
                                    [y0+height/2.0, y0+height/2.0],
                                    color='k',solid_capstyle='butt',
                                    linewidth=1.5,
                                    transform=handlebox.get_transform())
            handlebox.add_artist(patch)
            handlebox.add_artist(line)
            return patch

    if ax is None:
        fig, ax = figure_template('acp-1')
    else:
        fig = ax.figure

    # draw a plot with pandas to configure ax using pandas' date formatting
    # df.lld.plot(ax=ax)
    # del ax.lines[0]

    #ysc = 1.0/res


    t = df.index
    low = df[column_name+'_p16'].values  #_p16 or _p10
    #low[low<10] = 10 # prevent zeros
    # print(low.min())
    high = df[column_name+'_p84'].values  #_p84 or _p90
    r1 = ax.fill_between(t, low*ysc, high*ysc, alpha=0.2, color='k', linewidth=0)

    # for debugging
    # ax.plot(t, low*ysc, color='k', alpha=0.2)
    # ax.plot(t, high*ysc, color='k', alpha=0.2)

    r2 = ax.plot(t, df[column_name+'_mean'].values*ysc, 'k')

    leg_objs = []
    leg_entries = []
    leg_objs.append((r1,r2))
    leg_entries.append('Box model with BMC deconvolution')

    ret = ax.plot(t, df[column_name+'_scaled']*ysc)[0]
    leg_objs.append(ret)
    leg_entries.append('Box model with raw detector output')

    ret = ax.plot(t, df[column_name+'_scaled_shifted']*ysc)[0]
    leg_objs.append(ret)
    leg_entries.append('Box model with lagged detector output')

    if do_flight_data:
        #x,y = 2011-11-06 07:56:59.650000 671.056073212
        # (from 2014-10-goulburn-sbl-fluxes/extract_and_plot_soundings.py)
        x = datetime.datetime(2011,11,6,8,0)
        y = np.array([[583,671], [206,651]])

        x = datetime.datetime(2011,11,6,8,0), datetime.datetime(2011,11,6,7,25)

        ret = ax.errorbar(x, y.mean(axis=1), yerr=np.diff(y, axis=1).ravel()/2.0, fmt='none')
        leg_objs.append(ret)
        leg_entries.append('Aircraft measurements')

    for xdata, ydata, label in extra_plots:
        ret = ax.plot(xdata, ydata)[0]
        leg_objs.append(ret[0])
        leg_entries.append(label)

    # make the legend
    if draw_legend:
        ax.legend(leg_objs, leg_entries,
               handler_map={tuple: HardWiredHandler()},
               loc='upper left',
               frameon=False)


    # format x-axis
    #from micromet.plot import day_ticks
    #day_ticks(ax)  # TODO: switch to pandas convention
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    fancy_ylabel(ax, 'Boundary layer depth (m)')



    return fig, ax

dfss = dfsummary.ix[datetime.datetime(2011,11,6,6):
                    datetime.datetime(2011,11,6,12)]

f,ax = time_series_plot(dfss)

ax.set_yticks([0, 500, 1000, 1500])
xl = ax.get_xlim()
ax.set_xticks(ax.get_xticks()[::2])
savefig(f,'box_model_analysis')

print('average of deconvolved mixing height during morning / average of mixing height from scaled/lagged')
print(dfss.mean() / dfss.hmix_mean.mean())

# ouput:
#average of deconvolved mixing height during morning / average of mixing height from scaled/lagged
#hmix_mean              1.000000
#hmix_p10               0.672541
#hmix_p16               0.868123
#hmix_p84               1.168126
#hmix_p90               1.203829
#hmix_scaled            0.259855
#hmix_scaled_shifted    0.368004

if __name__ == "__main__":
    #code
    pass
