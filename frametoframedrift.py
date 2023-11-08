#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# frame-to-frame relative drift correction method, used in Vojnovic et al. 2023, Combining Single-molecule and Expansion microscopy in fission Yeast (SExY) to visualize protein structures at the nanostructural level 

import trackdata
import pandas as pd
import numpy as np
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    # A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8), pp 1627-1639.
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def calculate_drift(file, min_track_size=2, max_md=80, smooth_window=31, smooth_order=4):
    db = trackdata.load(file)

    tracks_filtered = db['Track'][(db['Track']['size'] > min_track_size) & (db['Track']['MD'] < max_md)][['MD', 'size']]
    data = pd.merge(tracks_filtered, db['Particle'][['Track_id', 'x', 'y', 't']], left_index = True, right_on = 'Track_id')
    data['t_next'] = data['t']+1
    data = pd.merge(data, data, left_on = ['t_next', 'Track_id'], right_on = ['t', 'Track_id'], suffixes = ['_a','_b'])
    drift = pd.DataFrame({'t':data['t_b'], 'd_x':data['x_b']-data['x_a'], 'd_y':data['y_b']-data['y_a']})
    drift = drift.groupby('t').mean().cumsum()
    drift['d_x_smooth'] = pd.Series(savitzky_golay(drift['d_x'].values, window_size=smooth_window, order=smooth_order), index = drift.index)
    drift['d_y_smooth'] = pd.Series(savitzky_golay(drift['d_y'].values, window_size=smooth_window, order=smooth_order), index = drift.index)
    T = pd.DataFrame({'t': range(1, db['Particle']['t'].iloc[-1])})
    drift['t_rev'] = drift.index
    drift = pd.merge(T, drift, left_on='t', right_index=True, how='left')
    drift['t_rev'] = drift['t_rev'].fillna(0).cummax()
    drift = pd.merge(pd.DataFrame(drift[['t_rev', 't']]), drift[['d_x_smooth', 'd_y_smooth', 'd_x', 'd_y', 't']], left_on='t_rev', right_on='t')
    drift.set_index('t_x')
    drift = drift[['d_x', 'd_y', 'd_x_smooth', 'd_y_smooth']]
    db['Particle']['x'] -= pd.merge(db['Particle'], drift, left_on='t', right_index=True, how='left').fillna(0)['d_x_smooth']
    db['Particle']['y'] -= pd.merge(db['Particle'], drift, left_on='t', right_index=True, how='left').fillna(0)['d_y_smooth']

    suffix = '_'+str(min_track_size)+'_'+str(max_md)+'_'+str(smooth_window)+'_'+str(smooth_order)

    trackdata.dump(db, file[:-4]+'_drift'+suffix+'_corrected.pmt')
    drift.plot().get_figure().savefig(file[:-4]+'_drift'+suffix+'.png')
    with open(file[:-4]+'_drift'+suffix+'.txt', 'w') as of:
        of.write(drift.to_csv())

calculate_drift('file.pmt', min_track_size=2, max_md=80, smooth_window=31, smooth_order=4)
