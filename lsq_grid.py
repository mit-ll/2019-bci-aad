"""Dump all the utility functions into this one file and load it for nipype loop

This branch is the software release for the 2019 paper: https://www.nature.com/articles/s41598-019-47795-0

See LICENSE.txt

Copyright 2019 Massachusetts Institute of Technology



"""
__author__ = "Greg Ciccarelli"
__date__ = "October 12, 2018"

import os
from glob import glob
import scipy
import scipy.io
import scipy.signal
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import datetime
import h5py
import re
import hashlib
import sys

def make_conv(X, y, z=None, idx_ch=None, num_context=26):
    """Return A matrix and b vector for Aw=b.
    
    Arguments
    ---------
    X : array, (ch by time sample) eeg
    y : array, (time sample) attd envelope
    z : array, (time sample) unattended envelope
    idx_ch : array, (num_ch) array of indices for kept eeg channels
    num_context : scalar, number of time samples in a frame aka number of columns in A.
    
    Returns
    -------
    X_out : array, (num frames, num_context * num_ch) Reshaped EEG for least squares
            ch0, t0 ... tN, ch1 t0 ... tN     
    y_out : array, (num frames, 1) Attended audio
    z_out : array, (num frames, 1) Unattended audio
    
    """
    num_ch = np.size(idx_ch)
    
    # Select non-nan data and desired eeg ch
    idx_keep = ~np.isnan(y)
    y = y[idx_keep]
    if z is not None:
        z = z[idx_keep]
    X = X[:, idx_keep]
    if idx_ch is not None:
        X = X[idx_ch]
    
    if False:
        print('filtering');
        stop_atten_dB = 40
        num_order = 12
        freq_crit =  0.3
        b, a = scipy.signal.cheby2(num_order, stop_atten_dB, freq_crit, 'low', analog=False)
        X = scipy.signal.lfilter(b, a, X)


    # Create output:  
    #  audio, (num_output, 1)
    #  eeg, (num_output, channel * num_context)
    num_output = y.size - num_context + 1
    y_out = y[:(-num_context+1)] 
    if z is not None:
        z_out = z[:(-num_context+1)]
        z_out = z_out[:, None] # None = np.newaxis
    else:
        z_out = None

    
    # python edge case of -0 should mean "take all the data" but it doesn't.
    # not a problem so long as num_context > 1
    X_out = np.nan * np.ones((num_output, num_context * num_ch))
    for idx in range(num_output):
        idx_keep = idx + np.arange(num_context)
        # ch0, t0 ... t25, ch1 t0 ... t25     
        X_out[idx] = np.ravel(X[:, idx_keep])
        
    return X_out, y_out[:, None], z_out


def cat_part(eeg, audio, audio_unatt=None, idx_ch=None, num_context=26):
    """Return big A matrix (concat of A from all parts) for Aw=b.
    
    Arguments
    ---------
    eeg:  array (part, ch, time) Attended audio
    audio:  array (part, time) Attended audio
    audio_unatt:  array (part, time) Unattended audio    
    idx_ch : array, (num_ch) array of indices for kept eeg channels
    num_context : scalar, number of time samples in a frame aka number of columns in A.
    
    Returns
    -------
    X_all : array, (num frames, num_context * num_ch) Reshaped EEG for least squares
            ch0, t0 ... tN, ch1 t0 ... tN     
    y_all : array, (num frames, 1) Attended audio
    z_all : array, (num frames, 1) Unattended audio
    
    """
    t_start = datetime.datetime.now()
    
    X = eeg[0]
    y = audio[0]
    if audio_unatt is not None:
        z = audio_unatt[0]
    else:
        z = None
    
    X_all, y_all, z_all = make_conv(X, y, z, idx_ch=idx_ch, num_context=num_context)
    groups = np.zeros((X_all.shape[0], 1))

    for idx_part in range(1, audio.shape[0]):
        #print(idx_part)
        y = audio[idx_part]
        if z is not None:
            z = audio_unatt[idx_part]
        X = eeg[idx_part]
        Xi, yi, zi = make_conv(X, y, z, idx_ch=idx_ch, num_context=num_context)
        X_all = np.concatenate((X_all, Xi), axis=0)
        y_all = np.concatenate((y_all, yi), axis=0)
        if z is not None:
            z_all = np.concatenate((z_all, zi), axis=0)
        groups = np.concatenate((groups, idx_part * np.ones((Xi.shape[0], 1))), axis=0)

    # Technically, this should not be necessary, but sometimes the eeg still has nan "inside" what should be good data.
    idx_keep = np.all(~np.isnan(X_all), axis=1)
    X_all = X_all[idx_keep]
    y_all = y_all[idx_keep]
    if z is not None:
        z_all = z_all[idx_keep]
    else:
        z_all = None
    
    t_end = datetime.datetime.now()
    print('- conv time -')
    print(t_end - t_start) 
   
    return X_all, y_all, z_all, np.ravel(groups)



def load_data(file_path_name_audio, file_path_name_eeg):
    """Return the attended and unattended audio and eeg.
    
    Arguments
    ---------
    file_path_name_audio: string, path to attended and unattende audio mat.
    file_path_name_eeg: string, path to eeg mat.
    
    Returns
    -------
    audio:  array (part, time) Attended audio
    eeg:  array (part, ch, time) Attended audio
    audio_unatt:  array (part, time) Unattended audio  
    
    """    
    loaded_data = scipy.io.loadmat(file_path_name_eeg)
    eeg_ht = loaded_data['data']

    loaded_data = scipy.io.loadmat(file_path_name_audio)
    audio_ht = loaded_data['data']
    audio_unatt_ht = loaded_data['data_unatt']
    
    return audio_ht, eeg_ht, audio_unatt_ht




