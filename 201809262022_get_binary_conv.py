"""
This branch is the software release for the 2019 paper: https://www.nature.com/articles/s41598-019-47795-0

See LICENSE.txt

Copyright 2019 Massachusetts Institute of Technology


Useage:
from importlib import reload
module = __import__(file_name)
reload(module)
get_data = getattr(module, 'get_data'
"""
__author__ = 'Greg Ciccarelli'
__date__ = 'June 12, 2018'



def load_data(file_path_name_audio, file_path_name_eeg, train=None):
    """Return attended audio, eeg, and unattended audio from *.mat file.

    """
    import scipy
    import scipy.io
    import sklearn.preprocessing
    import numpy as np
    
    # Load real data    
    loaded_data_audio = scipy.io.loadmat(file_path_name_audio)
    loaded_data_eeg = scipy.io.loadmat(file_path_name_eeg)

    audio = loaded_data_audio['data'] #audio
    eeg = loaded_data_eeg['data']
    eeg = eeg[:, :64, :]
    
    audio_unatt = loaded_data_audio['data_unatt']

    try:
        idx_part_keep = np.ravel(loaded_data_eeg['info'][0, 0]['reject'][0, 0]['idx_part_keep']).astype(np.bool)
        audio = audio[idx_part_keep]
        audio_unatt = audio_unatt[idx_part_keep]
    except:
        print('no reject field')
    
    return audio, eeg, audio_unatt


def get_data(audio, eeg, audio_unatt=None, idx_eeg=None, num_batch=None, idx_sample=None, num_context=1, num_predict=1, dct_params=None):
    """Select a sequence of audio, audio_unattnd, and eeg data
    
    Reshape the selected data into num_batch frames for prediction.
    
    Arguments
    ---------
    audio : (num_part, num_samples)
    eeg : (num_part, num_ch, num_samples)
    idx_sample : row idx of audio and eeg data
        Defaults to a random sample if not specified
    num_context : scalar
        Total number of samples of input used to predict an output sample.
        If one-to-one mapping with no delay, num_context=1    
        
    num_predict : scalar
        Total number of time samples to be predicted in the output
        
    dct_params['idx_keep_audioTime']:  index of samples into the time vector
    Returns
    -------
    X : Variable (num_batch, num_ch * num_context + num_context) eeg + audio
    
    y : Variable (num_batch, class)
        
    z_unatt : None
    
    """
    
    if (dct_params is not None) and ('idx_keep_audioTime' in dct_params):
        idx_keep_audioTime = dct_params['idx_keep_audioTime']
    else:
        idx_keep_audioTime = None
               
    
    ######################################################################################################
    import numpy as np
    import scipy
    import scipy.signal
    import torch
    from torch.autograd import Variable
    import sklearn
    import sklearn.preprocessing
    import time
    
    a = audio[idx_sample] # selected audio part
    e = eeg[idx_sample] # selected eeg part
    au = audio_unatt[idx_sample] # selected unattended audio
    
    
    # Trim off NaNs
    idx_a = np.logical_not(np.isnan(a))
    idx_e = np.logical_not(np.isnan(e[1]))
    if np.abs(np.sum(idx_a) - np.sum(idx_e)) > 3:
        print('unequal samples')
    idx_keep = np.logical_and(idx_a, idx_e)
    a = a[idx_keep]
    e = e[:, idx_keep]
    au = au[idx_keep]

    if a.shape[0] >= num_context:
        # Make a conv matrix out of the eeg
        # Make a conv matrix out of the attended audio
        # Make a conv matrix out of the unattended audio
        
        # Cat [X_eeg, X_audio], y = 1
        # Cat [X_eeg, X_audio_unatt], y = 0
        # Return X, y
        
        # No frame shifts are needed.
        
        num_time = a.size - num_context + 1
        num_ch = e.shape[0]

        if idx_keep_audioTime is None:
            num_column_audio = num_context
            idx_keep_audioTime = np.arange(num_context)
        else:
            num_column_audio = np.size(idx_keep_audioTime)
        
        X_eeg = np.nan * np.ones((num_time, num_ch, num_column_audio))
        X_audio = np.nan * np.ones((num_time, num_column_audio))
        X_audio_unatt = np.nan * np.ones((num_time, num_column_audio))        
        
        print(X_eeg.shape)
        for idx in range(num_time):
            idx_keep = np.arange(num_context) + idx
            for idx_ch in range(num_ch):
                X_eeg[idx, idx_ch] = np.ravel(e[idx_ch, idx_keep])[idx_keep_audioTime]
            X_audio[idx] = np.ravel(a[idx_keep])[idx_keep_audioTime]
            X_audio_unatt[idx] = np.ravel(au[idx_keep])[idx_keep_audioTime]
        X_audio = X_audio[:, None, :]
        X_audio_unatt = X_audio_unatt[:, None, :]
        
        X1 = np.concatenate((X_eeg, X_audio), axis=1)
        X0 = np.concatenate((X_eeg, X_audio_unatt), axis=1)
        X = np.concatenate((X0, X1), axis=0)
        y = np.concatenate((np.zeros((num_time, 1)), np.ones((num_time, 1))), axis=0)

        X = Variable(torch.from_numpy(X).type('torch.FloatTensor'))
        y = Variable(torch.from_numpy(np.array(y)).type('torch.FloatTensor'))        
        z_unatt = None
        
    else:
        print('-warning, too little data-')
        X = None
        y = None
        z_unatt = None
        a = None
        a_unatt = None

    return X, y, z_unatt
