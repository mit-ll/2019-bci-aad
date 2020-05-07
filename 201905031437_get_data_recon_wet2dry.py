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
__date__ = 'May 3, 2018'



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
    
    #Reduce the wet eeg ch to their corresponding dry counterparts
    #eeg = eeg[:, [45, 25,  7,  9, 11, 29, 49, 27, 32,  0,  2, 23, 43, 60, 62,  5, 13, 42, 51, 31], :] # full 20 ch map
    eeg = eeg[:, [45, 25,  7,  9, 11, 29, 49, 27,  0,  2, 23, 43, 60, 62,  5, 13, 51, 31], :] # full drop M1 and M2  (32, 42 zero based indexing  
    
    audio_unatt = loaded_data_audio['data_unatt']

    try:
        idx_part_keep = np.ravel(loaded_data_eeg['info'][0, 0]['reject'][0, 0]['idx_part_keep']).astype(np.bool)
        audio = audio[idx_part_keep]
        audio_unatt = audio_unatt[idx_part_keep]
    except:
        print('no reject field')
    
    return audio, eeg, audio_unatt

def get_data(audio, eeg, audio_unatt=None, idx_eeg=None, num_batch=None, idx_sample=None, num_context=1, 
             num_predict=1, dct_params=None):
    """Select a sequence of audio, audio_unattnd, and eeg data.
    
    Reshape the selected data into num_batch frames for prediction.
    
    Arguments
    ---------
    audio : (num_part, num_samples)
    
    eeg : (num_part, num_ch, num_samples)
    
    audio_unatt : (num_part, num_samples)
    
    idx_sample : row idx of audio and eeg data
        Defaults to a random sample if not specified
        
    num_context : scalar
        Total number of samples of input used to predict an output sample.
        If one-to-one mapping with no delay, num_context=1    
        
    num_predict : scalar
        Total number of time samples to be predicted in the output
        
    g_sr : Not used.
    
    Returns
    -------
    X : Variable (num_batch, num_ch, num_context) eeg
    
    y : Variable (num_batch, num_predict), audio
        
    z : Variable (num_batch, num_predict), audio unattended
    """
    
    
    def convmtx_i(a, num_kernel):
        """Return convolution matrix of indices rather than A itself
        # Make a convolution matrix, A from input timeseries a
        # This requires taking a timeseries that goes forward in time e.g. 1:10
        # assume the kernel, h, is ordered forward in time, so first element is the farthest in the past.
        # A *h = y == conv(a, h, 'valid')

        num_kernel = 3
        h = (np.arange(num_kernel)+3.1)**2
        print(h)
        a = np.arange(10)
        A = convmtx(a, num_kernel)
        print(A)
        print(np.dot(A, h))
        print(scipy.signal.convolve(a, h, 'valid'))
        ---
        A_toeplitz = scipy.linalg.toeplitz(a)
        print(A_toeplitz)

        A = A_toeplitz[(num_kernel-1):, :num_kernel]    

        """
        import scipy
        import numpy
        
        # For loop is 20x faster than toeplitz.
        num_row = np.size(a) - num_kernel + 1
        Ai = np.nan * np.ones((num_row, num_kernel))
        for idx_row in range(num_row):
            idx = np.arange(num_kernel) + idx_row
            idx = idx[::-1]
            Ai[idx_row] = idx
        Ai = Ai.astype(np.int64)
        return Ai
    
    def convmtx(a, num_kernel):
        """Return convolution matrix A
        # Make a convolution matrix, A from input timeseries a
        # This requires taking a timeseries that goes forward in time e.g. 1:10
        # assume the kernel, h, is ordered forward in time, so first element is the farthest in the past.
        # A *h = y == conv(a, h, 'valid')

        num_kernel = 3
        h = (np.arange(num_kernel)+3.1)**2
        print(h)
        a = np.arange(10)
        A = convmtx(a, num_kernel)
        print(A)
        print(np.dot(A, h))
        print(scipy.signal.convolve(a, h, 'valid'))
        ---
        A_toeplitz = scipy.linalg.toeplitz(a)
        print(A_toeplitz)

        A = A_toeplitz[(num_kernel-1):, :num_kernel]    

        """
        import scipy
        
        # For loop is 20x faster than toeplitz.
        num_row = np.size(a) - num_kernel + 1
        A = np.nan * np.ones((num_row, num_kernel))
        for idx_row in range(num_row):
            idx = np.arange(num_kernel) + idx_row
            idx = idx[::-1]
            A[idx_row] = a[idx]
        return A    
    
    ######################################################################################################
    import numpy as np
    import scipy
    import scipy.signal
    import torch
    from torch.autograd import Variable
    import sklearn
    import sklearn.preprocessing
    import time
    
    a = audio[idx_sample]
    e = eeg[idx_sample]
    # Trim off NaNs
    idx_a = np.logical_not(np.isnan(a))
    idx_e = np.logical_not(np.isnan(e[1]))
    if np.abs(np.sum(idx_a) - np.sum(idx_e)) > 3:
        print('unequal samples')
    idx_keep = np.logical_and(idx_a, idx_e)
    a = a[idx_keep]
    e = e[:, idx_keep]


    if a.shape[0] >= num_context:

        # Conv matrix the audio data using a for loop
        # Trim the trailing audio samples as these can't be predicted without eeg data that comes after the audio
        num_total_batch = a.shape[0] - (num_context -1)
        a_pred = np.nan * np.ones((num_total_batch, num_predict))
        for idx in range(num_total_batch):
            a_pred[idx] = a[np.arange(num_predict) + idx]

             
        # Get the reshaped eeg data [batch sample, ch, time]
        Ai = convmtx_i(e[0], num_context)
        X = np.nan * np.ones((Ai.shape[0], eeg.shape[1], Ai.shape[1]))
        for idx_ch in range(eeg.shape[1]):
            X[:, idx_ch, :] = e[idx_ch][Ai]
            
        X = Variable(torch.from_numpy(X).type('torch.FloatTensor'))
        y = Variable(torch.from_numpy(a_pred).type('torch.FloatTensor'))        

        # For predicting the eeg waveform during training, the unattended audio is not needed
        # Skip to save processing time
        if audio_unatt is None:
            a_unatt = None
            z_unatt = None
        else:
            a_unatt = audio_unatt[idx_sample]
            a_unatt = a_unatt[idx_keep] 
            a_unatt_pred = np.nan * np.ones((num_total_batch, num_predict))
            for idx in range(num_total_batch):
                a_unatt_pred[idx] = a_unatt[np.arange(num_predict) + idx]              
            z_unatt = Variable(torch.from_numpy(a_unatt_pred).type('torch.FloatTensor'))            

        if num_batch is not None:
            idx_keep = np.random.permutation(X.data.size(0))[:num_batch]
            idx_keep = torch.from_numpy(idx_keep).type('torch.LongTensor')
            X = X[idx_keep]
            y = y[idx_keep]
            if z_unatt is not None:
                z_unatt = z_unatt[idx_keep]           
        
    else:
        print('-warning, too little data-')
        X = None
        y = None
        z_unatt = None
        a = None
        a_unatt = None

    return X, y, z_unatt
