#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

import numpy as np
from scipy.signal import find_peaks


def nextpow2(length):
    """
    return the next power of 2 of a given number
    """
    # Returns next power of two following 'number'
    return np.ceil(np.log2(length))

def pad_nextpow2(dat):
    """
    return an array pad with zero to the next power of 2 of the input
    """
    g = nextpow2(np.shape(dat))
    ze=np.zeros(np.array(np.power(2,g) - np.shape(dat),dtype='int'))
    data = np.append(dat,ze)
    return data

def find_nearest(array, value):
    """
    Find the index of the closest element of array to value
    :param array: (n_value,)
    :param value: wanted value
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def EpochData(peaks, data, post_peak, pre_peak, Fs):
    """
    return an epochs of shape (n_peak, M) with M length of the time windows in points M = (post_peak+pre_peak) * Fs
    :param peaks: onsets of peaks (in point)
    :param data: raw data
    :param post_peak: post peaks windows (in sec)
    :param pre_peak: pre peaks windows (in sec)
    :param Fs: Sampling frequency
    :return: epochs of shape (n_peak, M)
    """
    if len(peaks) == 0:
        # Return empty array with correct shape
        expected_length = int((post_peak + pre_peak) * Fs)
        return np.zeros((0, expected_length))
    
    nb_peak = len(peaks)
    import matplotlib.pyplot as plt
    expected_length = int((post_peak + pre_peak) * Fs)
    ds_epochs = np.zeros((nb_peak, expected_length))
    
    pre_samples = int(pre_peak * Fs)
    post_samples = int(post_peak * Fs)
    
    valid_peaks = []
    valid_indices = []
    
    for j in range(nb_peak):
        peak_idx = int(peaks[j])
        start_idx = peak_idx - pre_samples
        end_idx = peak_idx + post_samples
        
        # Check if peak has enough data around it
        if start_idx < 0 or end_idx > len(data):
            # Skip peaks that are too close to boundaries
            continue
        
        temp = np.array(data[start_idx:end_idx])
        if len(temp) != expected_length:
            # Skip if we don't have the right length
            continue
            
        temp = temp - np.mean(temp[0:int(pre_samples/2)])
        
        try:
            ds_epochs[len(valid_peaks), :] = temp
            valid_peaks.append(peak_idx)
            valid_indices.append(j)
        except Exception as e:
            # If there's still an issue, skip this peak
            continue
    
    # Return only valid epochs
    if len(valid_peaks) == 0:
        return np.zeros((0, expected_length))
    
    return ds_epochs[:len(valid_peaks), :]

def findN2peaks(Stages, data, Fs, min, distance, criteria=30):
    """
    Find all peaks superior to the amplitude threshold 'min' in a given sleep stage
    :param Stages: Dataframe with a colonum 'onset' of onset of wanted sleep stages (in sec)
    :param data: Raw data
    :param Fs: Sampling frequency
    :param min: Amplitude threshold (in volt)
    :param distance: minimum distance between two peaks (in sec)
    :param criteria: length of sleep stage, 20 sec for RK ('default') or 30 sec for AASM criteria
    :return:
    """
    p = []
    stages = []
    
    # Need at least 4 rows to use [2:-2] slicing
    if len(Stages) < 4:
        print(f"Warning: Only {len(Stages)} stages provided, but need at least 4 for findN2peaks. "
              f"Using all available stages (may be less robust).")
        stag = Stages['label'].values
        onset = np.round(Stages['onset'].values * Fs).astype('int')
    else:
        stag = Stages['label'].values[2:-2]
        onset = np.round(Stages['onset'].values[2:-2] * Fs).astype('int')

    #print(onset)
    duration = np.round(criteria * Fs).astype('int')

    for j in np.arange(len(onset)):
        low_bound = onset[j]
        upper_bound = onset[j] + duration
        # Ensure bounds are within data range
        if upper_bound > len(data):
            upper_bound = len(data)
        if low_bound >= upper_bound:
            continue
        data_for_peak = data[low_bound:upper_bound] - np.mean(data[low_bound:upper_bound])
        temp, _ = find_peaks(data_for_peak, height=min, distance=distance * Fs)
        p.append(temp + onset[j])
        stages.append(np.ones(len(temp))*stag[j])
    
    # Handle empty arrays
    if len(p) == 0:
        return np.array([]), np.array([])
    elif all(len(arr) == 0 for arr in p):
        return np.array([]), np.array([])
    
    return np.hstack(p), np.hstack(stages)

