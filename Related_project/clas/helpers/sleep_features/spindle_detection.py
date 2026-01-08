import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def detect_spindles(data, sf, threshold, epochl, chOfInterest, ndxGreatEpo, timeframe=None, n_jobs=-2):
    """
    Detect sleep spindles in clean EEG epochs.
    
    Parameters:
    -----------
    data : ndarray
        EEG data. If not in (data points x channels) format, it is transposed.
    sf : float
        Sampling frequency.
    threshold : float
        Upper threshold used to detect sleep spindles.
    epochl : int or float
        Epoch length in seconds.
    chOfInterest : list or ndarray
        List of channel indices (0-indexed) to analyze.
    ndxGreatEpo : list or ndarray
        List of "clean" epoch indices (0-indexed).
    timeframe : int, optional
        Number of epochs (clean epochs) to include. Default is all epochs.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is -2 (all CPUs but one).
        
    Returns:
    --------
    df_events : pd.DataFrame
        One row per detected spindle. Columns include:
          'channel', 'spistart_sample', 'spiend_sample', 'duration_sec',
          'spimaxamp', 'spimaxamp_sample', and 'spifrq'.
    df_summary : pd.DataFrame
        One row per channel with summary statistics:
          'channel', 'duration_mn', 'maxamp_mn', 'spifrq_mn',
          'number_all', 'density_all', and (if available) 'spimaxamp_sample'.
    epochsamples_all : np.ndarray
        A 1D array of sample indices corresponding to the filtered epochs.
    """
    # Convert input lists to numpy arrays for faster processing
    if not isinstance(chOfInterest, np.ndarray):
        chOfInterest = np.array(chOfInterest)
    if not isinstance(ndxGreatEpo, np.ndarray):
        ndxGreatEpo = np.array(ndxGreatEpo)
        
    # Ensure data is in (data points x channels) format.
    if data.shape[0] < data.shape[1]:
        data = data.T

    # Initialize constants
    half_sf = sf / 2.0
    epochfractmin = epochl / 60.0
    nch = data.shape[1]
    numEpo = len(ndxGreatEpo)
    
    # Check if last epoch would extend past data length and adjust numEpo
    last_epoch_end = (ndxGreatEpo[numEpo - 1] + 1) * epochl * sf + 32  # 32 is Overlap
    if last_epoch_end > data.shape[0]:
        numEpo = numEpo - 1
    
    # Pre-compute the bandpass filter coefficients once
    Wp = np.array([11, 16]) / half_sf
    Ws = np.array([6, 28]) / half_sf
    Rp, Rs = 3, 40
    n_order, Wn = signal.cheb2ord(Wp, Ws, Rp, Rs)
    bbp, abp = signal.cheby2(n_order, Rs, Wn, btype='bandpass')
    
    # Set constant values
    Overlap = 32
    min_spindle_duration = 0.3  # minimum duration in seconds
    
    # Pre-calculate epoch boundaries for all epochs once
    # use 0-indexed epochs
    epoch_boundaries = np.zeros((numEpo, 2), dtype=np.int64)
    for i, epoch_idx in enumerate(ndxGreatEpo[:numEpo]):
        start_idx = max(0, int(epoch_idx * epochl * sf - Overlap + 1))
        end_idx = min(data.shape[0] - 1, int((epoch_idx + 1) * epochl * sf + Overlap))
        epoch_boundaries[i] = [start_idx, end_idx]
    
    # Process only channels of interest
    print(f"Processing {len(chOfInterest)} channels in parallel with {n_jobs} cores: ", end="")
    
    # Function to process a single channel
    def process_channel(channel):
        if channel >= nch:
            return None, None
            
        print(f"{channel}, ", end="", flush=True)
        
        # Pre-allocate structure for this channel
        channel_spindles = {
            'duration_mn': np.nan,
            'maxamp_mn': np.nan,
            'spifrq_mn': np.nan,
            'number_all': 0,
            'density_all': np.nan,
            'duration_all': np.array([], dtype=np.float64),
            'maxamp_all': np.array([], dtype=np.float64),
            'spifrq_all': np.array([], dtype=np.float64),
            'spistart': np.array([], dtype=np.int64),
            'spiend': np.array([], dtype=np.int64),
            'spimaxamp_sample': np.array([], dtype=np.int64)
        }
        
        # Extract this channel's time series
        datar = data[:, channel].astype(np.float64)  # Use float64 for better precision
        
        # Pre-allocate arrays for this channel
        filt_data = np.array([], dtype=np.float64)
        epochsamples = np.array([], dtype=np.int64)
        
        # Process each epoch for this channel
        for i in range(numEpo):
            start_idx, end_idx = epoch_boundaries[i]
            
            # Skip if out of bounds
            if start_idx < 0 or end_idx >= len(datar):
                continue
                
            # Extract epoch data
            segment = datar[start_idx:end_idx + 1]
            
            # Apply zero-phase filtering
            filtered_epoch = signal.filtfilt(bbp, abp, segment)
            
            # Get sample indices
            current_samples = np.arange(start_idx, end_idx + 1)
            
            # Remove overlap
            filtered_epoch = filtered_epoch[Overlap:int(epochl * sf + Overlap)]
            current_samples = current_samples[Overlap:int(epochl * sf + Overlap)]
            
            # Append to filtered data
            filt_data = np.append(filt_data, filtered_epoch)
            epochsamples = np.append(epochsamples, current_samples)
        
        # Skip if no data
        if len(filt_data) == 0:
            return None, epochsamples
            
        # Get absolute values
        absfilt_data = np.abs(filt_data)
        
        # Optimized peak detection
        # Compare current point with neighbors
        peak_mask = np.zeros(len(absfilt_data), dtype=bool)
        
        # Handle interior points efficiently
        interior = np.arange(1, len(absfilt_data)-1)
        peak_mask[interior] = ((absfilt_data[interior] >= absfilt_data[interior-1]) & 
                              (absfilt_data[interior] >= absfilt_data[interior+1]))
        
        # Get peak indices and values
        indexpeak = np.where(peak_mask)[0]
        if len(indexpeak) == 0:
            return None, epochsamples
            
        sizepeak = absfilt_data[indexpeak]
        
        # Identify local maxima among peaks
        if len(sizepeak) <= 2:
            return None, epochsamples
            
        # Find local maxima efficiently
        peak_indices = np.arange(len(sizepeak))
        interior_peaks = peak_indices[1:-1]
        
        # Local maxima mask
        max_mask = np.zeros(len(sizepeak), dtype=bool)
        max_mask[interior_peaks] = ((sizepeak[interior_peaks] >= sizepeak[interior_peaks-1]) & 
                                   (sizepeak[interior_peaks] >= sizepeak[interior_peaks+1]))
        
        # Get maxima indices and values
        max_indices = np.where(max_mask)[0]
        if len(max_indices) == 0:
            return None, epochsamples
            
        max_values = sizepeak[max_indices]
        max_original_indices = indexpeak[max_indices]
        
        # Find most common peak amplitude using histogram
        hist, bin_edges = np.histogram(max_values, bins=120)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        maxnn = bin_centers[np.argmax(hist)]
        
        # Local minima mask
        min_mask = np.zeros(len(sizepeak), dtype=bool)
        min_mask[interior_peaks] = ((sizepeak[interior_peaks] <= sizepeak[interior_peaks-1]) & 
                                   (sizepeak[interior_peaks] <= sizepeak[interior_peaks+1]))
        
        # Get minima indices and values
        min_indices = np.where(min_mask)[0]
        if len(min_indices) == 0:
            return None, epochsamples
            
        min_values = sizepeak[min_indices]
        min_original_indices = indexpeak[min_indices]
        
        # Apply thresholds
        low_threshold_indices = min_indices[min_values < 2 * maxnn]
        low_threshold_original = min_original_indices[min_values < 2 * maxnn]
        
        # Calculate mean once for upper threshold
        mean_abs_data = np.mean(absfilt_data)
        upper_threshold = threshold * mean_abs_data
        
        # Find peaks above upper threshold
        up_threshold_indices = max_indices[max_values > upper_threshold]
        if len(up_threshold_indices) == 0:
            return None, epochsamples
            
        up_threshold_original = max_original_indices[max_values > upper_threshold]
        up_threshold_values = max_values[max_values > upper_threshold]
        
        # Spindle detection stage
        # Pre-allocate arrays for detected spindles
        spistart = []
        spiend = []
        spimaxamp = []
        spimaxamp_sample = []
        
        # Process each peak above threshold
        for z in range(len(up_threshold_indices)):
            current_max_idx = up_threshold_original[z]
            current_max_amp = absfilt_data[current_max_idx]
            
            # Find minima before current peak
            if z == 0:
                ndxmaxbef = np.arange(0, current_max_idx + 1)
            else:
                ndxmaxbef = np.arange(up_threshold_original[z - 1], current_max_idx + 1)
                
            ndxminbef = np.intersect1d(ndxmaxbef, low_threshold_original)
            
            # Find minima after current peak
            if z == len(up_threshold_indices) - 1:
                ndxmaxaft = np.arange(current_max_idx, len(absfilt_data))
            else:
                ndxmaxaft = np.arange(current_max_idx, up_threshold_original[z + 1] + 1)
                
            ndxminaft = np.intersect1d(ndxmaxaft, low_threshold_original)
            
            # Process based on found minima
            if len(ndxminbef) > 0 and len(ndxminaft) > 0:
                spistart.append(ndxminbef[-1])
                spiend.append(ndxminaft[0])
                spimaxamp.append(current_max_amp)
                spimaxamp_sample.append(current_max_idx)
            elif len(ndxminbef) > 0 and len(ndxminaft) == 0:
                spistart.append(ndxminbef[-1])
                spimaxamp.append(current_max_amp)
                spimaxamp_sample.append(current_max_idx)
            elif len(ndxminbef) == 0 and len(ndxminaft) > 0 and len(spistart) > 0:
                if spimaxamp[-1] < current_max_amp:
                    spimaxamp[-1] = current_max_amp
                    spimaxamp_sample[-1] = current_max_idx
                spiend.append(ndxminaft[0])
            elif len(spistart) > 0:
                if spimaxamp[-1] < current_max_amp:
                    spimaxamp[-1] = current_max_amp
                    spimaxamp_sample[-1] = current_max_idx
        
        # Convert to numpy arrays
        spistart = np.array(spistart, dtype=np.int64)
        spiend = np.array(spiend, dtype=np.int64)
        spimaxamp = np.array(spimaxamp, dtype=np.float64)
        spimaxamp_sample = np.array(spimaxamp_sample, dtype=np.int64)
        
        # Check if we have complete spindles (start and end)
        if len(spistart) > len(spiend):
            # Remove incomplete spindles
            spistart = spistart[:len(spiend)]
            spimaxamp = spimaxamp[:len(spiend)]
            spimaxamp_sample = spimaxamp_sample[:len(spiend)]
        
        # Skip if no complete spindles found
        if len(spistart) == 0:
            return None, epochsamples
            
        # Calculate spindle frequency
        spifrqi = np.zeros(len(spistart), dtype=np.float64)
        for p in range(len(spistart)):
            ndxspi = np.arange(spistart[p], spiend[p] + 1)
            spipeaks = np.intersect1d(ndxspi, indexpeak)
            spifrqi[p] = len(spipeaks) / ((spiend[p] - spistart[p]) / sf) / 2
        
        # Store spindle data
        channel_spindles['duration_all'] = (spiend - spistart) / sf
        channel_spindles['maxamp_all'] = spimaxamp
        channel_spindles['spifrq_all'] = spifrqi
        channel_spindles['spistart'] = spistart
        channel_spindles['spiend'] = spiend
        channel_spindles['spimaxamp_sample'] = spimaxamp_sample
        
        # 1. Frequency filter (11-16 Hz)
        # 2. Duration filter (>= 0.3s)
        # 3. Timeframe filter
        
        # Create combined mask
        if len(spifrqi) > 0:
            freq_mask = (spifrqi >= 11) & (spifrqi <= 16)
            duration_mask = channel_spindles['duration_all'] >= min_spindle_duration
            
            # Apply timeframe filter if specified
            if timeframe is not None:
                time_limit = timeframe * sf * epochl
                time_mask = spistart <= time_limit
                # Combine all masks
                valid_mask = freq_mask & duration_mask & time_mask
            else:
                valid_mask = freq_mask & duration_mask
            
            # Apply combined mask to all arrays
            channel_spindles['spistart'] = spistart[valid_mask]
            channel_spindles['spiend'] = spiend[valid_mask]
            channel_spindles['duration_all'] = channel_spindles['duration_all'][valid_mask]
            channel_spindles['maxamp_all'] = spimaxamp[valid_mask]
            channel_spindles['spifrq_all'] = spifrqi[valid_mask]
            channel_spindles['spimaxamp_sample'] = spimaxamp_sample[valid_mask]
            
            # Update summary statistics
            if np.any(valid_mask):
                channel_spindles['duration_mn'] = np.mean(channel_spindles['duration_all'])
                channel_spindles['maxamp_mn'] = np.mean(channel_spindles['maxamp_all'])
                channel_spindles['spifrq_mn'] = np.mean(channel_spindles['spifrq_all'])
                channel_spindles['number_all'] = len(channel_spindles['spistart'])
                
                # Use timeframe or numEpo for density calculation
                if timeframe is not None:
                    channel_spindles['density_all'] = channel_spindles['number_all'] / (epochfractmin * timeframe)
                else:
                    channel_spindles['density_all'] = channel_spindles['number_all'] / (epochfractmin * numEpo)
        
        return channel_spindles, epochsamples
    
    # Execute channel processing in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_channel)(channel) for channel in chOfInterest
    )
    
    # Initialize storage for results
    spindles = [None] * nch
    all_epochsamples = []
    
    # Process results from parallel execution
    for channel, (channel_spindles, epochsamples) in zip(chOfInterest, results):
        if channel_spindles is not None:
            spindles[channel] = channel_spindles
        else:
            # Initialize empty structure for channels with no spindles
            spindles[channel] = {
                'duration_mn': np.nan,
                'maxamp_mn': np.nan,
                'spifrq_mn': np.nan,
                'number_all': 0,
                'density_all': np.nan,
                'duration_all': np.array([], dtype=np.float64),
                'maxamp_all': np.array([], dtype=np.float64),
                'spifrq_all': np.array([], dtype=np.float64),
                'spistart': np.array([], dtype=np.int64),
                'spiend': np.array([], dtype=np.int64),
                'spimaxamp_sample': np.array([], dtype=np.int64)
            }
            
        if epochsamples is not None and len(epochsamples) > 0:
            all_epochsamples.append(epochsamples)
    
    print("Success!")
    
    # Prepare DataFrames for return
    summary_list = []
    events_all = []
    
    # Process each channel for summary and events
    for ch in range(nch):
        if ch not in chOfInterest or spindles[ch] is None:
            continue
            
        # Add channel to summary
        summary = {
            'channel': ch,
            'duration_mn': spindles[ch]['duration_mn'],
            'maxamp_mn': spindles[ch]['maxamp_mn'],
            'spifrq_mn': spindles[ch]['spifrq_mn'],
            'number_all': spindles[ch]['number_all'],
            'density_all': spindles[ch]['density_all']
        }
        
        # Handle spimaxamp_sample
        if len(spindles[ch]['spimaxamp_sample']) > 0:
            summary['spimaxamp_sample'] = spindles[ch]['spimaxamp_sample']
        else:
            summary['spimaxamp_sample'] = np.nan
            
        summary_list.append(summary)
        
        # Add each spindle event
        if len(spindles[ch]['spistart']) > 0:
            for i in range(len(spindles[ch]['spistart'])):
                events_all.append({
                    'channel': ch,
                    'spistart_sample': spindles[ch]['spistart'][i],
                    'spiend_sample': spindles[ch]['spiend'][i],
                    'duration_sec': spindles[ch]['duration_all'][i],
                    'spimaxamp': spindles[ch]['maxamp_all'][i],
                    'spimaxamp_sample': spindles[ch]['spimaxamp_sample'][i],
                    'spifrq': spindles[ch]['spifrq_all'][i]
                })
    
    # Create final DataFrames
    df_summary = pd.DataFrame(summary_list).sort_values('channel').reset_index(drop=True)
    df_events = pd.DataFrame(events_all)
    
    # Concatenate epoch samples
    if all_epochsamples:
        epochsamples_all = np.concatenate(all_epochsamples)
    else:
        epochsamples_all = np.array([])
        
    return df_events, df_summary, epochsamples_all


