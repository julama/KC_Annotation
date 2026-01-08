"""
Functions for slow wave burst detection and analysis.
Extracted from SW_bursts_detection.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.ndimage import label


def sw_bursts_detection(global_events, fs, freq_range=(0.5, 4.0), min_sw_events=3):
    """
    Detects bursts/trains of GLOBAL slow wave events based on inter-event frequency.
    
    A burst is defined as a sequence of consecutive global SW events where the frequency
    between adjacent events (1 / inter-event interval) falls within the specified range.
    
    Parameters:
    - global_events: list of dicts, output from find_global_slow_waves()
                     Each dict has keys: id, start_sample, end_sample, peak_sample, 
                     max_concurrent_channels, involved_channels
    - fs: int, sampling frequency (Hz)
    - freq_range: tuple (min_freq, max_freq) in Hz, default (0.5, 4.0)
    - min_sw_events: int, minimum number of global events to constitute a burst (default 3)
    
    Returns:
    - bursts_df: DataFrame with columns [length_train, mean_freq, start_idx, stop_idx]
    """
    
    min_freq, max_freq = freq_range
    
    # If no events, return empty DataFrame
    if len(global_events) < min_sw_events:
        return pd.DataFrame(columns=['length_train', 'mean_freq', 'start_idx', 'stop_idx'])
    
    # Extract peak times (or could use start_sample) for inter-event interval calculation
    peak_samples = np.array([ev['peak_sample'] for ev in global_events])
    
    # Calculate inter-event intervals (in samples) and frequencies
    intervals_samples = np.diff(peak_samples)
    intervals_sec = intervals_samples / fs
    frequencies = 1.0 / intervals_sec  # Hz
    
    # Mark which intervals fall within the frequency range
    in_range = (frequencies >= min_freq) & (frequencies <= max_freq)
    
    # Find consecutive sequences of in-range intervals
    all_bursts = []
    current_burst_start = None
    current_burst_indices = []
    
    for i, is_valid in enumerate(in_range):
        if is_valid:
            # Valid connection between event i and event i+1
            if current_burst_start is None:
                # Start a new burst
                current_burst_start = i
                current_burst_indices = [i, i+1]
            else:
                # Check if this interval is consecutive with the previous one
                # If current_burst_indices ends with event k, and this is interval i,
                # then we're consecutive if i == k (interval from k to k+1)
                if i == current_burst_indices[-1]:
                    # Consecutive interval, just add next event
                    current_burst_indices.append(i+1)
                else:
                    # Gap detected, save previous burst and start new one
                    if len(current_burst_indices) >= min_sw_events:
                        # Save the burst
                        burst_events_subset = [global_events[idx] for idx in current_burst_indices]
                        burst_freqs = frequencies[current_burst_start:current_burst_indices[-1]]
                        
                        all_bursts.append({
                            'length_train': len(current_burst_indices),
                            'mean_freq': np.mean(burst_freqs),
                            'start_idx': burst_events_subset[0]['start_sample'],
                            'stop_idx': burst_events_subset[-1]['end_sample']
                        })
                    
                    # Start new burst
                    current_burst_start = i
                    current_burst_indices = [i, i+1]
        else:
            # Invalid interval, end current burst if exists
            if current_burst_start is not None and len(current_burst_indices) >= min_sw_events:
                # Save the burst
                burst_events_subset = [global_events[idx] for idx in current_burst_indices]
                burst_freqs = frequencies[current_burst_start:current_burst_indices[-1]]
                
                all_bursts.append({
                    'length_train': len(current_burst_indices),
                    'mean_freq': np.mean(burst_freqs),
                    'start_idx': burst_events_subset[0]['start_sample'],
                    'stop_idx': burst_events_subset[-1]['end_sample']
                })
            
            # Reset burst tracking
            current_burst_start = None
            current_burst_indices = []
    
    # Check if there's a burst at the end
    if current_burst_start is not None and len(current_burst_indices) >= min_sw_events:
        burst_events_subset = [global_events[idx] for idx in current_burst_indices]
        burst_freqs = frequencies[current_burst_start:current_burst_indices[-1]]
        
        all_bursts.append({
            'length_train': len(current_burst_indices),
            'mean_freq': np.mean(burst_freqs),
            'start_idx': burst_events_subset[0]['start_sample'],
            'stop_idx': burst_events_subset[-1]['end_sample']
        })
    
    # Convert to DataFrame
    if all_bursts:
        bursts_df = pd.DataFrame(all_bursts)
    else:
        # Return empty DataFrame with correct columns
        bursts_df = pd.DataFrame(columns=['length_train', 'mean_freq', 'start_idx', 'stop_idx'])
    
    return bursts_df


def find_fss(waves_df, sampling_rate=125, min_fss_length=2, duration_threshold=0.5, amplitude_threshold=75):
    """
    Identifies the Slow Wave Sequence ((F)SS) based on Siclari et al. (2014).
    
    Parameters:
    - waves_df: DataFrame with columns ['wave_start', 'wave_stop', 'amp_peak', 'amp_trough', 'channel']
      (wave_start/wave_stop are sample indices, amplitudes are in microvolts)
    - sampling_rate: Hz (used to define tolerance for "successive" waves and calculate duration)
    - min_fss_length: int, minimum number of consecutive connections (default 2, meaning 3 waves)
    - duration_threshold: float or None, minimum duration in seconds (default 0.5s). If None, no duration filtering.
    - amplitude_threshold: float or None, minimum peak-to-peak amplitude in microvolts (default 75uV). 
                          If None, no amplitude filtering.
    
    Returns:
    - fss_info: List of dictionaries with 'Channel', 'StartSample', 'FirstWaveIndex', 'WaveCount' 
                or None if not found.
    """
    
    # Calculate duration from wave_start and wave_stop
    waves_df = waves_df.copy()
    waves_df['Duration'] = (waves_df['wave_stop'] - waves_df['wave_start']) / sampling_rate

    waves_df['Duration_hw1'] = (waves_df['mid_xing'] - waves_df['wave_start']) / sampling_rate
    waves_df['Duration_hw2'] = (waves_df['wave_stop'] - waves_df['mid_xing']) / sampling_rate
    
    # 1. Filtering based on FSS definition
    # Duration > threshold (if specified)
    # Peak-to-peak amplitude > threshold (if specified)
    filter_mask = pd.Series(True, index=waves_df.index)
    
    if duration_threshold is not None:
        #filter_mask = filter_mask & (waves_df['Duration'] > duration_threshold)
        filter_mask = filter_mask & (waves_df['Duration_hw1'] > duration_threshold)
        filter_mask = filter_mask & (waves_df['Duration_hw2'] > duration_threshold)
    
    if amplitude_threshold is not None:
        # Calculate peak-to-peak amplitude: amp_peak - amp_trough
        peak_to_peak = np.abs(waves_df['amp_peak'] + np.abs(waves_df['amp_trough']))
        filter_mask = filter_mask & (peak_to_peak > amplitude_threshold)
    
    valid_waves = waves_df[filter_mask].copy()
    
    # Sort by time to ensure sequential processing
    valid_waves = valid_waves.sort_values(by=['channel', 'wave_start'])
    
    # We must look for the sequence *within each channel separately*
    channels = valid_waves['channel'].unique()
    
    potential_fss_events = []

    for ch in channels:
        ch_data = valid_waves[valid_waves['channel'] == ch]
        
        # If less than 3 waves exist on this channel, it can't be an FSS
        if len(ch_data) < 3:
            continue
            
        # Convert to numpy arrays for speed
        starts = ch_data['wave_start'].values
        ends = ch_data['wave_stop'].values
        indices = ch_data.index.values
        
        # 2. Sequence Detection
        # Calculate the gap between current wave Start and previous wave End
        # We allow a tiny tolerance (e.g., 10ms) for zero-crossing jitter
        tolerance_samples = int(0.01 * sampling_rate) 
        
        # Logic: Starts[i] should be very close to Ends[i-1]
        # We create a boolean array: True if "Successive", False if "Gap"
        gaps = starts[1:] - ends[:-1]

        is_successive = gaps <= tolerance_samples
        #print(f"Number of successive waves: {np.sum(is_successive)}")
        
        # We now look for runs of 'True' in is_successive with length >= 2
        # (Length 2 means: Wave1-Wave2 is connected AND Wave2-Wave3 is connected = 3 waves)
        
        consecutive_count = 0
        sequence_start_idx = 0
        
        for i, connected in enumerate(is_successive):
            if connected:
                if consecutive_count == 0:
                    # Mark the index of the first wave in this potential chain
                    sequence_start_idx = i 
                consecutive_count += 1
                
                # Check if we hit the threshold (2 connections = 3 waves)
                if consecutive_count >= min_fss_length:
                    # The first wave in this sequence is at indices[sequence_start_idx]
                    first_wave_real_idx = indices[sequence_start_idx]
                    start_time = starts[sequence_start_idx]
                    # The last wave in the sequence is at sequence_start_idx + consecutive_count
                    end_time = ends[sequence_start_idx + consecutive_count]
                    
                    potential_fss_events.append({
                        'Channel': ch,
                        'StartSample': start_time,
                        'EndSample': end_time,
                        'FirstWaveIndex': first_wave_real_idx,
                        'WaveCount': consecutive_count + 1 # +1 because 2 connections = 3 waves
                    })
            else:
                # Connection broken, reset counter
                consecutive_count = 0
    
    # 3. Global Selection
    # The paper defines FSS as the first occurrence across the recording.
    # We sort all found sequences by time and pick the earliest one.
    
    if not potential_fss_events:
        return None
        
    fss_df = pd.DataFrame(potential_fss_events)
    fss_df = fss_df.sort_values(by='StartSample')
    
    # Return all sequences
    return fss_df.to_dict(orient='records')


def analyze_traveling_waves(troughs, channel_coords, 
                            eps_space=35, eps_time=80, min_samples=5):
    """
    Refined clustering removing Amplitude from the distance metric.
    
    Parameters:
    - troughs: list of tuples [(channel_idx, time_ms, amp_trough), ...]
    - channel_coords: dict {channel_idx: (x, y)} positions in mm
    - eps_space: spatial neighborhood distance (mm)
    - eps_time: temporal neighborhood distance (ms)
    - min_samples: minimum detections to form a cluster
    """
    
    # 1. Prepare Data: Use only X, Y, T for clustering
    # We store Amp separately to re-attach it later
    data_for_cluster = []
    metadata = [] # stores (original_index, amplitude, channel_id)
    
    for i, (ch_idx, t_ms, amp) in enumerate(troughs):
        if ch_idx in channel_coords:
            x, y = channel_coords[ch_idx]
            data_for_cluster.append([x, y, t_ms])
            metadata.append({'orig_idx': i, 'amp': amp, 'ch': ch_idx})
            
    X = np.array(data_for_cluster) # Shape: (N, 3) -> X, Y, Time
    
    if len(X) == 0:
        return {}, [], []

    # 2. Scaling
    # We want 1 unit of spatial distance (mm) to equal:
    # - 1 unit of spatial distance (obviously)
    # - X units of time distance
    
    # Scaling Factor:
    # We want 'eps_time' (80ms) to be equivalent to 'eps_space' (35mm)
    # So 80 * factor = 35  -> factor = 35 / 80 = 0.43
    time_scale_factor = eps_space / eps_time
    
    X_scaled = X.copy()
    X_scaled[:, 2] = X[:, 2] * time_scale_factor
    
    # 3. DBSCAN
    # Metric is Euclidean. 
    # Because we scaled T, a distance of 'eps_space' in the 3D space 
    # means "within 35mm space AND within 80ms time" combined.
    db = DBSCAN(eps=eps_space, min_samples=min_samples, metric='euclidean')
    #db = HDBSCAN(min_cluster_size=min_samples, metric='euclidean')
    labels = db.fit_predict(X_scaled)
    
    # 4. Extract Features
    events = {}
    unique_labels = set(labels)
    
    for k in unique_labels:
        if k == -1: continue
        
        mask = (labels == k)
        points_xyz = X[mask] # X, Y, T (unscaled)
        
        # Retrieve metadata for points in this cluster
        cluster_meta = [metadata[i] for i in np.where(mask)[0]]
        amps = [m['amp'] for m in cluster_meta]
        channels = [m['ch'] for m in cluster_meta]
        
        # Sort by time
        sorted_indices = np.argsort(points_xyz[:, 2])
        sorted_points = points_xyz[sorted_indices]
        
        start_point = sorted_points[0]
        end_point = sorted_points[-1]
        
        duration = end_point[2] - start_point[2]
        
        # Spatial Distance (Euclidean on scalp)
        dist_x = end_point[0] - start_point[0]
        dist_y = end_point[1] - start_point[1]
        dist_mm = np.sqrt(dist_x**2 + dist_y**2)
        
        speed = (dist_mm / duration) if duration > 0 else 0 # m/s
        
        # Global Spread
        spread_x = np.ptp(points_xyz[:, 0])
        spread_y = np.ptp(points_xyz[:, 1])
        spatial_extent = np.sqrt(spread_x**2 + spread_y**2)
        
        events[k] = {
            'origin_xy': (start_point[0], start_point[1]),
            'end_xy': (end_point[0], end_point[1]),
            'start_t': start_point[2],
            'end_t': end_point[2],
            'duration_ms': duration,
            'distance_mm': dist_mm,
            'speed_m_s': speed,
            'spatial_extent_mm': spatial_extent,
            'mean_amplitude_uv': np.mean(amps),
            'amplitude_range_uv': np.ptp(amps),
            'size': len(points_xyz),
            'channels': list(set(channels)) # unique channels involved
        }
        
    return events, labels, X

# version of analyze_traveling_waves that includes post-hoc filtering to ensure unique channels per cluster. Could be improved to add hirarchical clustering if conditions are met
def analyze_traveling_waves2(troughs, channel_coords, 
                            eps_space=35, eps_time=80, min_samples=5):
    """
    Refined clustering removing Amplitude from the distance metric.
    Includes Post-Hoc filtering to ensure unique channels per cluster.
    """
    
    # 1. Prepare Data
    data_for_cluster = []
    metadata = [] 
    
    for i, (ch_idx, t_ms, amp) in enumerate(troughs):
        if ch_idx in channel_coords:
            x, y = channel_coords[ch_idx]
            data_for_cluster.append([x, y, t_ms])
            metadata.append({'orig_idx': i, 'amp': amp, 'ch': ch_idx})
            
    X = np.array(data_for_cluster) 
    
    if len(X) == 0:
        return {}, [], []

    # 2. Scaling
    time_scale_factor = eps_space / eps_time
    X_scaled = X.copy()
    X_scaled[:, 2] = X[:, 2] * time_scale_factor
    
    # 3. DBSCAN
    db = DBSCAN(eps=eps_space, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X_scaled)
    
    # 4. Extract Features & Apply Exclusion Criteria
    events = {}
    unique_labels = set(labels)
    
    for k in unique_labels:
        if k == -1: continue
        
        # indices of points in this cluster
        cluster_indices = np.where(labels == k)[0]
        
        # --- EXCLUSION CRITERIA: Unique Channels Only ---
        # We group indices by channel first
        points_by_channel = {}
        for idx in cluster_indices:
            ch = metadata[idx]['ch']
            if ch not in points_by_channel:
                points_by_channel[ch] = []
            points_by_channel[ch].append(idx)
            
        # Select the 'best' point for each channel
        unique_indices = []
        for ch, idx_list in points_by_channel.items():
            if len(idx_list) == 1:
                unique_indices.append(idx_list[0])
            else:
                # Conflict! Multiple detections for same channel.
                # Strategy: Keep the one with highest absolute amplitude
                # You could also choose 'earliest time' if preferred.
                #best_idx = max(idx_list, key=lambda i: abs(metadata[i]['amp']))
                earliest_idx = min(idx_list, key=lambda i: metadata[i]['orig_idx'])
                unique_indices.append(earliest_idx)
        
        # --- UPDATE LABELS: Remove duplicates from the global labels array ---
        # Points that were in cluster_indices but NOT in unique_indices are duplicates
        # We mark them as -1 (noise) so they don't show up in the cluster plot
        duplicates = list(set(cluster_indices) - set(unique_indices))
        if duplicates:
            labels[duplicates] = -1

        # Re-check min_samples after filtering
        if len(unique_indices) < min_samples:
            # Mark remaining points as noise if cluster becomes too small
            labels[unique_indices] = -1 
            continue

        # Now extract data using ONLY the unique points
        points_xyz = X[unique_indices] 
        cluster_meta = [metadata[i] for i in unique_indices]
        
        amps = [m['amp'] for m in cluster_meta]
        channels = [m['ch'] for m in cluster_meta]
        
        # Sort by time
        sorted_indices = np.argsort(points_xyz[:, 2])
        sorted_points = points_xyz[sorted_indices]
        
        start_point = sorted_points[0]
        end_point = sorted_points[-1]
        
        duration = end_point[2] - start_point[2]
        
        # Spatial Distance
        dist_x = end_point[0] - start_point[0]
        dist_y = end_point[1] - start_point[1]
        dist_mm = np.sqrt(dist_x**2 + dist_y**2)
        
        # CORRECTED SPEED CALCULATION (m/s)
        speed = (dist_mm / duration) if duration > 0 else 0 
        
        # Global Spread
        spread_x = np.ptp(points_xyz[:, 0])
        spread_y = np.ptp(points_xyz[:, 1])
        spatial_extent = np.sqrt(spread_x**2 + spread_y**2)
        
        events[k] = {
            'origin_xy': (start_point[0], start_point[1]),
            'end_xy': (end_point[0], end_point[1]),
            'start_t': start_point[2],
            'end_t': end_point[2],
            'duration_ms': duration,
            'distance_mm': dist_mm,
            'speed_m_s': speed,
            'spatial_extent_mm': spatial_extent,
            'mean_amplitude_uv': np.mean(amps),
            'amplitude_range_uv': np.ptp(amps),
            'min_amplitude_uv': np.min(amps),
            'min_amplitude_channel': channels[np.argmin(amps)],
            'min_amplitude_idx': cluster_meta[np.argmin(amps)]['orig_idx'], # Use cluster_meta here
            'size': len(points_xyz),
            'channels': channels 
        }
        
    return events, labels, X

#old function. Not used anymore.
def find_global_slow_waves(n_channels, n_samples, fs, trough_indices, 
                           window_ms=300, threshold_percent=0.3):
    """
    Identifies Global Slow Wave events based on temporal co-occurrence.
    
    Parameters:
    - n_channels: int, total number of EEG channels
    - n_samples: int, total length of recording
    - fs: int, sampling frequency (Hz)
    - trough_indices: dict, keys=channel_idx, values=list of trough sample indices
                      e.g., {0: [100, 1500], 1: [110, 1505]...}
    - window_ms: float, size of the sliding window in milliseconds (default 300ms)
    - threshold_percent: float, fraction of channels required to call it 'global' (0-1)
    
    Returns:
    - global_events: list of dicts, containing start, stop, peak_density, and involved_channels
    - density_signal: array, the time-course of concurrent detections
    """
    
    # 1. Create a binary raster (sparse matrix of detections)
    #    Shape: [n_channels, n_samples]
    #    We sum this immediately to get a 1D timeline of "instantaneous" detections, 
    #    but to support the windowing, we need the sum across channels first.
    
    raw_counts = np.zeros(n_samples)
    
    # Fill the counts
    # If multiple channels have a trough at exactly sample t, raw_counts[t] increases
    for ch, indices in trough_indices.items():
        # Filter indices that are within bounds
        valid_indices = [idx for idx in indices if 0 <= idx < n_samples]
        if valid_indices:
            np.add.at(raw_counts, valid_indices, 1)

    # 2. Define the sliding window (Boxcar kernel)
    #    This smooths the exact timing differences (jitter)
    window_samples = int((window_ms / 1000) * fs)
    kernel = np.ones(window_samples)
    
    # 3. Convolve to get "Density Signal"
    #    Value at t = "How many troughs occured in the window centered at t"
    #    mode='same' keeps the output size equal to n_samples
    density_signal = np.convolve(raw_counts, kernel, mode='same')
    
    # 4. Apply Threshold
    #    How many channels must be active simultaneously?
    min_channels = n_channels * threshold_percent
    binary_threshold_mask = density_signal >= min_channels
    
    # 5. Group consecutive time points into "Events"
    #    scipy.ndimage.label finds connected regions of 1s
    labeled_array, num_features = label(binary_threshold_mask)
    
    global_events = []
    
    # Iterate through found blobs to extract metadata
    for i in range(1, num_features + 1):
        # Find indices where this event is active
        event_indices = np.where(labeled_array == i)[0]
        
        t_start = event_indices[0]
        t_end = event_indices[-1]
        
        # Find the moment of maximum synchrony (peak of density)
        # We look at the density signal within this event's duration
        segment_density = density_signal[t_start:t_end+1]
        peak_relative_idx = np.argmax(segment_density)
        t_peak = t_start + peak_relative_idx
        peak_count = segment_density[peak_relative_idx]
        
        # Optional: Identify exactly WHICH channels contributed to this specific event.
        # We look at the original trough_indices in the range [t_start - window/2, t_end + window/2]
        # (The buffer is needed because the density signal is convolved/spread out)
        buffer = window_samples // 2
        search_start = max(0, t_start - buffer)
        search_end = min(n_samples, t_end + buffer)
        
        involved_channels = []
        for ch, indices in trough_indices.items():
            # Check if this channel has any trough in the event window
            # Using numpy for speed here would be better for massive data, 
            # but list comprehension is fine for readable prototypes.
            if any(search_start <= idx <= search_end for idx in indices):
                involved_channels.append(ch)
        
        global_events.append({
            'id': i,
            'start_sample': t_start,
            'end_sample': t_end,
            'peak_sample': t_peak,
            'max_concurrent_channels': peak_count,
            'involved_channels': involved_channels
        })
        
    return global_events, density_signal

