"""
Event detection functions for slow waves and spindles
Consolidated from d02_sw_spindle.py and d03_conclas_sw.py
"""

from typing import Dict, Any, Callable
import pandas as pd
from clas.helpers.sleep_features.slow_wave_detection import slowwave_detection
from clas.helpers.sleep_features.spindle_detection import detect_spindles
from scipy.signal import find_peaks
import numpy as np

# === SW detection with lazy loading ===========================================================
def _create_sw_detection_loader(data_loader, params):
    """Creates a lazy loader function for slow wave detection"""
    def sw_results_loader():
        try:
            # Load the EEG data only when needed
            eeg_data = data_loader()
            
            # Get data in the correct format and preserve channel mapping
            channel_names = None
            if isinstance(eeg_data, pd.DataFrame):
                data_array = eeg_data.values
                channel_names = eeg_data.columns.tolist()  # Preserve original column names
                sample_rate = params.get('sample_rate', 125)
            elif isinstance(eeg_data, dict) and 'data' in eeg_data:
                data_array = eeg_data['data']
                sample_rate = eeg_data.get('sf', params.get('sample_rate', 125))
            else:
                print(f"Unsupported data format for SW detection: {type(eeg_data)}")
                return pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
                
            # Call the actual detection function
            sw_results = slowwave_detection(
                data_array, 
                SampRate=sample_rate,
                WaveStart=params.get('WaveStart', 'NegZeroCrossings'),
                min_freq=params.get('min_freq', 0.5),
                max_freq=params.get('max_freq', 4.0)
            )
            
            # Convert results to DataFrame and remap channel indices to original names
            if sw_results is not None and len(sw_results) > 0:
                results_df = pd.DataFrame(sw_results)
                
                # Remap array indices to original channel names if available
                if channel_names is not None and 'channel' in results_df.columns:
                    results_df['channel'] = results_df['channel'].map(
                        lambda idx: channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else idx
                    )
            else:
                results_df = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
                
            return results_df
            
        except Exception as e:
            print(f"Error during slow wave detection: {e}")
            return pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
    
    return sw_results_loader

def sw_detection(input_data: Dict[str, Callable], params: Dict[str, Any] = None) -> Dict[str, Callable]:
    """
    Creates loader functions for slow wave detection results.
    
    Args:
        input_data: Dict with loader functions for EEG data
        params: Parameters for slow wave detection
        
    Returns:
        Dict with loader functions for SW detection results
    """
    if params is None:
        params = {}
    
    result_loaders = {}
    
    for data_id, data_loader in input_data.items():
        # Create a lazy loader function for this dataset
        result_loaders[data_id] = _create_sw_detection_loader(data_loader, params)
    
    return result_loaders

# === Spindle detection with lazy loading ===========================================================
def _create_spindle_detection_loader(data_loader, epochs_loader, params):
    """Creates a lazy loader function for spindle detection"""
    def spindle_results_loader():
        try:
            # Load the EEG data
            eeg_data = data_loader()
            
            # Load the epochs data to get ndx_epo_good
            epochs_data = epochs_loader()
            ndx_good_epo = None
            if 'is_good_epoch' in epochs_data.columns:
                # Filter out NaN values and convert to integers
                ndx_good_epo = epochs_data['is_good_epoch'].dropna().astype(int).tolist()
                # convert binary to indices
                ndx_good_epo = [i for i, x in enumerate(ndx_good_epo) if x == 1]
                
                # If empty, use a default fallback
                if not ndx_good_epo:
                    print("Warning: No valid clean epochs found")
            else:
                print("Warning: No 'ndx_epo_good' column found in epochs data.")
            
            print(f"Using {len(ndx_good_epo)} clean epochs for spindle detection")
            
            # Get data in the correct format and preserve channel mapping
            channel_names = None
            if isinstance(eeg_data, pd.DataFrame):
                data_array = eeg_data.values
                channel_names = eeg_data.columns.tolist()  # Preserve original column names
                sample_rate = params.get('sample_rate', 125)
            elif isinstance(eeg_data, dict) and 'data' in eeg_data:
                data_array = eeg_data['data']
                sample_rate = eeg_data.get('sf', params.get('sample_rate', 125))
            else:
                print(f"Unsupported data format for spindle detection: {type(eeg_data)}")
                return pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
            
            # Call the actual detection function
            spindle_results = detect_spindles(
                data_array,
                sample_rate,
                ndx_good_epo=ndx_good_epo,
                **params
            )
            
            # Convert results to DataFrame and remap channel indices to original names
            if spindle_results is not None and len(spindle_results) > 0:
                results_df = pd.DataFrame(spindle_results)
                
                # Remap array indices to original channel names if available
                if channel_names is not None and 'channel' in results_df.columns:
                    results_df['channel'] = results_df['channel'].map(
                        lambda idx: channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else idx
                    )
            else:
                results_df = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
                
            return results_df
            
        except Exception as e:
            print(f"Error during spindle detection: {e}")
            return pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency'])
    
    return spindle_results_loader

def spindle_detection(input_data: Dict[str, Callable], epochs_data: Dict[str, Callable], params: Dict[str, Any] = None) -> Dict[str, Callable]:
    """
    Creates loader functions for spindle detection results.
    
    Args:
        input_data: Dict with loader functions for EEG data
        epochs_data: Dict with loader functions for epochs data
        params: Parameters for spindle detection
        
    Returns:
        Dict with loader functions for spindle detection results
    """
    if params is None:
        params = {}
    
    result_loaders = {}
    
    for data_id, data_loader in input_data.items():
        # Get the corresponding epochs loader
        epochs_loader = epochs_data.get(data_id)
        if epochs_loader is None:
            print(f"Warning: No epochs data found for {data_id}")
            continue
            
        # Create a lazy loader function for this dataset
        result_loaders[data_id] = _create_spindle_detection_loader(data_loader, epochs_loader, params)
    
    return result_loaders

# === SW cut functions ===========================================================
def _create_partial_eeg_loader(eeg_loader: Callable, start: int, stop: int):
    def partial_eeg_loader():
        return eeg_loader.__self__._load(start=start, stop=stop)
    return partial_eeg_loader

def sw_cut(sw_df: pd.DataFrame, ds_df: pd.DataFrame, eeg: Dict, params_update: Dict = {}):
    """
    Cut EEG data around slow wave events for analysis.
    
    Args:
        sw_df: DataFrame with slow wave information
        ds_df: DataFrame with dataset information
        eeg: Dict with EEG data loaders
        params_update: Additional parameters to override defaults
        
    Returns:
        Dict with loader functions for cut EEG segments
    """
    params = {
        'sec_before_stim': 5,
        'sec_after_stim': 10
    }
    params.update(params_update)

    loaders = {}

    for index, row in sw_df.iterrows():
        # identify the data id
        sel = (ds_df['subject'] == row['BaseSubject']) & (ds_df['cond'] == row['Tag'])
        if sum(sel) == 0:
            continue
        elif sum(sel) > 1:
            print(f'WARNING: Found {sum(sel)} datasets for {row["BaseSubject"]} - {row["Tag"]}')
            continue
        data_id = ds_df.loc[sel, 'id'].values[0]
        if data_id not in eeg:
            print(f'WARNING: Data id {data_id} not found in eeg')
            continue

        # get the sampling rate
        srate = ds_df.loc[sel, 'srate'].values[0]

        # get the index of the stimulation onset
        index_stim = row['StimOnset']-1

        start = index_stim - params['sec_before_stim'] * srate
        stop = index_stim + params['sec_after_stim'] * srate

        # cut EEG data to match the time range of the SW
        key = f'{data_id}_ON{index_stim:07d}'
        loaders[key] = _create_partial_eeg_loader(eeg[data_id], start, stop)

    return loaders 


# === New Slow Wave Detection Functions =======================================================

def sw_detection_wavelet(real_data: Dict, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Slow wave detection based on wavelet real part (trough detection)
    
    Args:
        real_data: Dictionary of partition_id -> real part data
        params: Detection parameters including threshold
    
    Returns:
        Dictionary of partition_id -> DataFrame with detected slow waves
    """
    from clas.helpers.mymath import norm
    
    results = {}
    sw_threshold = params.get('sw_threshold', -50)  # Default threshold for real part
    
    for partition_id, data in real_data.items():
        try:
            # Handle callable data
            if callable(data):
                data = data()
            
            # Convert to numpy array if DataFrame
            if isinstance(data, pd.DataFrame):
                real_values = data.iloc[:, 0].values if len(data.columns) > 0 else data.values.flatten()
            else:
                real_values = np.array(data).flatten()
            
            # Create feature DataFrame for detection
            feat_df = pd.DataFrame(index=range(len(real_values)))
            feat_df['real'] = real_values
            
            # Add flags for slow wave detection
            # Real minima: local minima in the real part
            feat_df['real_min_flag'] = (
                (feat_df['real'] < feat_df['real'].shift(1)) & 
                (feat_df['real'] < feat_df['real'].shift(-1))
            )
            
            # Slow wave flag: real minima below threshold
            feat_df['sw_flag'] = feat_df['real_min_flag'] & (feat_df['real'] < sw_threshold)
            
            # Get slow wave indices
            sw_indices = feat_df[feat_df['sw_flag']].index.tolist()
            
            if len(sw_indices) > 0:
                # Create results DataFrame
                sw_results = pd.DataFrame({
                    'start': sw_indices,
                    'end': sw_indices,  # Point events
                    'duration': 0,  # Point events have no duration
                    'amplitude': feat_df.loc[sw_indices, 'real'].values,
                    'frequency': 0,  # Not calculated for point events
                    'method': 'wavelet_real'
                })
            else:
                sw_results = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency', 'method'])
            
            results[partition_id] = sw_results
            
        except Exception as e:
            print(f"Error in wavelet SW detection for {partition_id}: {e}")
            results[partition_id] = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency', 'method'])
    
    return results


def sw_detection_hilbert(norm_data: Dict, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Slow wave detection based on Hilbert norm (envelope detection)
    
    Args:
        norm_data: Dictionary of partition_id -> norm data (already computed norm of Hilbert transform)
        params: Detection parameters including prominence and distance
    
    Returns:
        Dictionary of partition_id -> DataFrame with detected slow waves
    """
    results = {}
    srate = params.get('sample_rate', 125)
    prominence = params.get('prominence', 500)
    distance = int(0.25 * srate)  # 0.25 seconds minimum distance between peaks
    
    for partition_id, data in norm_data.items():
        try:
            # Handle callable data
            if callable(data):
                data = data()
            
            # Convert to numpy array if DataFrame
            if isinstance(data, pd.DataFrame):
                norm_values = data.iloc[:, 0].values if len(data.columns) > 0 else data.values.flatten()
            else:
                norm_values = np.array(data).flatten()
            
            # Use the pre-computed norm (equivalent to mm.norm(wdata, axis=1))
            wn = norm_values
            
            # Find peaks in the envelope
            i_peaks, peak_properties = find_peaks(
                wn, 
                distance=distance, 
                prominence=prominence
            )
            
            if len(i_peaks) > 0:
                # Create results DataFrame
                sw_results = pd.DataFrame({
                    'start': i_peaks,
                    'end': i_peaks,  # Point events
                    'duration': 0,  # Point events have no duration
                    'amplitude': wn[i_peaks],
                    'frequency': 0,  # Not calculated for point events
                    'method': 'hilbert_amplitude'
                })
            else:
                sw_results = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency', 'method'])
            
            results[partition_id] = sw_results
            
        except Exception as e:
            print(f"Error in Hilbert SW detection for {partition_id}: {e}")
            results[partition_id] = pd.DataFrame(columns=['start', 'end', 'duration', 'amplitude', 'frequency', 'method'])
    
    return results