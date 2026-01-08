"""Optional preprocessing: re-referencing and bandpass filtering"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Optional, Tuple


def apply_reference(eeg_data: pd.DataFrame, reference: Optional[str] = None) -> pd.DataFrame:
    """
    Apply re-referencing to EEG data.
    
    Args:
        eeg_data: DataFrame with shape (samples × channels)
        reference: Reference type. Options:
            - 'average' or 'common': Common average reference
            - None: No re-referencing
            
    Returns:
        Re-referenced DataFrame
    """
    if reference is None or reference.lower() == 'none':
        return eeg_data.copy()
    
    if reference.lower() in ['average', 'common']:
        # Common average reference: subtract mean across channels for each sample
        data_array = eeg_data.values
        mean_across_channels = np.mean(data_array, axis=1, keepdims=True)
        referenced_data = data_array - mean_across_channels
        return pd.DataFrame(referenced_data, columns=eeg_data.columns, index=eeg_data.index)
    
    raise ValueError(f"Unknown reference type: {reference}")


def apply_bandpass_filter(eeg_data: pd.DataFrame, low_freq: float, high_freq: float, 
                          sampling_rate: float, order: int = 4) -> pd.DataFrame:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        eeg_data: DataFrame with shape (samples × channels)
        low_freq: Low cutoff frequency (Hz)
        high_freq: High cutoff frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        order: Filter order (default: 4)
        
    Returns:
        Filtered DataFrame
    """
    nyquist = sampling_rate / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = signal.filtfilt(b, a, eeg_data.values, axis=0)
    
    return pd.DataFrame(filtered_data, columns=eeg_data.columns, index=eeg_data.index)


def preprocess_eeg(eeg_data: pd.DataFrame, sampling_rate: float,
                   reference: Optional[str] = None,
                   bandpass: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Apply preprocessing steps to EEG data.
    
    Args:
        eeg_data: DataFrame with shape (samples × channels)
        sampling_rate: Sampling rate (Hz)
        reference: Reference type ('average', 'common', or None)
        bandpass: Tuple of (low_freq, high_freq) for bandpass filtering, or None
        
    Returns:
        Preprocessed DataFrame
    """
    processed_data = eeg_data.copy()
    
    # Apply re-referencing
    if reference is not None:
        processed_data = apply_reference(processed_data, reference)
    
    # Apply bandpass filtering
    if bandpass is not None:
        low_freq, high_freq = bandpass
        processed_data = apply_bandpass_filter(processed_data, low_freq, high_freq, sampling_rate)
    
    return processed_data

