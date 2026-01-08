import os
from kedro.io import AbstractDataset
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import h5py

# Import preprocessing functions from helpers
from clas.helpers.eeg_preprocessing import (
    apply_reference, 
    apply_bandpass, 
    apply_wavelet, 
    apply_hilbert,
    apply_resample,
    exclude_bad_channels
)


# Single, flexible HDF5Dataset class
class HDF5Dataset(AbstractDataset):
    def __init__(self, filepath: str, **kwargs):
        # Store the filepath and key for HDF5 access
        self._filepath = filepath
        
        # Extract load_args if they're nested (Kedro PartitionedDataset pattern)
        load_args = kwargs.get('load_args', {})
        
        self._key = load_args.get('key', 'data')
        
        # Compression settings
        self._use_compression = load_args.get('use_compression', True)  # Default to compression
        self._compression_level = load_args.get('compression_level', 9)  # 0-9, higher = better compression
        self._compression_lib = load_args.get('compression_lib', 'blosc')  # 'blosc', 'zlib', 'lzf', 'gzip'
        
        # Extract any preprocessing parameters
        self._reference_channels = load_args.get('reference_channels')
        self._exclude_channels = load_args.get('exclude_channels')
        self._min_freq = load_args.get('min_freq')
        self._max_freq = load_args.get('max_freq')
        self._freq = load_args.get('freq')
        self._cycles = load_args.get('cycles')
        self._sf = load_args.get('sf', 125)
        self._apply_hilbert = load_args.get('apply_hilbert', False)
        self._resample_rate = load_args.get('resample_rate')  # Target sampling rate in Hz

    def _load(self, start: Optional[int] = None, stop: Optional[int] = None):
        """
        Load data from HDF5 file and apply preprocessing if specified.
        Supports loading a specific slice of data using start and stop indices.
        """
        print("Loading data with HDF5Dataset")
        # Load the raw data, potentially a slice
        eeg_data = self._load_raw_data(start=start, stop=stop)
        
        # Apply resampling first if specified (before other preprocessing)
        if self._resample_rate:
            print("Applying resampling")
            eeg_data = apply_resample(eeg_data, {
                'original_sf': self._sf,
                'target_sf': self._resample_rate
            })
        
        # Apply preprocessing based on parameters
        if self._reference_channels:
            print("Applying reference")
            eeg_data = apply_reference(eeg_data, self._reference_channels, self._exclude_channels)

        # Exclude bad channels before transformations
        if self._exclude_channels:
            print("Excluding bad channels")
            eeg_data = exclude_bad_channels(eeg_data, self._exclude_channels)

        
        if self._min_freq and self._max_freq:
            # Use resampled sampling rate if available, otherwise original
            print("Applying bandpass filter")
            current_sf = self._resample_rate if self._resample_rate else self._sf
            eeg_data = apply_bandpass(eeg_data, {
                'min_freq': self._min_freq,
                'max_freq': self._max_freq,
                'sf': current_sf
            })
        
        if self._freq and self._cycles:
            print("Applying wavelet transform")
            # Use resampled sampling rate if available, otherwise original
            current_sf = self._resample_rate if self._resample_rate else self._sf
            eeg_data = apply_wavelet(eeg_data, {
                'freq': self._freq,
                'cycles': self._cycles,
                'sf': current_sf
            })
        
        if self._apply_hilbert:
            print("Applying Hilbert transform")
            # Use resampled sampling rate if available, otherwise original
            current_sf = self._resample_rate if self._resample_rate else self._sf
            eeg_data = apply_hilbert(eeg_data, {
                'sf': current_sf
            })
        
        return pd.DataFrame(eeg_data)

    def _load_raw_data(self, start: Optional[int] = None, stop: Optional[int] = None):
        """
        Load raw data from HDF5 file without preprocessing.
        Supports loading a specific slice of data using start and stop indices.
        """
        try:
            # Try to load as a regular pandas DataFrame first, applying slicing
            eeg_data = pd.read_hdf(self._filepath, key=self._key, start=start, stop=stop)
        except Exception:
            # If that fails, it might be a different HDF5 format.
            # Fallback to h5py and then slice. This is less memory-efficient.
            with h5py.File(self._filepath, 'r') as f:
                if self._key in f and 'data' in f[self._key]:
                    data_group = f[self._key]['data']
                    if 'block0_values' in data_group and 'block0_items' in data_group:
                        values = data_group['block0_values'][start:stop]
                        columns = data_group['block0_items'][:]
                        eeg_data = pd.DataFrame(values, columns=columns)
                    else:
                        eeg_data = pd.DataFrame(data_group[start:stop])
                else:
                    eeg_data = pd.DataFrame(f[self._key][start:stop])
        
        return eeg_data

    def _save(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Save data to HDF5 file with optional compression"""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Save with configurable compression settings
        if self._use_compression:
            data.to_hdf(
                self._filepath, 
                key=self._key, 
                complevel=self._compression_level, 
                complib=self._compression_lib
            )
        else:
            # Save without compression for raw data preservation
            data.to_hdf(self._filepath, key=self._key)

    def _exists(self) -> bool:
        """Check if the dataset exists"""
        return os.path.exists(self._filepath)

    def _describe(self) -> dict:
        """Describe the dataset"""
        return dict(
            filepath=self._filepath,
            key=self._key,
            use_compression=self._use_compression,
            compression_level=self._compression_level,
            compression_lib=self._compression_lib,
            reference_channels=self._reference_channels,
            exclude_channels=self._exclude_channels,
            min_freq=self._min_freq,
            max_freq=self._max_freq,
            freq=self._freq,
            cycles=self._cycles,
            sf=self._sf,
            apply_hilbert=self._apply_hilbert,
            resample_rate=self._resample_rate
        )
