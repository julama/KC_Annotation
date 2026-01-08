from kedro.io import AbstractDataset
from kedro.io.core import DatasetError
import os
import pandas as pd
import numpy as np
import mne
from typing import Optional
from clas.helpers.eeg_preprocessing import apply_highpass


class BrainVisionDataset(AbstractDataset):
    """A custom dataset for loading BrainVision files (.vhdr, .vmrk, .eeg) using MNE.

    Loads BrainVision format files and returns pandas DataFrame with EEG data.
    """

    def __init__(self, filepath: str, load_args: dict = {}):
        """
        Initialize BrainVision dataset loader.
        
        Args:
            filepath: Path to the .vhdr file (MNE will find .vmrk and .eeg automatically)
            **kwargs: Additional arguments including:
                - preload: Whether to load data into RAM (faster downstream, but uses more memory)
                - verbose: MNE verbosity level ("error", "warning", "info", "debug")
                - target_sfreq: Target sampling frequency for downsampling (default: 250 Hz)
        """
        self._filepath = filepath
        
        # Extract load_args
        self._load_args = load_args
        
        self._preload = self._load_args.get('preload', True)
        self._verbose = self._load_args.get('verbose', "error")
        self._target_sfreq = self._load_args.get('target_sfreq', 150)

    def _load(self) -> pd.DataFrame:
        """
        Load BrainVision file using MNE and return as pandas DataFrame.
        
        Returns:
            DataFrame with EEG data (samples x channels format)
        """
        if not os.path.exists(self._filepath):
            raise DatasetError(f"BrainVision file not found: {self._filepath}")
        
        # Use instance parameters (set during __init__)
        load_args = self._load_args
        preload = self._preload
        verbose = self._verbose
        target_sfreq = self._target_sfreq
        
        try:
            # Load BrainVision file using MNE
            raw = mne.io.read_raw_brainvision(
                self._filepath, 
                preload=preload, 
                verbose=verbose
            )
            
            # Apply downsampling if target frequency is different from original
            original_sfreq = raw.info['sfreq']
            if target_sfreq != original_sfreq:
                print(f"Downsampling from {original_sfreq} Hz to {target_sfreq} Hz")
                raw.resample(target_sfreq, verbose=verbose)

            # Apply highpass filter using MNE
            print("Applying highpass filter (0.1 Hz)")
            raw.filter(l_freq=0.1, h_freq=None, verbose=verbose)
            
            # Extract data as numpy array (channels x samples)
            data = raw.get_data()
            
            # Convert to float32 to reduce memory usage and file size
            data = data.astype(np.float32)
            
            # Transpose to samples x channels format (consistent with MATLAB data)
            data_transposed = data.T
            
            # Create DataFrame with channel names as columns
            channel_names = raw.ch_names
            df = pd.DataFrame(data_transposed, columns=channel_names)
            
            return df
            
        except Exception as e:
            raise DatasetError(f"Error loading BrainVision file {self._filepath}: {str(e)}")

    def _save(self, data) -> None:
        """Save operation is not implemented for BrainVision files (read-only)."""
        raise DatasetError("Save operation is not implemented for BrainVisionDataset.")

    def _exists(self) -> bool:
        """Check if the BrainVision file exists."""
        return os.path.exists(self._filepath)

    def _describe(self) -> dict:
        """Describe the dataset."""
        return dict(
            filepath=self._filepath,
            preload=self._preload,
            verbose=self._verbose,
            target_sfreq=self._target_sfreq
        )
