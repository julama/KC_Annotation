"""
BrainAmp data processing functions
Handles loading and processing of BrainVision files (.vhdr, .vmrk, .eeg) using MNE-Python
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Callable

# === Loading BrainVision EEG data =====================================================
def _create_eeg_loader_brainamp(brainvision_loader):
    """Creates a lazy loader function for BrainVision EEG data"""
    def load_and_process():
        # The brainvision_loader is already a function that returns DataFrame
        # We just need to call it and add some logging
        data_df = brainvision_loader()
        # Extract data ID from the file path or use a default
        print(f"Loading BrainVision EEG data with shape {data_df.shape} ...")
        return data_df
    return load_and_process

def eeg_loaders_brainamp(brainvision_data: Dict[str, Callable[[], pd.DataFrame]]) -> Dict[str, Any]:
    """
    Creates lazy loader functions for BrainVision EEG data.
    
    Args:
        brainvision_data: Dictionary mapping file paths to BrainVision loader functions
        
    Returns:
        Dictionary mapping subject IDs to EEG data loader functions
    """
    loaders = {}
    for path, loader in brainvision_data.items():
        # Extract subject ID from file path (e.g., "C-006" from the filename)
        filename = os.path.basename(path)
        # Extract subject ID from filename like "C-006_hdeeg_visit04_sleep_18-09-2025.vhdr"
        subject_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
        loaders[subject_id] = _create_eeg_loader_brainamp(loader)
    return loaders

# === Loading BrainVision channel metadata =============================================
def _create_channel_metadata_loader(brainvision_loader):
    """Creates a lazy loader function for BrainVision channel metadata"""
    def load_and_process():
        # Load the BrainVision file to extract metadata
        import mne
        vhdr_path = brainvision_loader.__self__._filepath if hasattr(brainvision_loader, '__self__') else None
        
        if vhdr_path is None:
            raise ValueError("Cannot extract VHDR path from loader")
        
        # Load BrainVision file using MNE to get metadata
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose='error')
        
        # Extract channel information
        channel_info = []
        for i, ch_name in enumerate(raw.ch_names):
            ch_type = raw.get_channel_types()[i]
            ch_info = raw.info['chs'][i]
            
            channel_info.append({
                'channel_number': i + 1,
                'channel_name': ch_name,
                'channel_type': ch_type,
                'unit': ch_info['unit'],
                'unit_mul': ch_info['unit_mul'],
                'coord_frame': ch_info['coord_frame'],
                'cal': ch_info['cal'],
                'logno': ch_info['logno'],
                'scanno': ch_info['scanno'],
                'kind': ch_info['kind'],
                'range': ch_info['range']
            })
        
        df = pd.DataFrame(channel_info)
        print(f"Loading channel metadata for {len(df)} channels...")
        return df
    
    return load_and_process

def channel_metadata_loaders_brainamp(brainvision_data: Dict[str, Callable[[], pd.DataFrame]]) -> Dict[str, Any]:
    """
    Creates lazy loader functions for BrainVision channel metadata.
    
    Args:
        brainvision_data: Dictionary mapping file paths to BrainVision loader functions
        
    Returns:
        Dictionary mapping subject IDs to channel metadata loader functions
    """
    loaders = {}
    for path, loader in brainvision_data.items():
        # Extract subject ID from file path
        filename = os.path.basename(path)
        subject_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
        loaders[subject_id] = _create_channel_metadata_loader(loader)
    return loaders
