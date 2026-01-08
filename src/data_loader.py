"""Load .mat files and extract EEG data, epochs, and channel information"""

import scipy.io as sio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class EEGData:
    """Container for loaded EEG data and metadata"""
    
    def __init__(self, data: pd.DataFrame, visnum: np.ndarray, srate: float, 
                 chanlocs: pd.DataFrame, data_id: str):
        self.data = data  # samples × channels
        self.visnum = visnum  # sleep stages array
        self.srate = srate  # sampling rate
        self.chanlocs = chanlocs  # channel locations DataFrame
        self.data_id = data_id  # dataset identifier


def load_mat_file(filepath: str) -> EEGData:
    """
    Load .mat file and extract EEG data, epochs, and channel information.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        EEGData object containing all extracted information
    """
    # Load mat file
    mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    
    # Extract EEG data
    eeg_struct = mat['EEG']
    data_id = eeg_struct.id
    
    # Get sampling rate
    srate = float(getattr(eeg_struct, 'srate', 125.0))
    
    # Extract EEG data (channels × samples) and transpose to (samples × channels)
    eeg_data = eeg_struct.data.T
    data_df = pd.DataFrame(eeg_data)
    
    # Extract sleep stages (visnum)
    visnum = getattr(eeg_struct, 'visnum', None)
    if visnum is not None:
        if isinstance(visnum, pd.Series):
            visnum = visnum.values
        elif not isinstance(visnum, np.ndarray):
            visnum = np.array(visnum)
    else:
        # Create default array if visnum doesn't exist
        n_epochs = int(len(data_df) / (20 * srate))  # 20 second epochs
        visnum = np.zeros(n_epochs, dtype=int)
    
    # Extract channel locations
    chanlocs = []
    if hasattr(eeg_struct, 'chanlocs') and eeg_struct.chanlocs is not None:
        for chanloc in eeg_struct.chanlocs:
            props = [prop for prop in dir(chanloc) if not prop.startswith('_')]
            chanlocs.append({prop: getattr(chanloc, prop) for prop in props})
    
    chanlocs_df = pd.DataFrame(chanlocs) if chanlocs else pd.DataFrame()
    
    # Reorder columns if 'labels' exists
    if not chanlocs_df.empty and 'labels' in chanlocs_df.columns:
        cols = chanlocs_df.columns.tolist()
        cols.remove('labels')
        cols = ['labels'] + cols
        chanlocs_df = chanlocs_df[cols]
    
    print(f"Loaded EEG data for {data_id}")
    print(f"  Data shape: {data_df.shape} (samples × channels)")
    print(f"  Sampling rate: {srate} Hz")
    print(f"  Number of epochs: {len(visnum)}")
    print(f"  Channels: {len(chanlocs_df) if not chanlocs_df.empty else data_df.shape[1]}")
    
    return EEGData(
        data=data_df,
        visnum=visnum,
        srate=srate,
        chanlocs=chanlocs_df,
        data_id=data_id
    )

