"""Manage epochs: filter by sleep stage and handle navigation"""

import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path
# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class EpochManager:
    """Manages epoch filtering and navigation"""
    
    def __init__(self, eeg_data: pd.DataFrame, visnum: np.ndarray, 
                 sampling_rate: float, epoch_length_sec: float = None,
                 sleep_stages: Optional[List[int]] = None):
        """
        Initialize epoch manager.
        
        Args:
            eeg_data: EEG data DataFrame (samples × channels)
            visnum: Sleep stage array (one per epoch)
            sampling_rate: Sampling rate (Hz)
            epoch_length_sec: Length of each epoch in seconds
            sleep_stages: List of sleep stages to include (None = all)
        """
        self.eeg_data = eeg_data
        self.visnum = visnum
        self.sampling_rate = sampling_rate
        self.epoch_length_sec = epoch_length_sec or config.EPOCH_LENGTH_SECONDS
        self.sleep_stages = sleep_stages or config.DEFAULT_SLEEP_STAGES
        
        # Calculate samples per epoch
        self.samples_per_epoch = int(self.epoch_length_sec * self.sampling_rate)
        
        # Create epochs DataFrame
        self.epochs_df = self._create_epochs_df()
        
        # Filter epochs by sleep stage
        self.filtered_epochs_df = self._filter_epochs()
        
        # Current epoch index (in filtered list)
        self.current_epoch_idx = 0
        
        print(f"Total epochs: {len(self.epochs_df)}")
        print(f"Filtered epochs (stages {self.sleep_stages}): {len(self.filtered_epochs_df)}")
    
    def _create_epochs_df(self) -> pd.DataFrame:
        """Create DataFrame with epoch information"""
        n_samples = len(self.eeg_data)
        n_epochs = len(self.visnum)
        
        epochs = []
        for epoch_idx in range(n_epochs):
            start_idx = int(epoch_idx * self.samples_per_epoch)
            end_idx = int(min((epoch_idx + 1) * self.samples_per_epoch - 1, n_samples - 1))
            
            # Get sleep stage for this epoch
            sleep_stage = self.visnum[epoch_idx] if epoch_idx < len(self.visnum) else 0
            
            epochs.append({
                'epoch_idx': epoch_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'sleep_stage': int(sleep_stage) if hasattr(sleep_stage, '__int__') else sleep_stage
            })
        
        df = pd.DataFrame(epochs)
        # Ensure integer dtypes
        df['epoch_idx'] = df['epoch_idx'].astype(int)
        df['start_idx'] = df['start_idx'].astype(int)
        df['end_idx'] = df['end_idx'].astype(int)
        df['sleep_stage'] = df['sleep_stage'].astype(int)
        
        return df
    
    def _filter_epochs(self) -> pd.DataFrame:
        """Filter epochs by sleep stage"""
        if self.sleep_stages is None:
            return self.epochs_df.copy()
        
        mask = self.epochs_df['sleep_stage'].isin(self.sleep_stages)
        filtered = self.epochs_df[mask].copy().reset_index(drop=True)
        
        # Reindex filtered epochs
        filtered['filtered_idx'] = range(len(filtered))
        
        return filtered
    
    def get_current_epoch(self) -> pd.DataFrame:
        """Get EEG data for current epoch"""
        if len(self.filtered_epochs_df) == 0:
            return pd.DataFrame()
        
        epoch_info = self.filtered_epochs_df.iloc[self.current_epoch_idx]
        # Ensure indices are integers (pandas iloc requires int)
        start_idx = int(epoch_info['start_idx'])
        end_idx = int(epoch_info['end_idx']) + 1  # +1 for inclusive end
        
        return self.eeg_data.iloc[start_idx:end_idx].copy()
    
    def get_current_epoch_info(self) -> dict:
        """Get information about current epoch"""
        if len(self.filtered_epochs_df) == 0:
            return {}
        
        epoch_info = self.filtered_epochs_df.iloc[self.current_epoch_idx].to_dict()
        epoch_info['current_idx'] = self.current_epoch_idx
        epoch_info['total_filtered'] = len(self.filtered_epochs_df)
        return epoch_info
    
    def next_epoch(self) -> bool:
        """Move to next epoch. Returns True if successful, False if at end"""
        if self.current_epoch_idx < len(self.filtered_epochs_df) - 1:
            self.current_epoch_idx += 1
            return True
        return False
    
    def prev_epoch(self) -> bool:
        """Move to previous epoch. Returns True if successful, False if at start"""
        if self.current_epoch_idx > 0:
            self.current_epoch_idx -= 1
            return True
        return False
    
    def go_to_epoch(self, filtered_idx: int) -> bool:
        """Go to specific epoch by filtered index. Returns True if successful"""
        if 0 <= filtered_idx < len(self.filtered_epochs_df):
            self.current_epoch_idx = filtered_idx
            return True
        return False
    
    def get_epoch_count(self) -> int:
        """Get total number of filtered epochs"""
        return len(self.filtered_epochs_df)
    
    def get_epoch_indices_for_sw(self, start_idx: int, stop_idx: int) -> List[int]:
        """
        Get list of epoch indices (in filtered list) that contain the SW event.
        
        Args:
            start_idx: Start sample index of SW event
            stop_idx: Stop sample index of SW event
            
        Returns:
            List of filtered epoch indices
        """
        matching_epochs = []
        for idx, row in self.filtered_epochs_df.iterrows():
            epoch_start = row['start_idx']
            epoch_end = row['end_idx']
            
            # Check if SW event overlaps with this epoch
            if not (stop_idx < epoch_start or start_idx > epoch_end):
                matching_epochs.append(idx)
        
        return matching_epochs

