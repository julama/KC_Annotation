"""Load Slow Wave events from CSV and filter by current epoch"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path


def load_sw_events(csv_path: str) -> pd.DataFrame:
    """
    Load Slow Wave events from CSV file.
    
    Expected CSV format:
        start_idx, stop_idx
    
    Args:
        csv_path: Path to CSV file containing SW events
        
    Returns:
        DataFrame with columns: start_idx, stop_idx
    """
    if not Path(csv_path).exists():
        print(f"Warning: SW events CSV not found at {csv_path}")
        return pd.DataFrame(columns=['start_idx', 'stop_idx'])
    
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_cols = ['start_idx', 'stop_idx']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Convert to integers
        df['start_idx'] = df['start_idx'].astype(int)
        df['stop_idx'] = df['stop_idx'].astype(int)
        
        # Add event ID for tracking (only if DataFrame is not empty)
        if len(df) > 0:
            df['event_id'] = range(len(df))
        else:
            df['event_id'] = []
        
        print(f"Loaded {len(df)} SW events from {csv_path}")
        return df
        
    except Exception as e:
        print(f"Error loading SW events CSV: {e}")
        return pd.DataFrame(columns=['start_idx', 'stop_idx', 'event_id'])


def filter_sw_events_for_epoch(sw_events: pd.DataFrame, epoch_start_idx: int, 
                                epoch_end_idx: int) -> pd.DataFrame:
    """
    Filter SW events that fall within the current epoch.
    
    Args:
        sw_events: DataFrame with SW events (columns: start_idx, stop_idx, event_id)
        epoch_start_idx: Start sample index of current epoch
        epoch_end_idx: End sample index of current epoch (inclusive)
        
    Returns:
        Filtered DataFrame with SW events in current epoch
    """
    if sw_events.empty:
        return sw_events.copy()
    
    # Ensure epoch indices are integers
    epoch_start_idx = int(epoch_start_idx)
    epoch_end_idx = int(epoch_end_idx)
    
    print(f"[DEBUG SW] Filtering {len(sw_events)} events for epoch [{epoch_start_idx}, {epoch_end_idx}]")
    
    # Find events that overlap with epoch
    # Event overlaps if: not (event_end < epoch_start or event_start > epoch_end)
    # Use vectorized operations for better performance
    mask = ~((sw_events['stop_idx'] < epoch_start_idx) | 
             (sw_events['start_idx'] > epoch_end_idx))
    
    filtered = sw_events[mask].copy()
    print(f"[DEBUG SW] Found {len(filtered)} overlapping events")
    
    # Clip event indices to epoch boundaries for display
    if not filtered.empty:
        filtered = filtered.copy()
        filtered['display_start'] = np.maximum(filtered['start_idx'], epoch_start_idx).astype(int)
        filtered['display_stop'] = np.minimum(filtered['stop_idx'], epoch_end_idx).astype(int)
        # Convert to relative indices within epoch
        filtered['relative_start'] = (filtered['display_start'] - epoch_start_idx).astype(int)
        filtered['relative_stop'] = (filtered['display_stop'] - epoch_start_idx).astype(int)
    
    return filtered


def get_sw_events_for_current_epoch(sw_events: pd.DataFrame, epoch_manager) -> pd.DataFrame:
    """
    Get SW events for the current epoch from epoch manager.
    
    Args:
        sw_events: DataFrame with all SW events
        epoch_manager: EpochManager instance
        
    Returns:
        Filtered DataFrame with SW events in current epoch
    """
    epoch_info = epoch_manager.get_current_epoch_info()
    if not epoch_info:
        return pd.DataFrame(columns=['start_idx', 'stop_idx', 'event_id'])
    
    # Ensure indices are integers
    epoch_start = int(epoch_info['start_idx'])
    epoch_end = int(epoch_info['end_idx'])
    
    return filter_sw_events_for_epoch(sw_events, epoch_start, epoch_end)

