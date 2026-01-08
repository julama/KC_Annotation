"""
Intermediate data processing functions
Handles storage and conversion of processed EEG data using per-partition lazy loading
Following the raw pipeline pattern: single functions that return final processed data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable


def _convert_for_storage(data):
    """Convert data to appropriate format for storage"""
    if isinstance(data, np.ndarray):
        if np.iscomplexobj(data):
            # Handle complex arrays - convert to DataFrame with real/imag columns
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            df_data = {}
            if len(data.shape) == 1:
                df_data['real'] = data.real.astype(np.float64)
                df_data['imag'] = data.imag.astype(np.float64)
            else:
                for ch in range(n_channels):
                    df_data[f'real_ch_{ch}'] = data[:, ch].real.astype(np.float64)
                    df_data[f'imag_ch_{ch}'] = data[:, ch].imag.astype(np.float64)
            return pd.DataFrame(df_data)
        else:
            # Handle real arrays - ensure compatible dtypes
            data_converted = data.astype(np.float64)
            if len(data.shape) == 1:
                return pd.DataFrame({'values': data_converted})
            else:
                return pd.DataFrame(data_converted, columns=[f'ch_{i}' for i in range(data.shape[1])])
    elif isinstance(data, pd.DataFrame):
        # Ensure DataFrame has compatible dtypes
        df_converted = data.copy()
        for col in df_converted.columns:
            if df_converted[col].dtype == np.complex128 or df_converted[col].dtype == np.complex64:
                # Split complex columns - handle complex data properly
                complex_values = df_converted[col].values  # Get underlying numpy array
                df_converted[f'{col}_real'] = complex_values.real.astype(np.float64)
                df_converted[f'{col}_imag'] = complex_values.imag.astype(np.float64)
                df_converted = df_converted.drop(columns=[col])
            else:
                # Convert to float64 for compatibility
                try:
                    df_converted[col] = df_converted[col].astype(np.float64)
                except (ValueError, TypeError):
                    # If conversion fails, leave as is
                    pass
        return df_converted
    else:
        # Handle other data types
        return pd.DataFrame(data)


def _create_partition_storage_loader(partition_id: str, preprocessed_loader: Callable):
    """Creates a lazy storage loader function for a single partition with already preprocessed data"""
    def store_partition_processed_eeg():
        """Inner function that stores one partition's preprocessed data"""
        print(f"Storing preprocessed EEG data for {partition_id} (lazy execution)...")
        
        # Data is already preprocessed by the HDF5Dataset (via catalog load_args)
        processed_data = preprocessed_loader()
        
        # Convert to storage format (handle complex arrays, etc.)
        result = _convert_for_storage(processed_data)
        
        return result
    
    return store_partition_processed_eeg

def create_storage_loaders(preprocessed_eeg_data: Dict) -> Dict:
    """
    Creates individual lazy storage loader functions for each partition
    Use the specific *_storage_loaders functions instead for direct output
    """
    loaders = {}
    for partition_id, preprocessed_loader in preprocessed_eeg_data.items():
        loaders[partition_id] = _create_partition_storage_loader(partition_id, preprocessed_loader)
    return loaders


