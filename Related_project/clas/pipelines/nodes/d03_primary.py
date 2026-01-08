import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the storage conversion function
from .d02_intermediate import _convert_for_storage




def _get_filepath_from_loader(partition_id: str, eeg_loader) -> str:
    """Extract filepath from partition loader"""
    if callable(eeg_loader):
        # For PartitionedDataset, construct filepath from partition ID and base path
        return f"data/01_raw/episl/eeg/{partition_id}.h5"
    else:
        # Fallback for other dataset types
        return getattr(eeg_loader, '_filepath', str(eeg_loader))


def extract_wavelet_features(raw_eeg_data: Dict, specific_params: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Extract all wavelet features from raw EEG with on-the-fly preprocessing
    Returns a dictionary with separate entries for each feature type
    """
    from clas.datasets.hdf5_dataset import HDF5Dataset
    import numpy as np
    from clas.helpers.mymath import norm, complex_cosine_similarity
    
    # The specific_params should already contain all needed parameters
    # Kedro's parameter resolution should have merged the hierarchy
    preprocessing_params = specific_params
    
    result = {
        'norm': {},
        'raw_mean': {},
        'real': {},
        'cos_sim': {}
    }
    
    for partition_id, eeg_loader in raw_eeg_data.items():
        # Get filepath from loader
        filepath = _get_filepath_from_loader(partition_id, eeg_loader)
        
        # Create HDF5Dataset with preprocessing (including wavelet)
        dataset = HDF5Dataset(filepath=filepath, **preprocessing_params)
        wavelet_data = dataset._load()
        
        # Create raw_params for raw_mean (no wavelet transform)
        raw_params = preprocessing_params.copy()
        if 'freq' in raw_params:
            del raw_params['freq']
        if 'cycles' in raw_params:
            del raw_params['cycles']
        
        # Load raw EEG data for raw_mean calculation
        raw_dataset = HDF5Dataset(filepath=filepath, **raw_params)
        raw_eeg_data = raw_dataset._load()
        
        # Compute all features
        if isinstance(wavelet_data, np.ndarray):
            # Wavelet norm
            norm_data = norm(wavelet_data, axis=1, keepdims=True)
            
            # Wavelet real part mean
            real_data = wavelet_data.real
            mean_real_data = np.mean(real_data, axis=1, keepdims=True)
            
            # Raw mean (from raw EEG, not wavelet)
            if isinstance(raw_eeg_data, np.ndarray):
                mean_raw_data = np.mean(raw_eeg_data, axis=1, keepdims=True)
            else:
                mean_raw_data = np.mean(raw_eeg_data.values, axis=1, keepdims=True)
            
            # Cosine similarity
            cos_sim_data = []
            for i in range(1, wavelet_data.shape[0]):
                sim = complex_cosine_similarity(wavelet_data[i-1], wavelet_data[i], axis=0)
                cos_sim_data.append(float(sim) if np.isscalar(sim) else float(np.mean(sim)))
            cos_sim_array = np.array(cos_sim_data, dtype=np.float64).reshape(-1, 1)
            
        else:
            # Handle DataFrame case
            norm_data = norm(wavelet_data.values, axis=1, keepdims=True)
            real_data = wavelet_data.real if hasattr(wavelet_data, 'real') else wavelet_data
            mean_real_data = np.mean(real_data.values, axis=1, keepdims=True)
            
            if isinstance(raw_eeg_data, np.ndarray):
                mean_raw_data = np.mean(raw_eeg_data, axis=1, keepdims=True)
            else:
                mean_raw_data = np.mean(raw_eeg_data.values, axis=1, keepdims=True)
            
            # Cosine similarity
            data_array = wavelet_data.values
            cos_sim_data = []
            for i in range(1, data_array.shape[0]):
                sim = complex_cosine_similarity(data_array[i-1], data_array[i], axis=0)
                cos_sim_data.append(float(sim) if np.isscalar(sim) else float(np.mean(sim)))
            cos_sim_array = np.array(cos_sim_data, dtype=np.float64).reshape(-1, 1)
        
        # Store results
        result['norm'][partition_id] = _convert_for_storage(norm_data)
        result['raw_mean'][partition_id] = _convert_for_storage(mean_raw_data)
        result['real'][partition_id] = _convert_for_storage(mean_real_data)
        result['cos_sim'][partition_id] = _convert_for_storage(cos_sim_array)
    
    return result


def extract_hilbert_features(raw_eeg_data: Dict, specific_params: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Extract all Hilbert features from raw EEG with on-the-fly preprocessing
    Returns a dictionary with separate entries for each feature type
    """
    from clas.datasets.hdf5_dataset import HDF5Dataset
    import numpy as np
    from clas.helpers.mymath import norm
    
    # The specific_params should already contain all needed parameters
    # Kedro's parameter resolution should have merged the hierarchy
    preprocessing_params = specific_params
    
    result = {
        'amplitude': {},
        'phase': {},
        'norm': {}
    }
    
    for partition_id, eeg_loader in raw_eeg_data.items():
        # Get filepath from loader
        filepath = _get_filepath_from_loader(partition_id, eeg_loader)
        
        # Create HDF5Dataset with preprocessing (including Hilbert transform)
        dataset = HDF5Dataset(filepath=filepath, **preprocessing_params)
        hilbert_data = dataset._load()
        
        # Compute all features
        if isinstance(hilbert_data, np.ndarray) and np.iscomplexobj(hilbert_data):
            # Extract amplitude (magnitude) from complex Hilbert coefficients
            amplitude_data = np.abs(hilbert_data)
            mean_amplitude = np.mean(amplitude_data, axis=1, keepdims=True)
            
            # Extract instantaneous phase from complex Hilbert coefficients
            phase_data = np.angle(hilbert_data)
            mean_phase = np.mean(phase_data, axis=1, keepdims=True)
            
            # Compute norm across channels using mm.norm function
            norm_data = norm(hilbert_data, axis=1, keepdims=True)
            
        elif isinstance(hilbert_data, pd.DataFrame):
            # Handle DataFrame case
            amplitude_data = hilbert_data.abs() if hasattr(hilbert_data, 'abs') else hilbert_data
            mean_amplitude = np.mean(amplitude_data.values, axis=1, keepdims=True)
            
            mean_phase = np.mean(hilbert_data.values, axis=1, keepdims=True)
            norm_data = norm(hilbert_data.values, axis=1, keepdims=True)
            
        else:
            # If data is already real, compute mean across channels
            if isinstance(hilbert_data, np.ndarray):
                mean_amplitude = np.mean(hilbert_data, axis=1, keepdims=True)
                mean_phase = np.mean(hilbert_data, axis=1, keepdims=True)
                norm_data = norm(hilbert_data, axis=1, keepdims=True)
            else:
                mean_amplitude = np.mean(hilbert_data.values, axis=1, keepdims=True)
                mean_phase = np.mean(hilbert_data.values, axis=1, keepdims=True)
                norm_data = norm(hilbert_data.values, axis=1, keepdims=True)
        
        # Store results
        result['amplitude'][partition_id] = _convert_for_storage(mean_amplitude)
        result['phase'][partition_id] = _convert_for_storage(mean_phase)
        result['norm'][partition_id] = _convert_for_storage(norm_data)
    
    return result


# Helper functions to split consolidated results for individual storage
def split_wavelet_features(wavelet_results: Dict[str, Dict]) -> tuple:
    """Split consolidated wavelet results into individual feature dictionaries"""
    return (
        wavelet_results['norm'],
        wavelet_results['raw_mean'], 
        wavelet_results['real'],
        wavelet_results['cos_sim']
    )


def split_hilbert_features(hilbert_results: Dict[str, Dict]) -> tuple:
    """Split consolidated Hilbert results into individual feature dictionaries"""
    return (
        hilbert_results['amplitude'],
        hilbert_results['phase'],
        hilbert_results['norm']
    )


# ====== Loader Functions for Preprocessed Data ======

def create_wavelet_features_loader(wavelet_preprocessed_data: Dict) -> callable:
    """
    Create a loader function for wavelet features from preprocessed data
    Returns a function that can be called to extract all wavelet features
    """
    def wavelet_features_loader():
        """Load and extract all wavelet features from preprocessed data"""
        return extract_wavelet_features(wavelet_preprocessed_data, {})
    
    return wavelet_features_loader


def create_hilbert_features_loader(hilbert_preprocessed_data: Dict) -> callable:
    """
    Create a loader function for Hilbert features from preprocessed data
    Returns a function that can be called to extract all Hilbert features
    """
    def hilbert_features_loader():
        """Load and extract all Hilbert features from preprocessed data"""
        return extract_hilbert_features(hilbert_preprocessed_data, {})
    
    return hilbert_features_loader


def split_wavelet_features_from_loader(wavelet_features_loader: callable) -> tuple:
    """Split wavelet features from loader function"""
    wavelet_results = wavelet_features_loader()
    return split_wavelet_features(wavelet_results)


def split_hilbert_features_from_loader(hilbert_features_loader: callable) -> tuple:
    """Split Hilbert features from loader function"""
    hilbert_results = hilbert_features_loader()
    return split_hilbert_features(hilbert_results)
