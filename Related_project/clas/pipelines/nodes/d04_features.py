import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, List, Tuple

# Import the storage conversion function
from .d02_intermediate import _convert_for_storage

# ====== Core Feature Extraction ======
def get_aggregation_function(agg_method: str) -> Callable:
    """Get aggregation function by name"""
    aggregation_functions = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "std": np.std,
        "median": np.median,
        "var": np.var,
        "sum": np.sum,
        "count": lambda x: len(x),
        "all": None,  # Special case for returning all aggregations
    }
    return aggregation_functions.get(agg_method, np.mean)


def extract_time_window_features(
    feature_data: np.ndarray, 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int,
    aggregation_method: str = "mean",
    aggregation_func: Callable = None
):
    """
    Extract features from a specific time window relative to stimulation
    
    Args:
        feature_data: Feature time series data
        stim_index: Index of stimulation event
        time_window: Tuple of (start_time, end_time) in seconds relative to stim
        sampling_rate: Sampling rate in Hz
        aggregation_method: Aggregation method name ("mean", "max", "min", "std", "all", etc.)
        aggregation_func: Optional pre-computed aggregation function (for backward compatibility)
    
    Returns:
        Single aggregated value (float) if single aggregation, or
        Dictionary with all aggregation values if aggregation_method == "all"
    """
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(len(feature_data), end_idx)
    
    # Handle edge cases
    if start_idx >= end_idx or start_idx >= len(feature_data):
        if aggregation_method == "all":
            return {agg: np.nan for agg in ["mean", "max", "min", "std", "median", "var", "sum", "count"]}
        return np.nan
    
    window_data = feature_data[start_idx:end_idx]
    if len(window_data) == 0:
        if aggregation_method == "all":
            return {agg: np.nan for agg in ["mean", "max", "min", "std", "median", "var", "sum", "count"]}
        return np.nan
    
    # Calculate features based on aggregation method
    if aggregation_method == "all":
        # Return all aggregations as dictionary
        return {
            "mean": np.mean(window_data),
            "max": np.max(window_data),
            "min": np.min(window_data),
            "std": np.std(window_data),
            "median": np.median(window_data),
            "var": np.var(window_data),
            "sum": np.sum(window_data),
            "count": len(window_data)
        }
    else:
        # Return single aggregation
        if aggregation_func is None:
            aggregation_func = get_aggregation_function(aggregation_method)
        return aggregation_func(window_data)


def extract_stim_features(source_features: Dict, stimulations: Dict, specific_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract stimulation features from precomputed primary features using configurable time windows
    
    Args:
        source_features: Precomputed features from primary pipeline
        stimulations: Stimulation metadata
        specific_params: Feature-specific parameters from catalog
    
    Returns:
        Dictionary with time window features (e.g., pre_stim, post_stim)
    """
    # Get parameters
    time_windows = specific_params['time_windows']
    sampling_rate = specific_params['sampling_rate']
    aggregation_method = specific_params['aggregation']
    aggregation_func = get_aggregation_function(aggregation_method)
    
    result = {}
    
    for partition_id in source_features.keys():
        # Get stimulations for this partition
        stim_data = stimulations.get(partition_id, pd.DataFrame())
        if callable(stim_data):
            stim_data = stim_data()
        
        if stim_data.empty:
            continue
        
        # Load precomputed features for this partition
        feature_data = source_features[partition_id]
        if callable(feature_data):
            feature_data = feature_data() 
        
        # Convert to numpy array for consistent processing
        if hasattr(feature_data, 'values'):
            feature_array = feature_data.values
        else:
            feature_array = np.array(feature_data)
        
        # Ensure 1D array for time series processing
        if feature_array.ndim > 1:
            feature_array = feature_array.flatten()
        
        # Extract features for each time window
        for window_name, time_window in time_windows.items():
            if window_name not in result:
                result[window_name] = {}
            
            window_features = []
            
            for _, stim_row in stim_data.iterrows():
                stim_index = stim_row['index']
                
                # Extract features from this time window (single or all aggregations)
                feature_value = extract_time_window_features(
                    feature_array, 
                    stim_index, 
                    time_window, 
                    sampling_rate,
                    aggregation_method=aggregation_method,
                    aggregation_func=aggregation_func
                )
                
                window_features.append(feature_value)
            
            if window_features:
                if aggregation_method == "all":
                    # Convert list of dicts to DataFrame with columns for each aggregation
                    features_df = pd.DataFrame(window_features)
                    result[window_name][partition_id] = _convert_for_storage(features_df)
                else:
                    # Convert list of values to single-column array
                    result[window_name][partition_id] = _convert_for_storage(
                        np.array(window_features).reshape(-1, 1)
                    )
    
    return result


# ====== Event-Based Feature Extraction ======
def extract_event_features(
    source_features: Dict, 
    stimulations: Dict, 
    events: Dict, 
    specific_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract event-based features from precomputed primary features
    
    Args:
        source_features: Precomputed features from primary pipeline
        stimulations: Stimulation metadata
        events: Event metadata (slow waves, spindles, etc.)
        specific_params: Event feature parameters from catalog
    
    Returns:
        Dictionary with event-based features
    """
    # Get parameters
    time_windows = specific_params['time_windows']
    sampling_rate = specific_params['sampling_rate']
    event_type = specific_params['event_type']  # 'stimulations', 'slow_waves', 'spindles', etc.
    feature_type = specific_params['feature_type']  # 'count', 'time_since_last', 'amplitude', etc.
    
    result = {}
    
    for partition_id in source_features.keys():
        # Get stimulations for this partition
        stim_data = stimulations.get(partition_id, pd.DataFrame())
        if callable(stim_data):
            stim_data = stim_data()
        
        if stim_data.empty:
            continue
        
        # Get events for this partition
        event_data = events.get(partition_id, pd.DataFrame())
        if callable(event_data):
            event_data = event_data()
        
        # Load precomputed features for this partition
        feature_data = source_features[partition_id]
        if callable(feature_data):
            feature_data = feature_data()
        
        # Convert to numpy array for consistent processing
        if hasattr(feature_data, 'values'):
            feature_array = feature_data.values
        else:
            feature_array = np.array(feature_data)
        
        # Ensure 1D array for time series processing
        if feature_array.ndim > 1:
            feature_array = feature_array.flatten()
        
        # Get event indices
        event_indices = event_data['index'].tolist() if 'index' in event_data.columns else []
        stim_indices = stim_data['index'].tolist() if 'index' in stim_data.columns else []
        
        # Extract features for each time window
        for window_name, time_window in time_windows.items():
            if window_name not in result:
                result[window_name] = {}
            
            window_features = []
            
            for _, stim_row in stim_data.iterrows():
                stim_index = stim_row['index']
                
                # Extract event-based features
                if feature_type == 'count':
                    feature_value = _count_events_in_window(
                        event_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'time_since_last':
                    feature_value = _time_since_last_event(
                        event_indices, stim_index, sampling_rate
                    )
                elif feature_type == 'amplitude':
                    feature_value = _extract_event_amplitude(
                        feature_array, event_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'first_event_amplitude':
                    feature_value = _extract_first_event_amplitude(
                        feature_array, event_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'mean_amplitude':
                    feature_value = _extract_mean_event_amplitude(
                        feature_array, event_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'first_sw_amplitude_phase':
                    feature_value = _extract_first_sw_amplitude_phase(
                        feature_array, event_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'stim_count_in_window':
                    feature_value = _count_events_in_window(
                        stim_indices, stim_index, time_window, sampling_rate
                    )
                elif feature_type == 'time_since_last_stim':
                    feature_value = _time_since_last_event(
                        stim_indices, stim_index, sampling_rate
                    )
                elif feature_type == 'time_since_last_sw':
                    feature_value = _time_since_last_event(
                        event_indices, stim_index, sampling_rate
                    )
                elif feature_type == 'time_since_last_spindle':
                    feature_value = _time_since_last_event(
                        event_indices, stim_index, sampling_rate
                    )
                else:
                    feature_value = np.nan
                
                window_features.append(feature_value)
            
            if window_features:
                result[window_name][partition_id] = _convert_for_storage(
                    np.array(window_features).reshape(-1, 1)
                )
    
    return result


# ====== Event Feature Helper Functions ======
def _count_events_in_window(
    event_indices: List[int], 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int
) -> int:
    """Count events in a time window relative to stimulation"""
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    events_in_window = [
        idx for idx in event_indices 
        if start_idx <= idx <= end_idx
    ]
    
    return len(events_in_window)


def _time_since_last_event(
    event_indices: List[int], 
    stim_index: int, 
    sampling_rate: int
) -> float:
    """Time since last event before stimulation"""
    prev_events = [idx for idx in event_indices if idx < stim_index]
    
    if not prev_events:
        return np.nan
    
    last_event = max(prev_events)
    time_diff = (stim_index - last_event) / sampling_rate
    
    return time_diff


def _extract_event_amplitude(
    feature_array: np.ndarray, 
    event_indices: List[int], 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int
) -> float:
    """Extract amplitude of events in time window"""
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    events_in_window = [
        idx for idx in event_indices 
        if start_idx <= idx <= end_idx
    ]
    
    if not events_in_window:
        return np.nan
    
    # Get amplitudes at event indices
    amplitudes = [feature_array[idx] for idx in events_in_window if idx < len(feature_array)]
    
    return np.mean(amplitudes) if amplitudes else np.nan


def _extract_first_event_amplitude(
    feature_array: np.ndarray, 
    event_indices: List[int], 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int
) -> float:
    """Extract amplitude of first event in time window"""
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    events_in_window = [
        idx for idx in event_indices 
        if start_idx <= idx <= end_idx
    ]
    
    if not events_in_window:
        return np.nan
    
    # Get first event (closest to start of window)
    first_event = min(events_in_window)
    
    if first_event < len(feature_array):
        return feature_array[first_event]
    
    return np.nan


def _extract_mean_event_amplitude(
    feature_array: np.ndarray, 
    event_indices: List[int], 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int
) -> float:
    """Extract mean amplitude of events in time window"""
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    events_in_window = [
        idx for idx in event_indices 
        if start_idx <= idx <= end_idx
    ]
    
    if not events_in_window:
        return np.nan
    
    # Get amplitudes at event indices
    amplitudes = [feature_array[idx] for idx in events_in_window if idx < len(feature_array)]
    
    return np.mean(amplitudes) if amplitudes else np.nan


def _extract_first_sw_amplitude_phase(
    feature_array: np.ndarray, 
    event_indices: List[int], 
    stim_index: int, 
    time_window: Tuple[float, float], 
    sampling_rate: int
) -> float:
    """
    Extract amplitude and phase norm from first slow wave event in time window
    
    This function extracts the first slow wave event in the specified time window
    and returns the amplitude and phase norm (magnitude of complex number)
    """
    start_time, end_time = time_window
    start_idx = stim_index + int(start_time * sampling_rate)
    end_idx = stim_index + int(end_time * sampling_rate)
    
    events_in_window = [
        idx for idx in event_indices 
        if start_idx <= idx <= end_idx
    ]
    
    if not events_in_window:
        return np.nan
    
    # Get first event (closest to start of window)
    first_event = min(events_in_window)
    
    if first_event >= len(feature_array):
        return np.nan
    
    # Extract amplitude and phase norm
    # Assuming feature_array contains complex values (amplitude + phase)
    if np.iscomplexobj(feature_array):
        # If complex, return magnitude (amplitude and phase norm)
        return np.abs(feature_array[first_event])
    else:
        # If real, return the value as amplitude
        return feature_array[first_event]


# ====== Helper Functions ======
def split_stim_features(stim_results: Dict[str, Any]) -> tuple:
    """Split stimulation results into pre and post stimulation features"""
    return (
        stim_results.get('pre_stim', {}),
        stim_results.get('post_stim', {})
    )


def split_stim_features_pre_only(stim_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only pre-stimulation features (when post is not defined)"""
    return stim_results.get('pre_stim', {})


def split_stim_features_generic(stim_results: Dict[str, Any], window_names: List[str]) -> tuple:
    """Split stimulation results into multiple time window features"""
    return tuple(stim_results.get(window_name, {}) for window_name in window_names)
