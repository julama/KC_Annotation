"""
Intermediate pipeline for optional EEG time series storage for Stroke study
Processes raw EEG with specified parameters and stores the results to disk using per-partition lazy loading
Uses catalog-defined load_args for preprocessing configuration
"""

from kedro.pipeline import Pipeline, node
from typing import Dict, Any

from ..nodes.d02_intermediate import create_storage_loaders


def create_pipeline(**kwargs) -> Pipeline:
    """Create optional intermediate pipeline for storing processed EEG to disk for Stroke study"""

    return Pipeline(
        [
            # === BRAINAMP PREPROCESSING ===
            # BrainAmp delta preprocessing (re-referencing + bandpass) using HDF5Dataset with preprocessing
            node(
                func=create_storage_loaders,
                inputs=["id.stroke.brainamp.intermediate.delta_preprocessed"],
                outputs="id.stroke.brainamp.delta_preprocessed",
                name="create_brainamp_delta_storage_loaders",
                tags=["preprocessing", "brainamp", "delta", "rereferencing", "optional", "loader"],
            ),
            
            # BrainAmp broadband preprocessing (ref_29_47 + bandpass 0.5-35 Hz) storage
            node(
                func=create_storage_loaders,
                inputs=["id.stroke.brainamp.intermediate.broadband_preprocessed"],
                outputs="id.stroke.brainamp.broadband_preprocessed",
                name="create_brainamp_broadband_storage_loaders",
                tags=["preprocessing", "brainamp", "broadband", "rereferencing", "optional", "loader"],
            ),
        ]
    )
