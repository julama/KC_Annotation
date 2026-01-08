"""
Intermediate pipeline for optional EEG time series storage for EPISL study
Processes raw EEG with specified parameters and stores the results to disk using per-partition lazy loading
Uses catalog-defined load_args for preprocessing configuration
"""

from kedro.pipeline import Pipeline, node
from typing import Dict, Any

from ..nodes.d02_intermediate import create_storage_loaders


def create_pipeline(**kwargs) -> Pipeline:
    """Create optional intermediate pipeline for storing processed EEG to disk for EPISL study"""

    return Pipeline(
        [
            # === OPTIONAL: STORE PREPROCESSED EEG TIME SERIES TO DISK (PER-PARTITION LAZY LOADING) ===
            # Delta band (referenced + bandpass) storage
            node( 
                func=create_storage_loaders,
                inputs=["id.episl.eeg.intermediate.delta_preprocessed"],
                outputs="id.episl.eeg.ref_48_55.bandpass.delta",
                name="create_delta_storage_loaders",
                tags=["preprocessing", "bandpass", "reference", "optional", "loader"],
            ),
            
            # Sigma band (average reference + bandpass) storage
            node(
                func=create_storage_loaders,
                inputs=["id.episl.eeg.intermediate.sigma_preprocessed"],
                outputs="id.episl.eeg.ref_av.bandpass.sigma",
                name="create_sigma_storage_loaders",
                tags=["preprocessing", "bandpass", "reference", "optional", "loader"],
            ),
            
            # Wavelet transform storage
            node(
                func=create_storage_loaders,
                inputs=["id.episl.eeg.intermediate.wavelet_preprocessed"],
                outputs="id.episl.eeg.ref_57_100.bandpass.wl_1_2",
                name="create_wavelet_storage_loaders",
                tags=["preprocessing", "wavelet", "reference", "optional", "loader"],
            ),
            
            # Hilbert transform storage
            node(
                func=create_storage_loaders,
                inputs=["id.episl.eeg.intermediate.hilbert_preprocessed"],
                outputs="id.episl.eeg.ref_av.bandpass.hilb_05_4",
                name="create_hilbert_storage_loaders",
                tags=["preprocessing", "hilbert", "reference", "optional", "loader"],
            ),

        ]
    )
