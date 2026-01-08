"""
Primary pipeline for feature extraction and event processing for EPISL study
Loads processed EEG time series and extracts features
"""

from kedro.pipeline import Pipeline, node
from typing import Dict, Any

# Import functions from the dedicated modules
from ..nodes.d03_events import sw_cut
from ..nodes.d03_primary import (
    create_wavelet_features_loader,
    create_hilbert_features_loader,
    split_wavelet_features_from_loader,
    split_hilbert_features_from_loader
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the primary data pipeline for feature extraction and event processing for EPISL study"""
    
    return Pipeline(
        [
            # === Feature Extraction from Processed EEG Time Series ===
            
            # Create wavelet features loader
            node(
                func=create_wavelet_features_loader,
                inputs=["id.episl.eeg.wavelet_preprocessed"],
                outputs="wavelet_features_loader",
                name="create_wavelet_features_loader",
                tags=["features", "wavelet", "loader"],
            ),
            
            # Split wavelet features
            node(
                func=split_wavelet_features_from_loader,
                inputs="wavelet_features_loader",
                outputs=[
                    "pd.episl.ref_57_100.wl_1_2.norm",
                    "pd.episl.ref_57_100.wl_1_2.raw_mean", 
                    "pd.episl.ref_57_100.wl_1_2.real",
                    "pd.episl.ref_57_100.wl_1_2.cos_sim"
                ],
                name="split_wavelet_features",
                tags=["features", "wavelet", "storage"],
            ),
            
            # Create Hilbert features loader
            node(
                func=create_hilbert_features_loader,
                inputs=["id.episl.eeg.hilbert_preprocessed"],
                outputs="hilbert_features_loader",
                name="create_hilbert_features_loader",
                tags=["features", "hilbert", "loader"],
            ),
            
            # Split Hilbert features
            node(
                func=split_hilbert_features_from_loader,
                inputs="hilbert_features_loader",
                outputs=[
                    "pd.episl.ref_av.hilb_05_4.amplitude",
                    "pd.episl.ref_av.hilb_05_4.phase",
                    "pd.episl.ref_av.hilb_05_4.norm"
                ],
                name="split_hilbert_features",
                tags=["features", "hilbert", "storage"],
            ),
            
            # === EEG Segment Extraction ===
            
            # # SW cut for type 1 (temporarily disabled due to HDF5Dataset._load parameter issue)
            # node(
            #     func=sw_cut,
            #     inputs=["raw.conclas.sw1", "id.episl.datasets", "id.episl.eeg", "params:conclas.sw_cut"],
            #     outputs="pd.conclas.sw1",
            #     name="sw_cut_sw1",
            #     tags=["eeg.cut.sw1", "eeg.cut.sw"],
            # ),
            
            # # SW cut for type 2 (temporarily disabled due to HDF5Dataset._load parameter issue)
            # node(
            #     func=sw_cut,
            #     inputs=["raw.conclas.sw2", "id.episl.datasets", "id.episl.eeg", "params:conclas.sw_cut"],
            #     outputs="pd.conclas.sw2",
            #     name="sw_cut_sw2",
            #     tags=["eeg.cut.sw2", "eeg.cut.sw"],
            # ),
        ]
    )
