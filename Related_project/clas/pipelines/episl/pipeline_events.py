"""
Events pipeline for sleep feature detection (slow waves, spindles) for EPISL study
Uses processed EEG time series and extracted features
"""

from kedro.pipeline import Pipeline, node
from typing import Dict, Any

from ..nodes.d03_events import sw_detection, spindle_detection, sw_detection_wavelet, sw_detection_hilbert


def create_pipeline(**kwargs) -> Pipeline:
    """Create events pipeline for sleep feature detection for EPISL study"""

    return Pipeline(
        [
            # === SLEEP FEATURE DETECTION ===
            
            # Slow wave detection - using processed bandpass delta data
            node(
                func=sw_detection,
                inputs=["id.episl.eeg.intermediate.delta_preprocessed", "params:episl_events.sleep_features.sw_detection"],
                outputs="pd.episl.sw_detection.channelwise",
                name="sw_detection",
                tags=["sleep_features", "sw.detection", "events"],
            ),
            
            # Spindle detection - using processed bandpass sigma data
            node(
                func=spindle_detection,
                inputs=["id.episl.eeg", "id.episl.eeg.epochs", "params:episl_events.sleep_features.spindle_detection"],
                outputs="pd.episl.spindle_detection.channelwise",
                name="spindle_detection",
                tags=["sleep_features", "spindle.detection", "events"],
            ),

            node(
                func=sw_detection_wavelet,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "params:episl_events.sleep_features.sw_detection_wavelet"],
                outputs="pd.episl.sw_detection.wavelet",
                name="sw_detection_wavelet",
                tags=["sleep_features", "sw.detection.wavelet", "events"],
            ),

            node(
                func=sw_detection_hilbert,
                inputs=["pd.episl.ref_av.hilb_05_4.norm", "params:episl_events.sleep_features.sw_detection_hilbert"],
                outputs="pd.episl.sw_detection.hilbert",
                name="sw_detection_hilbert",
                tags=["sleep_features", "sw.detection.hilbert", "events"],
            ),
        ]
    )
