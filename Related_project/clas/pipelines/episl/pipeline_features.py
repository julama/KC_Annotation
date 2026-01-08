"""
Features pipeline for pre/post stimulation feature extraction for EPISL study
Computes features around stimulation events using configurable parameters
"""

from kedro.pipeline import Pipeline, node
from typing import Dict, Any

# Import functions from the dedicated modules
from ..nodes.d04_features import (
    extract_stim_features,
    split_stim_features,
    split_stim_features_pre_only,
    extract_event_features
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the features pipeline for pre/post stimulation feature extraction for EPISL study"""
    
    return Pipeline(
        [
            # === Wavelet Norm Features ===
            
            # Extract wavelet norm features (pre and post stimulation)
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.norm", "id.episl.stimulations", "params:episl_features.wavelet_norm"],
                outputs="wavelet_norm_stim_features",
                name="extract_wavelet_norm_stim_features",
                tags=["features", "wavelet_norm", "computation"],
            ),
            
            # Split wavelet norm features into pre and post
            node(
                func=split_stim_features,
                inputs="wavelet_norm_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.av_norm.pre",
                    "fe.episl.ref_57_100.wl_1_2.av_norm.post"
                ],
                name="split_wavelet_norm_features",
                tags=["features", "wavelet_norm", "storage"],
            ),

            # === Time Series Features with All Aggregations ===
            
            # --- Norm Features: [-1.5, 0.3] pre / [0.3, 1.5] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.norm", "id.episl.stimulations", "params:episl_features.ts_features.all_1.5_0.3"],
                outputs="ts_features_norm_1_5_0_3_stim_features",
                name="extract_ts_features_norm_1_5_0_3",
                tags=["features", "ts_features", "norm", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_norm_1_5_0_3_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.pre_1_5_0_3",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.post_1_5_0_3"
                ],
                name="split_ts_features_norm_1_5_0_3",
                tags=["features", "ts_features", "norm", "storage"],
            ),
            
            # --- Norm Features: [-5.0, 0.0] pre / [0.0, 2.0] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.norm", "id.episl.stimulations", "params:episl_features.ts_features.all_5.0_0.0"],
                outputs="ts_features_norm_5_0_0_0_stim_features",
                name="extract_ts_features_norm_5_0_0_0",
                tags=["features", "ts_features", "norm", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_norm_5_0_0_0_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.pre_5_0_0_0",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.post_5_0_0_0"
                ],
                name="split_ts_features_norm_5_0_0_0",
                tags=["features", "ts_features", "norm", "storage"],
            ),
            
            # --- Norm Features: [-2.0, 0.0] pre / [0.0, 2.0] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.norm", "id.episl.stimulations", "params:episl_features.ts_features.all_2.0_0.0"],
                outputs="ts_features_norm_2_0_0_0_stim_features",
                name="extract_ts_features_norm_2_0_0_0",
                tags=["features", "ts_features", "norm", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_norm_2_0_0_0_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.pre_2_0_0_0",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.norm.post_2_0_0_0"
                ],
                name="split_ts_features_norm_2_0_0_0",
                tags=["features", "ts_features", "norm", "storage"],
            ),
            
            # --- Norm Features: [-0.5, 0.5] pre only ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.norm", "id.episl.stimulations", "params:episl_features.ts_features.all_0.5_0.5"],
                outputs="ts_features_norm_0_5_0_5_stim_features",
                name="extract_ts_features_norm_0_5_0_5",
                tags=["features", "ts_features", "norm", "computation"],
            ),
            node(
                func=split_stim_features_pre_only,
                inputs="ts_features_norm_0_5_0_5_stim_features",
                outputs="fe.episl.ref_57_100.wl_1_2.ts_features.norm.pre_0_5_0_5",
                name="split_ts_features_norm_0_5_0_5",
                tags=["features", "ts_features", "norm", "storage"],
            ),
            
            # --- Real Features: [-1.5, 0.3] pre / [0.3, 1.5] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "id.episl.stimulations", "params:episl_features.ts_features.all_1.5_0.3"],
                outputs="ts_features_real_1_5_0_3_stim_features",
                name="extract_ts_features_real_1_5_0_3",
                tags=["features", "ts_features", "real", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_real_1_5_0_3_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.pre_1_5_0_3",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.post_1_5_0_3"
                ],
                name="split_ts_features_real_1_5_0_3",
                tags=["features", "ts_features", "real", "storage"],
            ),
            
            # --- Real Features: [-5.0, 0.0] pre / [0.0, 2.0] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "id.episl.stimulations", "params:episl_features.ts_features.all_5.0_0.0"],
                outputs="ts_features_real_5_0_0_0_stim_features",
                name="extract_ts_features_real_5_0_0_0",
                tags=["features", "ts_features", "real", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_real_5_0_0_0_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.pre_5_0_0_0",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.post_5_0_0_0"
                ],
                name="split_ts_features_real_5_0_0_0",
                tags=["features", "ts_features", "real", "storage"],
            ),
            
            # --- Real Features: [-2.0, 0.0] pre / [0.0, 2.0] post ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "id.episl.stimulations", "params:episl_features.ts_features.all_2.0_0.0"],
                outputs="ts_features_real_2_0_0_0_stim_features",
                name="extract_ts_features_real_2_0_0_0",
                tags=["features", "ts_features", "real", "computation"],
            ),
            node(
                func=split_stim_features,
                inputs="ts_features_real_2_0_0_0_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.pre_2_0_0_0",
                    "fe.episl.ref_57_100.wl_1_2.ts_features.real.post_2_0_0_0"
                ],
                name="split_ts_features_real_2_0_0_0",
                tags=["features", "ts_features", "real", "storage"],
            ),
            
            # --- Real Features: [-0.5, 0.5] pre only ---
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "id.episl.stimulations", "params:episl_features.ts_features.all_0.5_0.5"],
                outputs="ts_features_real_0_5_0_5_stim_features",
                name="extract_ts_features_real_0_5_0_5",
                tags=["features", "ts_features", "real", "computation"],
            ),
            node(
                func=split_stim_features_pre_only,
                inputs="ts_features_real_0_5_0_5_stim_features",
                outputs="fe.episl.ref_57_100.wl_1_2.ts_features.real.pre_0_5_0_5",
                name="split_ts_features_real_0_5_0_5",
                tags=["features", "ts_features", "real", "storage"],
            ),
            
            # === Hilbert Amplitude Features ===
            
            # Extract Hilbert amplitude features (pre and post stimulation)
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_av.hilb_05_4.amplitude", "id.episl.stimulations", "params:episl_features.hilbert_amplitude"],
                outputs="hilbert_amplitude_stim_features",
                name="extract_hilbert_amplitude_stim_features",
                tags=["features", "hilbert_amplitude", "computation"],
            ),
            
            # Split Hilbert amplitude features into pre and post
            node(
                func=split_stim_features,
                inputs="hilbert_amplitude_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.amplitude.pre",
                    "fe.episl.ref_57_100.wl_1_2.amplitude.post"
                ],
                name="split_hilbert_amplitude_features",
                tags=["features", "hilbert_amplitude", "storage"],
            ),
            
            # === Wavelet Real Features ===
            
            # Extract wavelet real features (pre and post stimulation)
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.real", "id.episl.stimulations", "params:episl_features.wavelet_real"],
                outputs="wavelet_real_stim_features",
                name="extract_wavelet_real_stim_features",
                tags=["features", "wavelet_real", "computation"],
            ),
            
            # Split wavelet real features into pre and post
            node(
                func=split_stim_features,
                inputs="wavelet_real_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.real.pre",
                    "fe.episl.ref_57_100.wl_1_2.real.post"
                ],
                name="split_wavelet_real_features",
                tags=["features", "wavelet_real", "storage"],
            ),
            
            # === Wavelet Raw Mean Features ===
            
            # Extract wavelet raw mean features (pre and post stimulation)
            node(
                func=extract_stim_features,
                inputs=["pd.episl.ref_57_100.wl_1_2.raw_mean", "id.episl.stimulations", "params:episl_features.wavelet_raw_mean"],
                outputs="wavelet_raw_mean_stim_features",
                name="extract_wavelet_raw_mean_stim_features",
                tags=["features", "wavelet_raw_mean", "computation"],
            ),
            
            # Split wavelet raw mean features into pre and post
            node(
                func=split_stim_features,
                inputs="wavelet_raw_mean_stim_features",
                outputs=[
                    "fe.episl.ref_57_100.wl_1_2.raw_mean.pre",
                    "fe.episl.ref_57_100.wl_1_2.raw_mean.post"
                ],
                name="split_wavelet_raw_mean_features",
                tags=["features", "wavelet_raw_mean", "storage"],
            ),
        ]
    )
