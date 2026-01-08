from kedro.pipeline import node, pipeline
from ..nodes import d01_parse_mat, d01_parse_brainamp
from typing import Dict, Any

def create_pipeline(**kwargs):
    """Raw data pipeline for parsing .mat files and creating basic datasets for EPISL study"""

    # === 01_raw_data Generation - Data from parsed .mat files ======================
    my_pipeline = pipeline(
        [
            node(
                func=d01_parse_mat.eeg_loaders,
                inputs="raw.episl.mat", 
                outputs="id.episl.eeg",
                tags=["mat.all","mat.eeg"],
            ),
            node(
                func=d01_parse_mat.epochs_loaders,
                inputs="raw.episl.mat",
                outputs="id.episl.eeg.epochs",
                tags=["mat.all", "mat.epochs"],
            ),
            node(
                func=d01_parse_mat.dataset_info,
                inputs="raw.episl.mat",
                outputs="id.episl.datasets", 
                tags=["mat.all", "dataset.infos"],
            ),
            node(
                func=d01_parse_mat.section_info_loaders,
                inputs="raw.episl.mat",
                outputs="id.episl.sections",
                tags=["mat.all", "mat.sections"],
            ),
            node(
                func=d01_parse_mat.stim_info_loaders,
                inputs="raw.episl.mat",
                outputs="id.episl.stimulations",
                tags=["mat.all","mat.stim"],
            ),
            node(
                func=d01_parse_mat.channel_info_loaders,
                inputs="raw.episl.mat",
                outputs="id.episl.channels",
                tags=["mat.all", "mat.channels"],
            ),
        ],
    )

    return my_pipeline
