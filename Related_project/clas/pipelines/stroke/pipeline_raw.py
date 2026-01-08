from kedro.pipeline import node, pipeline
from ..nodes import d01_parse_mat, d01_parse_brainamp
from typing import Dict, Any

def create_pipeline(**kwargs):
    """Raw data pipeline for parsing BrainVision files and creating basic datasets for Stroke study"""

    # === 01_raw_data Generation - Data from parsed BrainVision files ======================
    my_pipeline = pipeline(
        [
            node(
                func=d01_parse_brainamp.eeg_loaders_brainamp,
                inputs="raw.stroke.brainamp",
                outputs="id.stroke.brainamp",
                tags=["brainvision.all","brainvision.eeg.brainamp"],
            ),
            node(
                func=d01_parse_brainamp.channel_metadata_loaders_brainamp,
                inputs="raw.stroke.brainamp",
                outputs="id.stroke.channels",
                tags=["brainvision.all","brainvision.channels.brainamp"],
            ),
            node(
                func=d01_parse_mat.axo_stim_loaders,
                inputs="raw.stroke.axo",
                outputs="id.stroke.axo",
                tags=["mat.all","mat.axo"],
            ),
        ],
    )

    return my_pipeline
