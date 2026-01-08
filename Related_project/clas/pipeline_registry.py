"""Project pipelines."""
from clas.pipelines.episl import (
    pipeline_raw as episl_pipeline_raw,
    pipeline_intermediate as episl_pipeline_intermediate,
    pipeline_primary as episl_pipeline_primary,
    pipeline_events as episl_pipeline_events,
    pipeline_features as episl_pipeline_features,
)
from clas.pipelines.stroke import (
    pipeline_raw as stroke_pipeline_raw,
    pipeline_intermediate as stroke_pipeline_intermediate,
    pipeline_primary as stroke_pipeline_primary,
    pipeline_events as stroke_pipeline_events,
    pipeline_features as stroke_pipeline_features,
)
from kedro.pipeline import Pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        # Default pipeline (EPISL raw)
        "__default__": episl_pipeline_raw.create_pipeline(),
        
        # EPISL study pipelines
        "episl_raw": episl_pipeline_raw.create_pipeline(),
        "episl_intermediate": episl_pipeline_intermediate.create_pipeline(),
        "episl_primary": episl_pipeline_primary.create_pipeline(),
        "episl_events": episl_pipeline_events.create_pipeline(),
        "episl_features": episl_pipeline_features.create_pipeline(),
        
        # Stroke study pipelines
        "stroke_raw": stroke_pipeline_raw.create_pipeline(),
        "stroke_intermediate": stroke_pipeline_intermediate.create_pipeline(),
        "stroke_primary": stroke_pipeline_primary.create_pipeline(),
        "stroke_events": stroke_pipeline_events.create_pipeline(),
        "stroke_features": stroke_pipeline_features.create_pipeline(),
        
        # Legacy pipeline names for backward compatibility
        "raw": episl_pipeline_raw.create_pipeline(),
        "intermediate": episl_pipeline_intermediate.create_pipeline(),
        "primary": episl_pipeline_primary.create_pipeline(),
        "events": episl_pipeline_events.create_pipeline(),
        "features": episl_pipeline_features.create_pipeline(),
    }