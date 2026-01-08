# Pipeline Structure

This directory contains the pipeline definitions organized by shared nodes and study-specific pipelines.

## Directory Structure

```
clas/pipelines/
├── nodes/                        # Shared processing nodes
│   ├── __init__.py
│   ├── d01_parse_mat.py          # MATLAB file parsing
│   ├── d01_parse_brainamp.py     # BrainVision file parsing
│   ├── d02_intermediate.py       # Intermediate data processing
│   ├── d03_events.py             # Sleep feature detection
│   └── d04_features.py           # Feature extraction
├── episl/                        # EPISL study-specific pipelines
│   ├── __init__.py
│   ├── pipeline_raw.py
│   ├── pipeline_intermediate.py
│   ├── pipeline_primary.py
│   ├── pipeline_events.py
│   └── pipeline_features.py
├── stroke/                       # Stroke study-specific pipelines
│   ├── __init__.py
│   ├── pipeline_raw.py
│   ├── pipeline_intermediate.py
│   ├── pipeline_primary.py
│   ├── pipeline_events.py
│   └── pipeline_features.py
└── README.md
```

## Design Principles

### Shared Nodes (`nodes/`)
- **Reusable**: Can be used across different studies
- **Generic**: Not tied to specific study data or parameters
- **Focused**: Each node has a single responsibility

### Study-Specific Pipelines (`episl/`, `stroke/`)
- **Orchestration**: Combine shared nodes with study-specific configurations
- **Study-specific**: Tailored to the data and requirements of each study
- **Independent**: Changes to one study don't affect others

## Usage Examples

### Running Complete Pipelines

```bash
# EPISL study pipelines
kedro run --pipeline episl_raw
kedro run --pipeline episl_intermediate
kedro run --pipeline episl_primary
kedro run --pipeline episl_events
kedro run --pipeline episl_features

# Stroke study pipelines
kedro run --pipeline stroke_raw
kedro run --pipeline stroke_intermediate
kedro run --pipeline stroke_primary
kedro run --pipeline stroke_events
kedro run --pipeline stroke_features
```

### Running Specific Tasks with Tags

#### EPISL Study - MATLAB Data Processing
```bash
# Read out dataset infos from .mat files and store in CSV format
kedro run --pipeline episl_raw --tags mat.dataset

# Read out section infos from .mat files and store in CSV format
kedro run --pipeline episl_raw --tags mat.section

# Read out stimulation infos from .mat files and store in CSV format
kedro run --pipeline episl_raw --tags mat.stim

# Read out channel infos from .mat files and store in CSV format
kedro run --pipeline episl_raw --tags mat.channel

# Read out EEG epochs from .mat files and store in CSV format
kedro run --pipeline episl_raw --tags mat.epochs

# Process all MATLAB data
kedro run --pipeline episl_raw --tags mat.all
```

#### Stroke Study - BrainVision Data Processing
```bash
# Process BrainVision EEG data
kedro run --pipeline stroke_raw --tags brainvision.eeg.brainamp

# Process BrainVision channel metadata
kedro run --pipeline stroke_raw --tags brainvision.channels.brainamp

# Process AXO stimulation data
kedro run --pipeline stroke_raw --tags mat.axo

# Process all BrainVision data
kedro run --pipeline stroke_raw --tags brainvision.all
```

#### Sleep Feature Detection
```bash
# EPISL sleep feature detection
kedro run --pipeline episl_events --tags sleep_features

# Slow wave detection
kedro run --pipeline episl_events --tags sw.detection

# Spindle detection
kedro run --pipeline episl_events --tags spindle.detection

# Wavelet-based slow wave detection
kedro run --pipeline episl_events --tags sw.detection.wavelet

# Hilbert-based slow wave detection
kedro run --pipeline episl_events --tags sw.detection.hilbert
```

#### Feature Extraction
```bash
# EPISL feature extraction
kedro run --pipeline episl_features --tags features

# Wavelet norm features
kedro run --pipeline episl_features --tags wavelet_norm

# Hilbert amplitude features
kedro run --pipeline episl_features --tags hilbert_amplitude

# Wavelet real features
kedro run --pipeline episl_features --tags wavelet_real

# Wavelet raw mean features
kedro run --pipeline episl_features --tags wavelet_raw_mean
```

#### Preprocessing and Storage
```bash
# EPISL intermediate processing
kedro run --pipeline episl_intermediate --tags preprocessing

# Delta band preprocessing
kedro run --pipeline episl_intermediate --tags bandpass

# Wavelet preprocessing
kedro run --pipeline episl_intermediate --tags wavelet

# Hilbert preprocessing
kedro run --pipeline episl_intermediate --tags hilbert

# Reference processing
kedro run --pipeline episl_intermediate --tags reference
```

## Adding New Studies

To add a new study:

1. **Create study directory**:
   ```bash
   mkdir clas/pipelines/your_study
   ```

2. **Create pipeline files**:
   - `pipeline_raw.py` - Raw data processing
   - `pipeline_intermediate.py` - Intermediate data processing
   - `pipeline_primary.py` - Primary data processing
   - `pipeline_events.py` - Event detection
   - `pipeline_features.py` - Feature extraction

3. **Import shared nodes**:
   ```python
   from ..nodes import d01_parse_mat, d02_intermediate, d03_primary, d03_events, d04_features
   ```

4. **Update pipeline registry** in `clas/pipeline_registry.py`

## Benefits

- **Modularity**: Shared nodes can be reused and tested independently
- **Maintainability**: Changes to shared logic affect all studies consistently
- **Clarity**: Clear separation between shared and study-specific code
- **Scalability**: Easy to add new studies without duplicating code
- **Flexibility**: Run specific tasks using tags for targeted processing