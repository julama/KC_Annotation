"""Configuration parameters for the EEG annotation tool"""

# Epoch parameters
EPOCH_LENGTH_SECONDS = 20.0

# Default channel indices to display individually
DEFAULT_CHANNEL_INDICES = [34, 55, 70]

# Slow Wave filter parameters
SW_FILTER_LOW = 0.5  # Hz
SW_FILTER_HIGH = 2.0  # Hz

# Default sleep stages to analyze
DEFAULT_SLEEP_STAGES = [1, 0, -1, -2, -3]

# Sampling rate (will be read from data, but default fallback)
DEFAULT_SAMPLING_RATE = 125.0  # Hz

