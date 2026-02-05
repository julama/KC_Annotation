"""Configuration parameters for the EEG annotation tool

python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat

"""

# Epoch parameters
EPOCH_LENGTH_SECONDS = 20.0

# Default channel indices to display individually (0-based)
DEFAULT_CHANNEL_INDICES = [14, 10, 54] #Fz, 6, Pz

# Slow Wave filter parameters
SW_FILTER_LOW = 0.5  # Hz
SW_FILTER_HIGH = 18.0  # Hz

# Default sleep stages to analyze
DEFAULT_SLEEP_STAGES = [-2]#[1, 0, -1, -2, -3]

# Sampling rate (will be read from data, but default fallback)
DEFAULT_SAMPLING_RATE = 125.0  # Hz

# Channels to exclude from analysis (0-based indices)
# These will be discarded immediately after loading
EXCLUDE_CHANNELS = None #[42, 47, 48, 55, 62, 67, 72, 80, 87, 93, 98, 106, 112, 118, 119, 124, 125, 126, 127, 128]

# Channels to display in the main butterfly plot
# If None, it will use the automatic subsetting (max 30 channels)
# Example: [0, 1, 2, 3] or range(0, 64) (indices are 0-based)
# Frontocentral cluster (approx 30 channels) excluding the excluded_channels list
MAIN_PLOT_CHANNELS = [
    3, 4, 5, 6, 10, 11, 12, 18, 19, 20, 
    23, 24, 27, 28, 29, 30, 34, 35, 36, 
    39, 40, 41, 103, 104, 105, 109, 110, 111, 123
]

# Referencing configuration
# Options:
# - 'average': Use average of all channels
# - List[int]: Use average of specific channel indices (0-based, e.g., [48, 55] for mastoids)
# - None: No re-referencing
REFERENCE = [56, 99]

# Threshold line for amplitude visualization
# Set to None to disable threshold lines, or a numeric value (e.g., -35) to show threshold line
AMPLITUDE_THRESHOLD = -35.0
