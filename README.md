# EEG K-Complex Annotation Tool

A Plotly Dash-based GUI application for annotating K-complexes in EEG data.

## Installation

1. Install dependencies:
## Installation

### Prerequisites
- Python 3.11 ([Microsoft Store](https://apps.microsoft.com/detail/9nrwmjp3717k))

### Steps
**0. Open the project in VSCode
'''
open a new terminal -> terminal: new terminal
'''

**1. Create and activate a virtual environment**
```bash
python -m venv kcannot
kcannot\Scripts\activate.bat
```

**2. Upgrade pip**
```bash
python -m pip install --upgrade pip
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the app

Each time you return to the project, activate the environment first:
```bash
kcannot\Scripts\activate.bat
# to start the app point to the mat file you want to load. E.g.
python main_panel.py --mat-file  C:\Users\extamackerj\Julian\SW_detection_v01\data_EPISL\EPISL_01_W1\EPISL_01_W1\EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat
```

## Usage

### Basic Usage

```bash
python main.py --mat-file path/to/file.mat --sw-csv path/to/sw_events.csv
```

### Command Line Options

- `--mat-file`: Path to .mat file containing EEG data (required, or will prompt for file selection)
- `--sw-csv`: Path to CSV file with SW events (columns: `start_idx`, `stop_idx`)
- `--annotation-file`: Output file for annotations (default: `annotations.csv`)
- `--sleep-stages`: Sleep stages to analyze (default: `1 0 -1 -2 -3`)
- `--channels`: Channel indices to display individually (default: `34 55 70`)
- `--reference`: Re-referencing method: `average`, `common`, or `none` (default: none)
- `--bandpass`: Bandpass filter range in Hz, e.g., `--bandpass 0.5 30`
- `--epoch-length`: Epoch length in seconds (default: 20.0)

### Example

```bash
python main.py \
  --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat \
  --sw-csv sw_events.csv \
  --sleep-stages -2 \
  --channels 34 55 70 \
  --reference average
```

## SW Events CSV Format

The SW events CSV file should have the following format:

```csv
start_idx,stop_idx
1000,2000
5000,6000
...
```

Where `start_idx` and `stop_idx` are sample indices (0-based).

## Annotation Output

Annotations are saved to a CSV file with columns:
- `start_idx`: Start sample index of SW event
- `stop_idx`: Stop sample index of SW event
- `event_id`: Unique event identifier
- `is_kc`: 1 for KC, 0 for non-KC, -1 for unannotated

## GUI Controls

- **Previous/Next Epoch**: Navigate through epochs
- **Click on SW events**: Toggle annotation (unannotated → KC → non-KC → unannotated)
- **Zoom/Pan**: Use Plotly's built-in zoom and pan tools
- **Home button**: Reset zoom to default view

## Project Structure

```
KC_Annotation/
├── src/
│   ├── data_loader.py          # Load .mat files
│   ├── preprocessing.py         # Re-referencing and filtering
│   ├── epoch_manager.py         # Epoch filtering and navigation
│   ├── sw_loader.py             # Load SW events from CSV
│   ├── annotation_manager.py    # Handle annotations
│   └── gui/
│       ├── app.py               # Main Dash application
│       └── components.py        # Plot components
├── config.py                    # Configuration parameters
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

## Features

- Load EEG data from .mat files (compatible with EPISL data format)
- Filter epochs by sleep stage
- Visualize raw epoch data with multiple channels
- Display individual channel plots for selected channels
- Generate topoplots for Slow Wave events (0.5-2 Hz filtered)
- Interactive navigation through epochs
- Click-to-annotate SW events as KC or non-KC
- Automatic saving of annotations to CSV
