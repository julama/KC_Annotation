# EEG K-Complex Annotation Tool

Panel-based GUI for annotating K-complex regions in EEG `.mat` files.

## Main Entry Point

Run the app with:

```bash
python main_panel.py --mat-file path/to/file.mat
```

If `--mat-file` is omitted, the app prompts for file selection.

## Installation

```bash
python -m venv kcannot
source kcannot/bin/activate  # Windows: kcannot\Scripts\activate
pip install -r requirements.txt
```

## Useful Options

- `--annotation-file`: Output CSV path (default: `annotations.csv`)
- `--sleep-stages`: Sleep stages to include
- `--channels`: Focus channel indices
- `--reference`: `average`, `common`, or `none`
- `--bandpass`: Filter range, e.g. `--bandpass 0.5 30`
- `--epoch-length`: Epoch length in seconds
- `--port`: Panel server port (default: `5006`)
- `--no-show`: Do not auto-open browser

## Minimal Project Structure

```
KC_Annotation/
├── main_panel.py
├── config.py
├── requirements.txt
├── Data/
│   └── channels.csv
└── src/
    ├── annotation_manager.py
    ├── data_loader.py
    ├── epoch_manager.py
    ├── preprocessing.py
    └── panel_app/
        ├── __init__.py
        └── dashboard.py
```

