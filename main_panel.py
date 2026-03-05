#!/usr/bin/env python3
"""
Main entry point for Panel-based EEG K-Complex Annotation Tool.

Usage:
    python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat
    python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_02_W1/EPISL_02_W1_EEG_FiltDwn_05to30Hz.mat
    
For standalone EXE:
    kc_annotation.exe --mat-file <path_to_file.mat>
    kc_annotation.exe  # Opens file dialog
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import panel as pn
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from src.data_loader import load_mat_file
from src.preprocessing import preprocess_eeg
from src.epoch_manager import EpochManager
from src.annotation_manager import AnnotationManager
from src.panel_app.dashboard import create_dashboard
import config

def _normalize_artndxn_shape(artndxn: np.ndarray, n_epochs: int, n_channels: int) -> np.ndarray:
    """Normalize artndxn to shape (epochs, channels) when possible."""
    if artndxn is None:
        return None

    arr = np.asarray(artndxn)
    if arr.ndim != 2:
        return None

    if arr.shape == (n_epochs, n_channels):
        return arr
    if arr.shape == (n_channels, n_epochs):
        return arr.T
    if arr.shape[1] == n_channels:
        return arr
    if arr.shape[0] == n_channels:
        return arr.T
    return None


def _make_unique_channel_names(raw_names):
    """Ensure channel names are unique for MNE."""
    seen = {}
    unique = []
    for idx, name in enumerate(raw_names):
        base = str(name) if name is not None and str(name) else f"ch_{idx}"
        count = seen.get(base, 0)
        if count == 0:
            unique_name = base
        else:
            unique_name = f"{base}_{count}"
        seen[base] = count + 1
        unique.append(unique_name)
    return unique


def _interpolate_globally_bad_channels(eeg_data_obj):
    """
    Interpolate channels that are marked bad (0) across all epochs in EEG.artndxn.
    Returns a list of interpolated channel indices.
    """
    if not MNE_AVAILABLE:
        print("Warning: MNE not available; skipping bad-channel interpolation.")
        return []

    if eeg_data_obj.artndxn is None:
        return []

    n_channels = eeg_data_obj.data.shape[1]
    n_epochs = len(eeg_data_obj.visnum)
    artndxn = _normalize_artndxn_shape(eeg_data_obj.artndxn, n_epochs=n_epochs, n_channels=n_channels)
    if artndxn is None:
        print("Warning: Could not align EEG.artndxn to (epochs, channels). Skipping interpolation.")
        return []

    globally_bad_mask = np.all(artndxn == 0, axis=0)
    bad_indices = np.where(globally_bad_mask)[0].tolist()
    if not bad_indices:
        print("No globally bad channels detected in EEG.artndxn.")
        return []

    print(f"Globally bad channels detected (to interpolate): {bad_indices}")

    try:
        chanlocs = eeg_data_obj.chanlocs
        if chanlocs.empty:
            print("Warning: No channel locations available; skipping interpolation.")
            return []

        if 'labels' in chanlocs.columns and len(chanlocs['labels']) >= n_channels:
            channel_names = _make_unique_channel_names(chanlocs['labels'].tolist()[:n_channels])
        else:
            channel_names = [f"ch_{i}" for i in range(n_channels)]

        if all(col in chanlocs.columns for col in ['X', 'Y', 'Z']):
            x_vals = chanlocs['X'].values
            y_vals = chanlocs['Y'].values
            z_vals = chanlocs['Z'].values
        elif all(col in chanlocs.columns for col in ['x', 'y', 'z']):
            x_vals = chanlocs['x'].values
            y_vals = chanlocs['y'].values
            z_vals = chanlocs['z'].values
        else:
            print("Warning: No XYZ channel coordinates found; skipping interpolation.")
            return []

        ch_pos = {}
        for idx in range(min(n_channels, len(x_vals), len(y_vals), len(z_vals))):
            xyz = np.array([x_vals[idx], y_vals[idx], z_vals[idx]], dtype=float)
            if np.all(np.isfinite(xyz)):
                ch_pos[channel_names[idx]] = xyz

        if len(ch_pos) < 4:
            print("Warning: Not enough valid channel positions for interpolation.")
            return []

        info = mne.create_info(ch_names=channel_names, sfreq=eeg_data_obj.srate, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(eeg_data_obj.data.values.T, info, verbose='ERROR')
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        raw.set_montage(montage, on_missing='ignore')
        raw.info['bads'] = [channel_names[idx] for idx in bad_indices if idx < len(channel_names)]

        raw.interpolate_bads(reset_bads=False, verbose='ERROR')
        eeg_data_obj.data = pd.DataFrame(
            raw.get_data().T,
            columns=eeg_data_obj.data.columns,
            index=eeg_data_obj.data.index,
        )
        print(f"Interpolated {len(raw.info['bads'])} globally bad channels using MNE.")
        return bad_indices
    except Exception as exc:
        print(f"Warning: MNE interpolation failed: {exc}")
        return []


def get_base_path():
    """Get the base path for the application.
    
    Returns the appropriate base path whether running as:
    - A frozen PyInstaller executable (uses sys._MEIPASS)
    - A regular Python script (uses script directory)
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        return Path(sys._MEIPASS)
    else:
        # Running as a script
        return Path(__file__).parent


def is_frozen():
    """Check if running as a frozen PyInstaller executable."""
    return getattr(sys, 'frozen', False)


def main():
    """Main function to launch the Panel annotation tool."""
    # Get base path for resource loading
    base_path = get_base_path()
    
    # Print startup info
    if is_frozen():
        print("Running as standalone executable")
        print(f"Resource path: {base_path}")
    
    parser = argparse.ArgumentParser(
        description='EEG K-Complex Annotation Tool (Panel/HoloViews)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat
    python main_panel.py --mat-file data.mat --channels 34 55 70
    python main_panel.py --mat-file data.mat --port 5007
    
    # Standalone EXE:
    kc_annotation.exe --mat-file <path_to_file.mat>
    kc_annotation.exe  # Opens file dialog to select .mat file
        """
    )
    parser.add_argument('--mat-file', type=str, help='Path to .mat file')
    parser.add_argument('--annotation-file', type=str, default='annotations.csv',
                       help='Path to annotation output file (default: annotations.csv)')
    parser.add_argument('--sleep-stages', type=int, nargs='+',
                       default=config.DEFAULT_SLEEP_STAGES,
                       help=f'Sleep stages to analyze (default: {config.DEFAULT_SLEEP_STAGES})')
    parser.add_argument('--channels', type=int, nargs='+',
                       default=config.DEFAULT_CHANNEL_INDICES,
                       help=f'Focus channel indices to display (0-based, default: {config.DEFAULT_CHANNEL_INDICES})')
    parser.add_argument('--reference', type=str, choices=['average', 'common', 'none'],
                       default=None, help='Re-referencing method (default: none)')
    parser.add_argument('--bandpass', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                       help='Bandpass filter range in Hz (e.g., --bandpass 0.5 30)')
    parser.add_argument('--epoch-length', type=float, default=config.EPOCH_LENGTH_SECONDS,
                       help=f'Epoch length in seconds (default: {config.EPOCH_LENGTH_SECONDS})')
    parser.add_argument('--port', type=int, default=5006,
                       help='Port to run the Panel server (default: 5006)')
    parser.add_argument('--show', action='store_true', default=True,
                       help='Open browser automatically (default: True)')
    parser.add_argument('--no-show', action='store_false', dest='show',
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Get file paths
    mat_file = None
    if args.mat_file:
        mat_file = Path(args.mat_file)
    else:
        # Try to find a default .mat file (only when not frozen)
        if not is_frozen():
            default_mat = Path("Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat")
            if default_mat.exists():
                mat_file = default_mat
                print(f"Using default .mat file: {mat_file}")
        
        # If no file found, use file dialog
        if mat_file is None:
            try:
                from tkinter import filedialog
                import tkinter as tk
                
                print("Opening file selection dialog...")
                root = tk.Tk()
                root.withdraw()
                # Bring dialog to front
                root.lift()
                root.attributes('-topmost', True)
                
                selected_file = filedialog.askopenfilename(
                    title="Select EEG .mat file",
                    filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
                )
                root.destroy()
                
                if selected_file:
                    mat_file = Path(selected_file)
                else:
                    print("No file selected. Exiting.")
                    sys.exit(0)
                    
            except ImportError:
                print("Error: Please provide --mat-file argument or install tkinter")
                sys.exit(1)
            except Exception as e:
                print(f"Error opening file dialog: {e}")
                print("Please provide --mat-file argument instead.")
                sys.exit(1)
    
    if not mat_file or not mat_file.exists():
        print(f"Error: .mat file not found: {mat_file}")
        sys.exit(1)
    
    # Load .mat file
    print(f"\n{'='*60}")
    print(f"Loading .mat file: {mat_file}")
    print(f"{'='*60}")
    eeg_data_obj = load_mat_file(str(mat_file))

    # Interpolate globally bad channels from EEG.artndxn (0=bad, 1=good)
    interpolated_bad_channels = _interpolate_globally_bad_channels(eeg_data_obj)
    
    # Apply preprocessing
    print("\nApplying preprocessing...")
    bandpass_range = tuple(args.bandpass) if args.bandpass else None
    
    # Determine reference (CLI arg overrides config if provided)
    reference = args.reference if args.reference is not None else config.REFERENCE
    
    processed_data = preprocess_eeg(
        eeg_data_obj.data,
        eeg_data_obj.srate,
        reference=reference,
        bandpass=bandpass_range
    )
    
    # Create epoch manager
    print("\nCreating epoch manager...")
    epoch_manager = EpochManager(
        processed_data,
        eeg_data_obj.visnum,
        eeg_data_obj.srate,
        epoch_length_sec=args.epoch_length,
        sleep_stages=args.sleep_stages
    )
    
    # Create annotation manager (no longer needs SW events)
    annotation_file = args.annotation_file
    if not Path(annotation_file).is_absolute():
        # Save in same directory as .mat file
        annotation_file = str(mat_file.parent / annotation_file)
    
    print(f"\nInitializing annotation manager (output: {annotation_file})...")
    annotation_manager = AnnotationManager(annotation_file, epoch_manager.sampling_rate)
    
    # Find channels.csv file
    channels_file = None
    possible_channels_files = [
        mat_file.parent / "channels.csv",
        mat_file.parent.parent / "channels.csv",
        Path("Data/channels.csv"),
    ]
    # Also check in the bundled resources for frozen apps
    if is_frozen():
        possible_channels_files.append(base_path / "Data" / "channels.csv")
    
    for path in possible_channels_files:
        if path.exists():
            channels_file = str(path)
            print(f"Found channels file: {channels_file}")
            break
    
    # Create Panel dashboard
    print("\nCreating Panel dashboard...")
    dashboard = create_dashboard(
        epoch_manager=epoch_manager,
        annotation_manager=annotation_manager,
        focus_channels=args.channels,
        main_plot_channels=config.MAIN_PLOT_CHANNELS,
        chanlocs=eeg_data_obj.chanlocs,  # Pass channel locations for topoplots
        channels_file=channels_file,
        exclude_channels=sorted(set((config.EXCLUDE_CHANNELS or []) + interpolated_bad_channels)),
    )
    
    # Wrap in a servable template
    template = pn.template.FastListTemplate(
        title="EEG K-Complex Annotation Tool",
        main=[dashboard],
        accent_base_color="#3498db",
        header_background="#2c3e50",
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Panel server on port {args.port}...")
    print(f"Open your browser to: http://localhost:{args.port}")
    print(f"{'='*60}\n")
    
    # Serve the dashboard
    pn.serve(
        template,
        port=args.port,
        show=args.show,
        title="EEG KC Annotation",
        websocket_origin=['localhost:' + str(args.port)],
    )


if __name__ == '__main__':
    main()
