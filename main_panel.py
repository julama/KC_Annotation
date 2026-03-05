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

from src.data_loader import load_mat_file
from src.preprocessing import preprocess_eeg
from src.epoch_manager import EpochManager
from src.annotation_manager import AnnotationManager
from src.panel_app.dashboard import create_dashboard
import config


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
        exclude_channels=config.EXCLUDE_CHANNELS,
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
