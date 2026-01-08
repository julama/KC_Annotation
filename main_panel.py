#!/usr/bin/env python3
"""
Main entry point for Panel-based EEG K-Complex Annotation Tool.

Usage:
    python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat
    
Or with SW events CSV:
    python main_panel.py --mat-file <path.mat> --sw-csv <path.csv>
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import panel as pn

from src.data_loader import load_mat_file
from src.preprocessing import preprocess_eeg
from src.epoch_manager import EpochManager
from src.sw_loader import load_sw_events
from src.annotation_manager import AnnotationManager
from src.panel_app.dashboard import create_dashboard
import config


def main():
    """Main function to launch the Panel annotation tool."""
    parser = argparse.ArgumentParser(
        description='EEG K-Complex Annotation Tool (Panel/HoloViews)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_panel.py --mat-file Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat
    python main_panel.py --mat-file data.mat --sw-csv sw_events.csv --channels 34 55 70
    python main_panel.py --mat-file data.mat --port 5007
        """
    )
    parser.add_argument('--mat-file', type=str, help='Path to .mat file')
    parser.add_argument('--sw-csv', type=str, help='Path to SW events CSV file')
    parser.add_argument('--annotation-file', type=str, default='annotations.csv',
                       help='Path to annotation output file (default: annotations.csv)')
    parser.add_argument('--sleep-stages', type=int, nargs='+',
                       default=config.DEFAULT_SLEEP_STAGES,
                       help=f'Sleep stages to analyze (default: {config.DEFAULT_SLEEP_STAGES})')
    parser.add_argument('--channels', type=int, nargs='+',
                       default=config.DEFAULT_CHANNEL_INDICES,
                       help=f'Focus channel indices to display (default: {config.DEFAULT_CHANNEL_INDICES})')
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
    if args.mat_file:
        mat_file = Path(args.mat_file)
    else:
        # Try to find a default .mat file
        default_mat = Path("Data/EPISL_01_W1/EPISL_01_W1/EPISL_01_W1_EEG_FiltDwn_05to30Hz.mat")
        if default_mat.exists():
            mat_file = default_mat
            print(f"Using default .mat file: {mat_file}")
        else:
            # Interactive file selection
            try:
                from tkinter import filedialog
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                mat_file = Path(filedialog.askopenfilename(
                    title="Select .mat file",
                    filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
                ))
                root.destroy()
            except ImportError:
                print("Error: Please provide --mat-file argument or install tkinter")
                sys.exit(1)
    
    if not mat_file or not mat_file.exists():
        print(f"Error: .mat file not found: {mat_file}")
        sys.exit(1)
    
    # SW events CSV
    if args.sw_csv:
        sw_csv = Path(args.sw_csv)
    else:
        # Try to find SW CSV with common naming patterns
        possible_sw_files = [
            mat_file.parent.parent / f"{mat_file.stem.replace('_EEG_FiltDwn_05to30Hz', '')}_sw_events.csv",
            mat_file.parent / f"{mat_file.stem}_sw_events.csv",
            Path("Data") / f"{mat_file.stem.replace('_EEG_FiltDwn_05to30Hz', '')}_sw_events.csv",
        ]
        
        sw_csv = None
        for possible_path in possible_sw_files:
            if possible_path.exists():
                sw_csv = possible_path
                print(f"Found SW events CSV: {sw_csv}")
                break
        
        if sw_csv is None:
            print(f"Warning: SW events CSV not found.")
            print("  You can create one with columns: start_idx, stop_idx")
    
    # Load .mat file
    print(f"\n{'='*60}")
    print(f"Loading .mat file: {mat_file}")
    print(f"{'='*60}")
    eeg_data_obj = load_mat_file(str(mat_file))
    
    # Apply preprocessing
    print("\nApplying preprocessing...")
    bandpass_range = tuple(args.bandpass) if args.bandpass else None
    processed_data = preprocess_eeg(
        eeg_data_obj.data,
        eeg_data_obj.srate,
        reference=args.reference,
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
    
    # Load SW events
    print("\nLoading SW events...")
    if sw_csv and sw_csv.exists():
        sw_events = load_sw_events(str(sw_csv))
    else:
        print("  No SW events CSV found. Creating empty DataFrame.")
        sw_events = pd.DataFrame(columns=['start_idx', 'stop_idx', 'event_id'])
    
    # Create annotation manager
    annotation_file = args.annotation_file
    if not Path(annotation_file).is_absolute():
        # Save in same directory as .mat file
        annotation_file = str(mat_file.parent / annotation_file)
    
    print(f"\nInitializing annotation manager (output: {annotation_file})...")
    annotation_manager = AnnotationManager(sw_events, annotation_file)
    
    # Create Panel dashboard
    print("\nCreating Panel dashboard...")
    dashboard = create_dashboard(
        epoch_manager=epoch_manager,
        sw_events=sw_events,
        annotation_manager=annotation_manager,
        focus_channels=args.channels,
        chanlocs=eeg_data_obj.chanlocs,  # Pass channel locations for topoplots
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

