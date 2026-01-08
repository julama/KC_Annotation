"""Main entry point for EEG K-Complex Annotation Tool"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import load_mat_file
from src.preprocessing import preprocess_eeg
from src.epoch_manager import EpochManager
from src.sw_loader import load_sw_events
from src.annotation_manager import AnnotationManager
from src.gui.app import create_app
import config


def main():
    """Main function to launch the annotation tool"""
    parser = argparse.ArgumentParser(description='EEG K-Complex Annotation Tool')
    parser.add_argument('--mat-file', type=str, help='Path to .mat file')
    parser.add_argument('--sw-csv', type=str, help='Path to SW events CSV file')
    parser.add_argument('--annotation-file', type=str, default='annotations.csv',
                       help='Path to annotation output file (default: annotations.csv)')
    parser.add_argument('--sleep-stages', type=int, nargs='+',
                       default=config.DEFAULT_SLEEP_STAGES,
                       help=f'Sleep stages to analyze (default: {config.DEFAULT_SLEEP_STAGES})')
    parser.add_argument('--channels', type=int, nargs='+',
                       default=config.DEFAULT_CHANNEL_INDICES,
                       help=f'Channel indices to display (default: {config.DEFAULT_CHANNEL_INDICES})')
    parser.add_argument('--reference', type=str, choices=['average', 'common', 'none'],
                       default=None, help='Re-referencing method (default: none)')
    parser.add_argument('--bandpass', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                       help='Bandpass filter range in Hz (e.g., --bandpass 0.5 30)')
    parser.add_argument('--epoch-length', type=float, default=config.EPOCH_LENGTH_SECONDS,
                       help=f'Epoch length in seconds (default: {config.EPOCH_LENGTH_SECONDS})')
    
    args = parser.parse_args()
    
    # Get file paths
    if args.mat_file:
        mat_file = Path(args.mat_file)
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
    
    if not mat_file.exists():
        print(f"Error: .mat file not found: {mat_file}")
        sys.exit(1)
    
    # SW events CSV
    if args.sw_csv:
        sw_csv = Path(args.sw_csv)
    else:
        # Try to find SW CSV in same directory as .mat file
        sw_csv = mat_file.parent / f"{mat_file.stem}_sw_events.csv"
        if not sw_csv.exists():
            print(f"Warning: SW events CSV not found. Expected at: {sw_csv}")
            print("  You can create one with columns: start_idx, stop_idx")
            sw_csv = None
    
    # Load .mat file
    print(f"\nLoading .mat file: {mat_file}")
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
    if sw_csv and Path(sw_csv).exists():
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
    
    # Create and run Dash app
    print("\nStarting GUI...")
    app = create_app(
        processed_data,
        eeg_data_obj.visnum,
        eeg_data_obj.srate,
        eeg_data_obj.chanlocs,
        sw_events,
        annotation_manager,
        epoch_manager,
        channel_indices=args.channels
    )
    
    print("\n" + "="*60)
    print("GUI is starting. Open your browser to view the annotation tool.")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)


if __name__ == '__main__':
    main()

