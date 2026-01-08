import mne
import numpy as np
import pandas as pd
import os
import struct
from KC_algorithm.model import score_KCs
from KC_algorithm.plotting import KC_from_probas


def parse_edf_header_manual(edf_file):
    """
    Manually parse EDF header bytes to extract channel information.
    This is a fallback when pyedflib and MNE fail to detect channels.
    
    EDF format:
    - Bytes 0-7: version (8 bytes)
    - Bytes 8-87: patient info (80 bytes)
    - Bytes 88-167: recording info (80 bytes)
    - Bytes 168-175: start date (8 bytes)
    - Bytes 176-183: start time (8 bytes)
    - Bytes 184-191: header size (8 bytes, ASCII)
    - Bytes 192-195: reserved (44 bytes)
    - Bytes 236-239: number of data records (8 bytes, ASCII)
    - Bytes 244-247: duration of data record (8 bytes, ASCII)
    - Bytes 252-255: number of signals (4 bytes, ASCII)
    - Then 256 bytes per signal for signal headers
    """
    with open(edf_file, 'rb') as f:
        # Read main header (256 bytes)
        header = f.read(256)
        
        if len(header) < 256:
            raise ValueError(f"EDF file too short: only {len(header)} bytes")
        
        # Extract number of signals from bytes 252-255 (4 bytes, ASCII)
        try:
            n_signals_str = header[252:256].decode('ascii', errors='ignore').strip()
            n_signals = int(n_signals_str) if n_signals_str.isdigit() else 0
            print(f"  Manual header parse: found {n_signals} signals in header")
        except:
            n_signals = 0
        
        # Extract data record duration (bytes 244-247, 8 bytes ASCII)
        try:
            record_duration_str = header[244:252].decode('ascii', errors='ignore').strip()
            record_duration = float(record_duration_str) if record_duration_str else 1.0
        except:
            record_duration = 1.0
        
        # Extract number of data records (bytes 236-239, 8 bytes ASCII)
        try:
            n_records_str = header[236:244].decode('ascii', errors='ignore').strip()
            n_records = int(n_records_str) if n_records_str.isdigit() else 0
        except:
            n_records = 0
        
        if n_signals == 0:
            # Try to infer from signal header section
            # Signal headers start at byte 256 and are 256 bytes total
            # Each signal has 16 bytes per field
            f.seek(256)
            signal_headers = f.read(256)
            
            # Try to count signals by looking for non-empty channel labels
            # Channel labels are at offset 0 in signal header (16 bytes each)
            # So we have n_signals * 16 bytes for labels
            if len(signal_headers) >= 16:
                # Try reading first label
                first_label = signal_headers[0:16].decode('ascii', errors='ignore').strip()
                if first_label:
                    # Estimate: if we have at least one label, try to count how many we have
                    # Each signal header is 16 bytes per field, with 16 fields = 256 bytes total
                    # But labels are first, so we can count non-empty 16-byte chunks
                    label_count = 0
                    for i in range(0, min(256, len(signal_headers)), 16):
                        label = signal_headers[i:i+16].decode('ascii', errors='ignore').strip()
                        if label:
                            label_count += 1
                        else:
                            break
                    if label_count > 0:
                        n_signals = label_count
                        print(f"  Inferred {n_signals} signals from signal headers")
        
        if n_signals == 0:
            return None
        
        # Read signal headers (256 bytes total, 16 bytes per field)
        f.seek(256)
        signal_headers = f.read(256 * n_signals)
        
        if len(signal_headers) < 256:
            # If we don't have full headers, try to read what we can
            signal_headers = f.read(256)
        
        # Parse signal information
        # Each signal has 16 fields of 16 bytes each (256 bytes total per signal)
        # Field 0: label (16 bytes)
        # Field 1: transducer type (16 bytes)
        # Field 2: physical dimension (16 bytes)
        # Field 3: physical minimum (16 bytes)
        # Field 4: physical maximum (16 bytes)
        # Field 5: digital minimum (16 bytes)
        # Field 6: digital maximum (16 bytes)
        # Field 7: prefiltering (16 bytes)
        # Field 8: number of samples in data record (16 bytes)
        # Fields 9-15: reserved
        
        ch_labels = []
        n_samples_per_record = []
        
        for sig_idx in range(n_signals):
            offset = sig_idx * 256
            
            # Extract label (first 16 bytes)
            label = signal_headers[offset:offset+16].decode('ascii', errors='ignore').strip()
            if not label:
                label = f'EEG_{sig_idx+1}'
            ch_labels.append(label)
            
            # Extract samples per record (field 8, bytes 128-143)
            try:
                samples_str = signal_headers[offset+128:offset+144].decode('ascii', errors='ignore').strip()
                samples_per_rec = int(samples_str) if samples_str.isdigit() else 0
                n_samples_per_record.append(samples_per_rec)
            except:
                n_samples_per_record.append(0)
        
        # Calculate sampling frequency
        if n_samples_per_record[0] > 0 and record_duration > 0:
            sfreq = n_samples_per_record[0] / record_duration
        else:
            sfreq = 128.0  # Default fallback
        
        # Calculate total samples
        total_samples = n_samples_per_record[0] * n_records if n_records > 0 else 0
        
        print(f"  Parsed {n_signals} channels: {ch_labels}")
        print(f"  Sampling frequency: {sfreq:.2f} Hz")
        print(f"  Total samples: {total_samples}")
        print(f"  Record duration: {record_duration} seconds")
        print(f"  Number of records: {n_records}")
        
        return {
            'n_signals': n_signals,
            'ch_labels': ch_labels,
            'sfreq': sfreq,
            'n_samples_per_record': n_samples_per_record,
            'n_records': n_records,
            'total_samples': total_samples,
            'record_duration': record_duration
        }


def read_edf_data_manual(edf_file, header_info):
    """
    Manually read EDF data bytes based on parsed header information.
    """
    n_signals = header_info['n_signals']
    n_samples_per_record = header_info['n_samples_per_record']
    n_records = header_info['n_records']
    
    # Calculate header size: 256 (main) + 256 * n_signals (signal headers)
    header_size = 256 + 256 * n_signals
    
    # Each data record contains samples for all signals
    # Samples are stored as 2-byte integers (int16)
    # Record size = sum of samples_per_record for all signals * 2 bytes
    
    with open(edf_file, 'rb') as f:
        # Skip header
        f.seek(header_size)
        
        # Read all data records
        all_data = []
        
        for sig_idx in range(n_signals):
            signal_data = []
            samples_per_rec = n_samples_per_record[sig_idx]
            
            # Calculate byte offset for this signal in each record
            # Signals are interleaved: signal0 samples, signal1 samples, etc.
            # So we need to skip samples from previous signals
            bytes_per_sample = 2
            skip_bytes_before = sum(n_samples_per_record[:sig_idx]) * bytes_per_sample
            bytes_per_record_for_signal = samples_per_rec * bytes_per_sample
            
            for rec_idx in range(n_records):
                # Position at start of record
                record_start = header_size + rec_idx * (sum(n_samples_per_record) * bytes_per_sample)
                
                # Skip to this signal's data
                f.seek(record_start + skip_bytes_before)
                
                # Read this signal's samples for this record
                data_bytes = f.read(bytes_per_record_for_signal)
                
                if len(data_bytes) < bytes_per_record_for_signal:
                    print(f"  Warning: Only read {len(data_bytes)} bytes, expected {bytes_per_record_for_signal}")
                    break
                
                # Unpack as int16 (little-endian)
                samples = np.frombuffer(data_bytes, dtype=np.int16)
                signal_data.extend(samples)
            
            all_data.append(np.array(signal_data))
        
        return np.array(all_data)


def create_raw_from_pyedflib(edf_file):
    """
    Create an MNE Raw object from EDF file using pyedflib when MNE fails.
    This is a fallback method when MNE cannot read the file directly.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required for this fallback method. Install with: pip install pyedflib")
    
    f = pyedflib.EdfReader(edf_file)
    n_channels = f.signals_in_file
    
    # If no channels detected, try to read header manually
    if n_channels == 0:
        print("  pyedflib reports 0 channels. Attempting manual EDF header parsing...")
        
        # Try to read raw header bytes to inspect
        try:
            # Get file header info
            file_duration = f.file_duration
            print(f"  File duration: {file_duration} seconds")
            
            # Try multiple approaches to read data
            # Approach 1: Try getNSamples() - sometimes works even when signals_in_file is 0
            try:
                n_samples_list = f.getNSamples()
                print(f"  getNSamples() returned: {n_samples_list} (length: {len(n_samples_list) if isinstance(n_samples_list, list) else 'N/A'})")
                
                # If we have sample info, there might be data
                if isinstance(n_samples_list, (list, tuple)) and len(n_samples_list) > 0 and n_samples_list[0] > 0:
                    print(f"  Found {len(n_samples_list)} potential signals with {n_samples_list[0]} samples")
                    try:
                        # Try reading signal 0
                        signal_data = f.readSignal(0)
                        if len(signal_data) > 0:
                            print(f"  Successfully read {len(signal_data)} samples from signal 0")
                            data = np.array([signal_data])
                            
                            # Calculate sampling frequency
                            sfreq = n_samples_list[0] / file_duration if file_duration > 0 else 128.0
                            if sfreq <= 0:
                                # Fallback: estimate from data length
                                sfreq = len(signal_data) / file_duration if file_duration > 0 else 128.0
                            
                            # Try to get channel labels from file header
                            ch_labels = ['EEG_C3']  # Default name
                            try:
                                # Try to read the EDF header bytes directly
                                all_labels = f.getSignalLabels()
                                if all_labels and len(all_labels) > 0:
                                    ch_labels = all_labels[:1]  # Use first label if available
                                    print(f"  Found channel label in header: {ch_labels[0]}")
                            except:
                                pass
                            
                            print(f"  Created single channel '{ch_labels[0]}' with {len(signal_data)} samples at {sfreq:.2f} Hz")
                            
                            f.close()
                            
                            info = mne.create_info(ch_names=ch_labels, sfreq=sfreq, ch_types=['eeg'])
                            raw = mne.io.RawArray(data, info)
                            return raw
                    except Exception as e_read_signal:
                        print(f"  Failed to read signal 0: {e_read_signal}")
            except Exception as e_nsamples:
                print(f"  getNSamples() failed: {e_nsamples}")
            
            # Approach 2: Try reading from raw MNE data if it exists
            # (This would only work if we have access to the raw object created earlier)
            
            f.close()
            
            # Last resort: try manual binary header parsing
            print("\n  Attempting manual binary EDF header parsing...")
            try:
                header_info = parse_edf_header_manual(edf_file)
                if header_info and header_info['n_signals'] > 0:
                    print("  Manual parsing successful! Reading data...")
                    data = read_edf_data_manual(edf_file, header_info)
                    
                    # Convert digital values to physical values if needed
                    # For now, just use raw values - user can scale later if needed
                    # EDF typically stores as int16, we'll keep as float
                    data = data.astype(np.float64)
                    
                    # Create MNE info structure
                    ch_labels = header_info['ch_labels']
                    sfreq = header_info['sfreq']
                    
                    info = mne.create_info(ch_names=ch_labels, sfreq=sfreq, ch_types=['eeg'] * len(ch_labels))
                    raw = mne.io.RawArray(data, info)
                    
                    print(f"  Successfully created Raw object with {len(ch_labels)} channels from manual parsing")
                    return raw
            except Exception as e_manual:
                print(f"  Manual parsing also failed: {e_manual}")
            
            raise ValueError(
                "Cannot read EDF file: pyedflib reports 0 channels and manual parsing failed. "
                "The file header may be corrupted or in a non-standard format. "
                "Consider using a tool like EDFbrowser or converting the file to a different format."
            )
                
        except Exception as e_header:
            print(f"  Failed to parse header: {e_header}")
            f.close()
            raise ValueError(
                f"Cannot read EDF file: pyedflib reports 0 channels and header parsing failed: {e_header}"
            )
    
    ch_labels = f.getSignalLabels()
    
    # Get sampling frequency - it may differ per channel, use first channel
    n_samples = f.getNSamples()[0]
    file_duration = f.file_duration
    sfreq = n_samples / file_duration if file_duration > 0 else 128.0  # Default fallback
    
    # Read all channel data
    data = []
    for i in range(n_channels):
        signal_data = f.readSignal(i)
        data.append(signal_data)
    
    data = np.array(data)  # shape: (n_channels, n_samples)
    
    # Create MNE info structure
    ch_types = ['eeg'] * n_channels  # Default all to EEG, user can adjust later
    info = mne.create_info(ch_names=ch_labels, sfreq=sfreq, ch_types=ch_types)
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    f.close()
    return raw


def main():
    # Path to local EDF file (relative to this script's location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edf_file = os.path.join(script_dir, 'annotated_KC', '01-02-0002 KComplexes_E1.edf')
    
    # Default channel type mappings (adjust if needed)
    mapping = {'EOG horizontal': 'eog',
               'EOG': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'EMG': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}

    #### Load edf file##
    print(f"Loading EDF file: {edf_file}")
    # Try reading with exclude parameter to ensure no channels are excluded
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='warning', exclude=[])
    except Exception as e:
        print(f"Error with exclude=[], trying without exclude parameter: {e}")
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=True)
    
    # Print detailed info about the raw object
    print(f"\nRaw object info:")
    print(f"  Number of channels: {len(raw.info['ch_names'])}")
    print(f"  Channel names: {raw.info['ch_names']}")
    print(f"  Channel types: {[raw.get_channel_types()[i] for i in range(len(raw.info['ch_names']))]}")
    print(f"  Sampling frequency: {raw.info['sfreq']} Hz")
    
    if len(raw.info['ch_names']) == 0:
        # Try to get data directly to see if it exists
        print("\nAttempting to inspect file structure...")
        try:
            # Try to access the data array
            data_shape = raw._data.shape if hasattr(raw, '_data') else "No _data attribute"
            print(f"  Data shape (if available): {data_shape}")
            print(f"  Raw info keys: {list(raw.info.keys())}")
            
            # Check if chs list has any information
            chs_list = raw.info.get('chs', [])
            print(f"  Number of entries in 'chs': {len(chs_list)}")
            if len(chs_list) > 0:
                print(f"  First ch entry keys: {list(chs_list[0].keys()) if chs_list else 'N/A'}")
                print(f"  First ch entry: {chs_list[0] if chs_list else 'N/A'}")
            
            # Try using pyedflib directly as alternative
            try:
                import pyedflib
                print("\nTrying to read with pyedflib to inspect file structure...")
                f = pyedflib.EdfReader(edf_file)
                n_channels = f.signals_in_file
                print(f"  pyedflib found {n_channels} channels")
                ch_labels = f.getSignalLabels()
                print(f"  Channel labels: {ch_labels}")
                
                # Get additional channel info
                if n_channels > 0:
                    ch_info = []
                    for i in range(n_channels):
                        try:
                            n_samples = f.getNSamples()[i]
                            physical_max = f.getPhysicalMaximum(i)
                            physical_min = f.getPhysicalMinimum(i)
                            digital_max = f.getDigitalMaximum(i)
                            digital_min = f.getDigitalMinimum(i)
                            prefilter = f.getPrefilter(i)
                            transducer = f.getTransducer(i)
                            print(f"  Channel {i} ({ch_labels[i]}): {n_samples} samples, "
                                  f"phys_range=[{physical_min}, {physical_max}], "
                                  f"dig_range=[{digital_min}, {digital_max}]")
                            ch_info.append({
                                'label': ch_labels[i],
                                'n_samples': n_samples,
                                'physical_range': (physical_min, physical_max),
                                'digital_range': (digital_min, digital_max)
                            })
                        except Exception as e_ch:
                            print(f"    Error getting info for channel {i}: {e_ch}")
                    
                    f.close()
                    
                    # Try reading again with explicit channel handling
                    print(f"\nAttempting to recreate raw object with channel information...")
                    # Try different approaches
                    for attempt in [
                        {'preload': True, 'verbose': True, 'stim_channel': None, 'exclude': []},
                        {'preload': True, 'verbose': True, 'exclude': []},
                        {'preload': False, 'verbose': True, 'exclude': []},
                    ]:
                        try:
                            print(f"  Trying parameters: {attempt}")
                            raw_retry = mne.io.read_raw_edf(edf_file, **attempt)
                            if len(raw_retry.info['ch_names']) > 0:
                                print(f"  Success! Found {len(raw_retry.info['ch_names'])} channels")
                                if attempt.get('preload', True):
                                    raw = raw_retry
                                else:
                                    raw = raw_retry.copy().load_data()
                                break
                        except Exception as e_retry:
                            print(f"  Failed: {e_retry}")
                    else:
                        # If all attempts failed, try to manually create raw object from pyedflib
                        print("\nAll retry attempts failed. Creating MNE Raw object from pyedflib data...")
                        try:
                            raw = create_raw_from_pyedflib(edf_file)
                            print(f"  Successfully created Raw object with {len(raw.info['ch_names'])} channels from pyedflib")
                        except Exception as e_py:
                            raise ValueError(
                                f"File has {n_channels} channels according to pyedflib but MNE cannot read them. "
                                f"Channel labels: {ch_labels}. Error creating Raw from pyedflib: {e_py}. "
                                f"Consider converting the file to a different format."
                            )
                else:
                    # n_channels is 0 - try to recover using create_raw_from_pyedflib which has special handling
                    print("\n  pyedflib reports 0 channels. Attempting recovery method...")
                    f.close()
                    try:
                        raw = create_raw_from_pyedflib(edf_file)
                        print(f"  Successfully created Raw object with {len(raw.info['ch_names'])} channels using recovery method")
                        # Success - raw object created with channels
                    except Exception as e_recover:
                        print(f"  Recovery method failed: {e_recover}")
                        # Try manual parsing as last resort
                        print("  Attempting manual binary EDF header parsing...")
                        try:
                            header_info = parse_edf_header_manual(edf_file)
                            if header_info and header_info['n_signals'] > 0:
                                print("  Manual parsing successful! Reading data...")
                                data = read_edf_data_manual(edf_file, header_info)
                                data = data.astype(np.float64)
                                
                                ch_labels = header_info['ch_labels']
                                sfreq = header_info['sfreq']
                                
                                info = mne.create_info(ch_names=ch_labels, sfreq=sfreq, ch_types=['eeg'] * len(ch_labels))
                                raw = mne.io.RawArray(data, info)
                                print(f"  Successfully created Raw object with {len(ch_labels)} channels from manual parsing")
                                # Success - raw object created with channels
                        except Exception as e_manual:
                            print(f"  Manual parsing failed: {e_manual}")
                        # Keep going to try other methods below
            except ImportError:
                print("  pyedflib not available - attempting alternative methods...")
                print("  NOTE: Installing pyedflib (pip install pyedflib) may help with this file format issue.")
                # Try reading as BDF instead of EDF
                try:
                    print("  Trying to read as BDF format...")
                    raw = mne.io.read_raw_bdf(edf_file, preload=True, verbose=True)
                    if len(raw.info['ch_names']) > 0:
                        print(f"  Success! BDF format worked. Found {len(raw.info['ch_names'])} channels")
                except Exception as e_bdf:
                    print(f"  BDF reading failed: {e_bdf}")
                    
                # Try with different EDF reading parameters
                try:
                    print("  Trying EDF with infer_types=False...")
                    raw = mne.io.read_raw_edf(
                        edf_file, 
                        preload=True, 
                        verbose=True,
                        infer_types=False,
                        exclude=[]
                    )
                    if len(raw.info['ch_names']) > 0:
                        print(f"  Success with infer_types=False! Found {len(raw.info['ch_names'])} channels")
                except Exception as e_infer:
                    print(f"  infer_types=False failed: {e_infer}")
                    
                # If still no channels, suggest installing pyedflib
                if len(raw.info['ch_names']) == 0:
                    print("\n  RECOMMENDATION: This file appears to have a format issue that MNE cannot handle.")
                    print("  Please try: pip install pyedflib")
                    print("  Then rerun this script. Pyedflib can read the file and convert it to MNE format.")
            except Exception as e2:
                print(f"  pyedflib error: {e2}")
        except Exception as e:
            print(f"  Error inspecting file: {e}")
        
        if len(raw.info['ch_names']) == 0:
            raise ValueError(
                "No channels found in EDF file. The file may be corrupted, "
                "in an unsupported format, or MNE may not be able to parse the channel information. "
                "Try installing pyedflib (pip install pyedflib) to get better file inspection capabilities, "
                "or check the file with a different tool."
            )
    
    # Only set reference if we have EEG channels
    ch_types = raw.get_channel_types()
    has_eeg = any(ch_type in ['eeg', 'ecog', 'seeg', 'dbs'] for ch_type in ch_types)
    
    if has_eeg:
        raw, _ = mne.set_eeg_reference(raw, [], verbose='warning')
    else:
        print("Warning: No EEG channels detected in channel types.")
        print(f"  Current channel types: {ch_types}")
        # Set all channels to EEG type if none are detected
        print("Attempting to set all channels as EEG type...")
        raw.set_channel_types({ch: 'eeg' for ch in raw.info['ch_names']})
        raw, _ = mne.set_eeg_reference(raw, [], verbose='warning')
    
    raw.resample(128)
    raw = raw.filter(0.3, None)
    Fs = raw.info['sfreq']
    
    print(f"Sampling frequency: {Fs} Hz")
    print(f"Recording duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")

    ### Load and transform hypnogram ####
    # Check if annotations exist in the file
    if len(raw.annotations) > 0:
        print(f"\nFound {len(raw.annotations)} annotations in the file")
        print(f"Annotation descriptions: {set(raw.annotations.description)}")
        
        # Try to extract sleep stage annotations
        annotation_desc_2_event_id = {}
        
        # Check for common sleep stage annotation formats
        for desc in set(raw.annotations.description):
            desc_lower = desc.lower()
            if 'w' in desc_lower or 'wake' in desc_lower:
                annotation_desc_2_event_id[desc] = 1
            elif '1' in desc or 'n1' in desc_lower:
                annotation_desc_2_event_id[desc] = 2
            elif '2' in desc or 'n2' in desc_lower:
                annotation_desc_2_event_id[desc] = 3
            elif '3' in desc or 'n3' in desc_lower or 's3' in desc_lower or 'slow wave' in desc_lower:
                annotation_desc_2_event_id[desc] = 4
            elif '4' in desc or 's4' in desc_lower:
                annotation_desc_2_event_id[desc] = 4
            elif 'r' in desc_lower or 'rem' in desc_lower:
                annotation_desc_2_event_id[desc] = 5
        
        if annotation_desc_2_event_id:
            print(f"Sleep stage mapping: {annotation_desc_2_event_id}")
            events_train, _ = mne.events_from_annotations(
                raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        else:
            print("No sleep stage annotations found. Creating default hypnogram (assumes all N2/N3).")
            # Create a default hypnogram assuming stage 2 sleep for the entire recording
            duration_sec = raw.times[-1]
            n_epochs = int(duration_sec / 30)
            events_train = np.zeros((n_epochs, 3), dtype=int)
            events_train[:, 0] = np.arange(0, n_epochs * 30 * Fs, 30 * Fs, dtype=int)
            events_train[:, 2] = 3  # Stage 2
    else:
        print("\nNo annotations found in file. Creating default hypnogram (assumes all N2/N3).")
        # Create a default hypnogram assuming stage 2 sleep for the entire recording
        duration_sec = raw.times[-1]
        n_epochs = int(duration_sec / 30)
        events_train = np.zeros((n_epochs, 3), dtype=int)
        events_train[:, 0] = np.arange(0, n_epochs * 30 * Fs, 30 * Fs, dtype=int)
        events_train[:, 2] = 3  # Stage 2

    # Set channel types if mapping matches
    for ch_name in raw.info['ch_names']:
        if ch_name in mapping:
            raw.set_channel_types({ch_name: mapping[ch_name]})

    hypno = pd.DataFrame([])
    hypno['onset'] = events_train[:, 0] / Fs
    hypno['dur'] = np.ones_like(events_train[:, 0]) * 30
    hypno['label'] = events_train[:, 2]

    print(f"\nHypnogram: {len(hypno)} epochs")
    print(f"Stage distribution: {hypno['label'].value_counts().to_dict()}")

    ## Parameters for K-complex scoring##

    # Find available EEG channels, excluding annotation channels
    eeg_channels = []
    for ch in raw.info['ch_names']:
        ch_type = raw.get_channel_types()[raw.info['ch_names'].index(ch)]
        # Exclude annotation channels and other non-EEG channels
        if ch_type == 'eeg' and 'annotation' not in ch.lower() and 'edf annotation' not in ch.lower():
            eeg_channels.append(ch)
    
    # Also check for channels starting with EEG
    if not eeg_channels:
        for ch in raw.info['ch_names']:
            if ch.upper().startswith('EEG') and 'annotation' not in ch.lower():
                eeg_channels.append(ch)
    
    if not eeg_channels:
        # If no EEG channels found, try to find channels that might be EEG
        print("Warning: No EEG channels found. Looking for other potential EEG channels...")
        # Try to find C3, C4, Fpz-Cz, or any channel containing common EEG channel names
        potential_eeg = ['C3', 'C4', 'Fpz', 'Cz', 'Fz', 'Pz', 'O1', 'O2']
        for ch in raw.info['ch_names']:
            ch_lower = ch.lower()
            # Skip annotation channels
            if 'annotation' in ch_lower:
                continue
            for p_ch in potential_eeg:
                if p_ch in ch.upper():
                    eeg_channels.append(ch)
                    break
    
    if not eeg_channels:
        # Last resort: use first non-annotation channel
        for ch in raw.info['ch_names']:
            if 'annotation' not in ch.lower():
                print(f"Warning: Using channel '{ch}' as EEG (no better option found)")
                eeg_channels = [ch]
                break
        if not eeg_channels:
            raise ValueError("No suitable EEG channels found. All channels appear to be annotations.")
    
    # Prefer C3 if available, otherwise use first EEG channel
    wanted_channel = None
    for ch in eeg_channels:
        if 'C3' in ch.upper():
            wanted_channel = ch
            break
    
    if wanted_channel is None:
        wanted_channel = eeg_channels[0]
    
    print(f"\nUsing channel: {wanted_channel}")
    
    # Extract channel data
    ch_idx = raw.info['ch_names'].index(wanted_channel)
    C3 = raw[ch_idx, :][0].ravel() * -1  # Invert signal (common for EEG)

    Fs = raw.info['sfreq']

    print(f"\nRunning K-complex detection...")
    peaks, stage_peaks, d, probas = score_KCs(C3, Fs, hypno, sleep_stages=[2, 3])

    #######################################################################
    probability_threshold = 0.5  # include only waveform scored with a probability of at least 50%

    labels = np.where(probas > 0.5, 1, 0)
    onsets = peaks[probas > 0.5]
    probas_filtered = probas[probas > 0.5]

    print(f'\n{np.sum(labels)} K-complexes were detected (probability > {probability_threshold})')
    print(f'Total peaks analyzed: {len(peaks)}')
    print(f'Probability range: [{probas.min():.3f}, {probas.max():.3f}]')
    if len(probas_filtered) > 0:
        print(f'Filtered probability range: [{probas_filtered.min():.3f}, {probas_filtered.max():.3f}]')

    ########################################################################
    ####                        VIZUALISATION                           ####

    if len(onsets) > 0:
        ##----- Average K-complexes for different probability threshold ------##
        KC_from_probas(C3 * -1, onsets, probas_filtered, Fs)

        ##------------------------Plotting with mne --------------------------##

        tmin = -2
        tmax = 2

        # peaks from score_KCs are already in sample indices, not seconds
        events_mne = np.vstack([onsets, np.zeros_like(onsets), np.ones_like(onsets)]).T
        events_mne = events_mne.astype(int)

        # Ensure we're using the correct channel name for picking
        if wanted_channel in raw.info['ch_names']:
            ep = mne.Epochs(raw, events_mne, picks=[wanted_channel], 
                          baseline=None, tmin=tmin, tmax=tmax, verbose='warning')
        else:
            ep = mne.Epochs(raw, events_mne, picks=['eeg'], 
                          baseline=None, tmin=tmin, tmax=tmax, verbose='warning')

        # Plot each individual K-complex
        print("\nPlotting individual K-complexes...")
        ep.plot(block=True)

        # Plot a similar Figure 4 in our manuscript
        print("\nPlotting K-complex image...")
        ep.plot_image(group_by=None, picks=[wanted_channel] if wanted_channel in raw.info['ch_names'] else 'eeg', 
                     vmin=-50, vmax=50)
    else:
        print("\nNo K-complexes detected above threshold. Skipping visualization.")


if __name__ == "__main__":
    main()

