import pandas as pd
import numpy as np
import mne
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings


def create_mne_raw_from_catalog(
    eeg_data: Optional[pd.DataFrame] = None,
    catalog = None,
    dataset: Optional[str] = None,
    eeg_catalog_name: Optional[str] = None,
    sfreq: float = 125.0,
    reference_channels: Optional[Union[List[int], str]] = None,
    exclude_channels: Optional[List[int]] = None,
    channel_label_col: Optional[str] = None,
    head_radius_mm: float = 95.0,
    create_montage: bool = True,
    epoch_duration_sec: float = 20.0,
    epoch_range: Optional[Union[slice, Tuple[int, int], List[int]]] = None,
    return_epochs: bool = False,
    scale_uv_to_v: bool = True, # NEW PARAMETER
    verbose: bool = False
) -> Union[mne.io.Raw, mne.Epochs]:
    
    if mne is None: raise ImportError("MNE is required")

    # --- 1. DATA LOADING STRATEGY ---
    # Determine indices to load
    samples_per_epoch = int(epoch_duration_sec * sfreq)
    start_sample, stop_sample = 0, None
    
    if epoch_range is not None and eeg_data is None:
        if isinstance(epoch_range, (slice, tuple)):
            s = epoch_range.start if isinstance(epoch_range, slice) else epoch_range[0]
            e = epoch_range.stop if isinstance(epoch_range, slice) else epoch_range[1]
            start_sample = (s or 0) * samples_per_epoch
            stop_sample = e * samples_per_epoch
        elif isinstance(epoch_range, list):
            # Load contiguous block covering min to max
            start_sample = min(epoch_range) * samples_per_epoch
            stop_sample = (max(epoch_range) + 1) * samples_per_epoch
    
    # Load Data if not provided
    if eeg_data is None:
        loader = catalog.load(eeg_catalog_name)[dataset]
        # Attempt optimized partial load
        if hasattr(loader, '__self__') and hasattr(loader.__self__, '_load') and stop_sample:
            if verbose: print(f"Partial loading samples {start_sample}:{stop_sample}")
            eeg_data = loader.__self__._load(start=start_sample, stop=stop_sample)
        else:
            if verbose: print("Full loading data (no partial loader found)")
            full_data = loader()
            if stop_sample:
                eeg_data = full_data.iloc[start_sample:stop_sample].reset_index(drop=True)
            else:
                eeg_data = full_data

    # --- 2. PREPROCESSING (Pandas Level) ---
    # Handle Exclusions (Drop columns before creating MNE object to save memory)
    if exclude_channels:
        # Convert integers to string column names if dataframe uses stringified ints
        cols_to_drop = [c for c in eeg_data.columns if int(c) in exclude_channels]
        eeg_data = eeg_data.drop(columns=cols_to_drop)

    # --- 3. METADATA SETUP ---
    # Get Channel Names
    ch_names, ch_df = _get_channel_info(
        catalog, dataset, eeg_catalog_name, eeg_data.columns, channel_label_col
    )
    
    # Create Info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Setup Montage
    if create_montage:
        montage = _create_montage(ch_names, ch_df, eeg_data.columns, head_radius_mm)
        if montage:
            info.set_montage(montage)

    # --- 4. DATA SCALING & OBJECT CREATION ---
    # Convert to standard (n_channels, n_times)
    data_values = eeg_data.values.T 
    
    # Critical: Convert uV to V
    if scale_uv_to_v:
        data_values = data_values * 1e-6
        if verbose: print("Scaled data by 1e-6 (uV -> V)")

    # --- 5. RETURN PATHS ---
    
    # Path A: Return Epochs Directly (Efficient)
    if return_epochs and epoch_range is not None:
        if verbose: print("Creating EpochsArray directly...")
        
        # Reshape data to (n_epochs, n_channels, n_times)
        n_channels, n_samples = data_values.shape
        n_epochs = n_samples // samples_per_epoch
        
        # Truncate any dangling samples that don't fit a full epoch
        data_values = data_values[:, :n_epochs * samples_per_epoch]
        
        # Reshape: (Channels, Epochs, Time) -> (Epochs, Channels, Time)
        # We reshape to (Channels, Epochs, Time) first
        data_reshaped = data_values.reshape(n_channels, n_epochs, samples_per_epoch)
        data_final = np.swapaxes(data_reshaped, 0, 1)
        
        # Handle specific epoch selection if list was provided
        if isinstance(epoch_range, list):
            # The data currently loaded is the block min(range)..max(range)
            # We need to filter for the specific indices requested
            min_idx = min(epoch_range)
            indices_relative_to_block = [i - min_idx for i in sorted(epoch_range)]
            data_final = data_final[indices_relative_to_block]
            events_idx = sorted(epoch_range)
        elif isinstance(epoch_range, slice):
            start = epoch_range.start or 0
            events_idx = list(range(start, start + n_epochs))
        else:
            # Tuple
            events_idx = list(range(epoch_range[0], epoch_range[1]))

        # Create Events Array
        events = np.zeros((len(events_idx), 3), dtype=int)
        events[:, 0] = np.arange(len(events_idx)) * samples_per_epoch # Arbitrary timing
        events[:, 2] = events_idx # Event ID = Epoch Index
        
        event_id = {f"Epoch {i}": i for i in events_idx}

        epochs = mne.EpochsArray(
            data_final, 
            info, 
            events=events, 
            event_id=event_id, 
            tmin=0.0, 
            verbose=verbose
        )
        
        # Apply Reference (MNE handles this better)
        if reference_channels:
            if reference_channels == 'average':
                epochs.set_eeg_reference('average', projection=False)
            else:
                # Map indices to names
                ref_names = [ch_names[i] for i in reference_channels if i < len(ch_names)]
                epochs.set_eeg_reference(ref_names)
                
        return epochs

    # Path B: Return Raw
    raw = mne.io.RawArray(data_values, info, verbose=verbose)
    
    # Apply Reference
    if reference_channels:
        if reference_channels == 'average':
            raw.set_eeg_reference('average', projection=False)
        else:
            # Note: This logic assumes reference_channels indices match the *current* # loaded channels. Since we dropped excluded channels, indices might shift.
            # Ideally, pass names, not indices.
            ref_names = [ch_names[i] for i in reference_channels if i < len(ch_names)]
            raw.set_eeg_reference(ref_names)

    # Note: Epoch annotations logic (sleep stages) removed for brevity, 
    # but should be added here using raw.set_annotations
    
    return raw


def _get_channel_info(
    catalog, 
    dataset: str, 
    eeg_catalog_name: str, 
    eeg_columns: pd.Index, 
    channel_label_col: Optional[str] = None
) -> Tuple[List[str], pd.DataFrame]:
    """Helper to derive channel names and load channel metadata."""
    
    # 1. Derive channel catalog name
    parts = eeg_catalog_name.split('.')
    if 'eeg' in parts:
        parts[parts.index('eeg')] = 'channels'
    elif 'brainamp' in parts:
        parts[parts.index('brainamp')] = 'channels'
    else:
        parts.append('channels')
    channel_catalog_name = '.'.join(parts)

    # 2. Load catalog
    try:
        ch_df = catalog.load(channel_catalog_name)[dataset]()
    except Exception:
        # Fallback if specific dataset entry doesn't exist, try loading root
        try:
            ch_df = catalog.load(channel_catalog_name)()
        except Exception:
            warnings.warn(f"Could not load channel info from {channel_catalog_name}. Using generic names.")
            return [f"CH{i}" for i in eeg_columns], pd.DataFrame()

    # 3. Map DataFrame columns (0-indexed) to Catalog Rows (1-indexed 'nbr')
    # Determine label column
    if not channel_label_col:
        for candidate in ['labels', 'name', 'label']:
            if candidate in ch_df.columns:
                channel_label_col = candidate
                break

    channel_names = []
    for col in eeg_columns:
        # Assuming column names are '0', '1', etc. representing indices
        try:
            ch_idx = int(col) + 1 # Convert 0-index to 1-index
            match = ch_df[ch_df['nbr'] == ch_idx]
            if not match.empty and channel_label_col:
                channel_names.append(str(match.iloc[0][channel_label_col]))
            else:
                channel_names.append(f"CH{ch_idx}")
        except ValueError:
             # If columns are already named strings
            channel_names.append(str(col))

    return channel_names, ch_df

def _create_montage(
    ch_names: List[str], 
    ch_df: pd.DataFrame, 
    eeg_columns: pd.Index, 
    head_radius_mm: float
) -> Optional[mne.channels.DigMontage]:
    """Helper to create MNE montage from coordinates."""
    if ch_df.empty or not all(c in ch_df.columns for c in ['X', 'Y', 'Z']):
        return None

    ch_pos = {}
    head_radius_m = head_radius_mm / 1000.0

    for col, name in zip(eeg_columns, ch_names):
        try:
            ch_idx = int(col) + 1
            match = ch_df[ch_df['nbr'] == ch_idx]
            if not match.empty:
                row = match.iloc[0]
                # ROTATION: Verify this rotation is actually required for your specific data!
                # Transforming to MNE Head Coordinates (RAS)
                pos = np.array([-row['Y'], row['X'], row['Z']]) * head_radius_m
                ch_pos[name] = pos
        except ValueError:
            continue
            
    if ch_pos:
        return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    return None

def _create_epoch_annotations(
    epochs_df: pd.DataFrame,
    epoch_duration_sec: float,
    sfreq: float,
    data_length_samples: int,
    epoch_offset: int = 0
) -> Optional['mne.Annotations']:
    """
    Create MNE Annotations from epochs dataframe.
    
    Args:
        epochs_df: DataFrame with epoch information (must have 'start_idx' or 'epoch_idx' column)
        epoch_duration_sec: Duration of each epoch in seconds
        sfreq: Sampling frequency in Hz
        data_length_samples: Total number of samples in the data
        epoch_offset: Offset to subtract from epoch indices (for partial data loading).
                     Default is 0 (no offset).
    
    Returns:
        MNE Annotations object or None if creation fails
    """
    if mne is None:
        return None
    
    if epochs_df.empty:
        return None
    
    # Determine epoch start indices
    if 'start_idx' in epochs_df.columns:
        epoch_starts = epochs_df['start_idx'].values
        # Adjust for partial data: subtract the offset samples
        if epoch_offset > 0:
            samples_per_epoch = int(epoch_duration_sec * sfreq)
            offset_samples = epoch_offset * samples_per_epoch
            epoch_starts = epoch_starts - offset_samples
    elif 'epoch_idx' in epochs_df.columns:
        samples_per_epoch = int(epoch_duration_sec * sfreq)
        # Adjust epoch indices for partial data
        adjusted_epoch_indices = epochs_df['epoch_idx'].values - epoch_offset
        epoch_starts = adjusted_epoch_indices * samples_per_epoch
    else:
        warnings.warn("Epochs dataframe must have 'start_idx' or 'epoch_idx' column")
        return None
    
    # Convert sample indices to time (seconds)
    # For partial data, onsets should start from 0
    onsets = epoch_starts / sfreq
    # Ensure onsets start from 0 (in case of partial loading)
    if len(onsets) > 0 and onsets[0] < 0:
        onsets = onsets - onsets[0]  # Shift to start at 0
    
    # Filter out epochs that exceed data length
    valid_mask = epoch_starts < data_length_samples
    onsets = onsets[valid_mask]
    epochs_df_valid = epochs_df[valid_mask].copy()
    
    if len(onsets) == 0:
        return None
    
    # Create durations (all epochs have the same duration)
    durations = np.full(len(onsets), epoch_duration_sec)
    
    # Create descriptions from sleep stages if available
    descriptions = []
    if 'sleep_stage' in epochs_df_valid.columns:
        # Map sleep stage numbers to labels
        stage_map = {
            -4: 'N4', -3: 'N3', -2: 'N2', -1: 'N1',
            0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N4',
            5: 'REM', -5: 'REM'
        }
        
        for stage in epochs_df_valid['sleep_stage'].values:
            stage_val = float(stage) if not pd.isna(stage) else np.nan
            if not pd.isna(stage_val):
                stage_label = stage_map.get(int(stage_val), f'Stage{int(stage_val)}')
            else:
                stage_label = 'Unknown'
            descriptions.append(f'Epoch_{stage_label}')
    else:
        # No sleep stage info, just label as epochs
        descriptions = [f'Epoch_{i}' for i in range(len(onsets))]
    
    # Create MNE Annotations
    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions
    )
    
    return annotations


def extract_epochs_from_raw(
    raw: 'mne.io.Raw',
    epoch_indices: Union[int, List[int], slice],
    epoch_duration_sec: float = 20.0,
    sfreq: Optional[float] = None,
    return_epochs: bool = False
) -> Union['mne.io.Raw', 'mne.Epochs']:
    """
    Extract one or multiple epochs from an MNE Raw object.
    
    Args:
        raw: MNE Raw object
        epoch_indices: Index or indices of epochs to extract. Can be:
                      - int: Single epoch index (0-based)
                      - List[int]: Multiple epoch indices (e.g., [60, 61, 62, 63, 64, 65])
                      - slice: Range of epochs (e.g., slice(60, 66) for epochs 60-65)
        epoch_duration_sec: Duration of each epoch in seconds. Default is 20.0 seconds.
        sfreq: Sampling frequency in Hz. If None, uses raw.info['sfreq']
        return_epochs: If True and multiple epochs requested, returns MNE Epochs object.
                      If False, returns concatenated Raw object. Default is False.
    
    Returns:
        - If single epoch or return_epochs=False: MNE Raw object containing requested epochs
        - If multiple epochs and return_epochs=True: MNE Epochs object
    
    Example:
        >>> raw = create_mne_raw_from_catalog(eeg_df, catalog, dataset, "id.episl.eeg", sfreq=125)
        >>> 
        >>> # Extract single epoch
        >>> epoch_5 = extract_epochs_from_raw(raw, epoch_idx=5, epoch_duration_sec=20.0)
        >>> 
        >>> # Extract multiple epochs (60-65)
        >>> epochs_60_65 = extract_epochs_from_raw(raw, epoch_indices=[60, 61, 62, 63, 64, 65])
        >>> 
        >>> # Extract using slice
        >>> epochs_range = extract_epochs_from_raw(raw, epoch_indices=slice(60, 66))
        >>> 
        >>> # Return as MNE Epochs object (better for analysis)
        >>> epochs_obj = extract_epochs_from_raw(raw, epoch_indices=slice(60, 66), return_epochs=True)
    """
    if mne is None:
        raise ImportError("MNE is required for this function. Install with: pip install mne")
    
    if sfreq is None:
        sfreq = raw.info['sfreq']
    
    # Convert epoch_indices to list
    if isinstance(epoch_indices, int):
        epoch_indices = [epoch_indices]
    elif isinstance(epoch_indices, slice):
        epoch_indices = list(range(epoch_indices.start, epoch_indices.stop, epoch_indices.step or 1))
    elif not isinstance(epoch_indices, (list, np.ndarray)):
        raise TypeError(f"epoch_indices must be int, list, or slice, got {type(epoch_indices)}")
    
    if len(epoch_indices) == 0:
        raise ValueError("No epochs specified")
    
    # Calculate epoch boundaries
    data_duration = raw.times[-1] + (1.0 / sfreq)
    epoch_starts = [idx * epoch_duration_sec for idx in epoch_indices]
    epoch_ends = [(idx + 1) * epoch_duration_sec for idx in epoch_indices]
    
    # Validate epochs
    valid_epochs = []
    valid_indices = []
    for i, (start, end, idx) in enumerate(zip(epoch_starts, epoch_ends, epoch_indices)):
        if start >= data_duration:
            warnings.warn(f"Epoch {idx} starts at {start:.2f}s, but data only has {data_duration:.2f}s. Skipping.")
            continue
        valid_epochs.append((start, min(end, data_duration)))
        valid_indices.append(idx)
    
    if len(valid_epochs) == 0:
        raise ValueError("No valid epochs found")
    
    # If single epoch and return_epochs=False, return Raw object
    if len(valid_epochs) == 1 and not return_epochs:
        start, end = valid_epochs[0]
        return raw.copy().crop(tmin=start, tmax=end)
    
    # Multiple epochs: create Epochs object or concatenate Raw objects
    if return_epochs:
        # Create events array for MNE Epochs
        events = []
        for (start, end), idx in zip(valid_epochs, valid_indices):
            sample = int(start * sfreq)
            events.append([sample, 0, idx + 1])  # event_id = epoch_idx + 1
        
        events = np.array(events)
        
        # Create Epochs object
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id={f'epoch_{idx}': idx + 1 for idx in valid_indices},
            tmin=0.0,
            tmax=epoch_duration_sec,
            baseline=None,
            preload=True,
            verbose=False
        )
        return epochs
    else:
        # Concatenate multiple Raw objects
        raw_epochs = []
        for start, end in valid_epochs:
            epoch_raw = raw.copy().crop(tmin=start, tmax=end)
            raw_epochs.append(epoch_raw)
        
        # Concatenate all epochs
        concatenated = mne.concatenate_raws(raw_epochs, preload=True)
        return concatenated


# Backward compatibility alias
def extract_epoch_from_raw(
    raw: 'mne.io.Raw',
    epoch_idx: int,
    epoch_duration_sec: float = 20.0,
    sfreq: Optional[float] = None
) -> 'mne.io.Raw':
    """
    Extract a single epoch from an MNE Raw object.
    
    This is a convenience wrapper around extract_epochs_from_raw for backward compatibility.
    
    Args:
        raw: MNE Raw object
        epoch_idx: Index of the epoch to extract (0-based)
        epoch_duration_sec: Duration of each epoch in seconds. Default is 20.0 seconds.
        sfreq: Sampling frequency in Hz. If None, uses raw.info['sfreq']
    
    Returns:
        MNE Raw object containing only the requested epoch
    
    Example:
        >>> raw = create_mne_raw_from_catalog(eeg_df, catalog, dataset, "id.episl.eeg", sfreq=125)
        >>> epoch_5 = extract_epoch_from_raw(raw, epoch_idx=5, epoch_duration_sec=20.0)
        >>> # Extract data as numpy array
        >>> epoch_data, times = epoch_5[:, :]
    """
    return extract_epochs_from_raw(raw, epoch_idx, epoch_duration_sec, sfreq, return_epochs=False)


def extract_epochs_by_sleep_stage(
    raw: 'mne.io.Raw',
    sleep_stage: Union[int, List[int], str, List[str]],
    epoch_indices: Optional[Union[int, List[int], slice]] = None,
    epoch_duration_sec: float = 20.0,
    return_epochs: bool = False,
    return_hypno: bool = True,
    verbose: bool = False
) -> Union[
    Union['mne.io.Raw', 'mne.Epochs'],
    Tuple[Union['mne.io.Raw', 'mne.Epochs'], pd.DataFrame]
]:
    """
    Extract epochs filtered by sleep stage directly from MNE Raw object annotations.
    
    This function extracts epochs from the Raw object's annotations, which should have been
    added by create_mne_raw_from_catalog() with add_epoch_annotations=True.
    
    Args:
        raw: MNE Raw object with epoch annotations (created by create_mne_raw_from_catalog)
        sleep_stage: Sleep stage(s) to filter. Can be:
                    - int: Sleep stage number (e.g., -2 for N2, -3 for N3)
                    - str: Sleep stage label (e.g., 'N2', 'N3', 'N2' or 'N3')
                    - List: Multiple sleep stages (e.g., [-2, -3] or ['N2', 'N3'])
        epoch_indices: Optional filter for specific epoch indices. Can be:
                      - int: Single epoch index (0-based)
                      - List[int]: Multiple epoch indices
                      - slice: Range of epochs (e.g., slice(60, 66))
                      - None: Use all epochs matching sleep stage
        epoch_duration_sec: Duration of each epoch in seconds. Default is 20.0 seconds.
        return_epochs: If True, returns MNE Epochs object. If False, returns concatenated Raw object.
        return_hypno: If True, also returns a hypnogram DataFrame compatible with score_KCs function.
                     Default is True.
        verbose: Whether to print verbose output
    
    Returns:
        If return_hypno=False: MNE Raw or Epochs object containing filtered epochs
        If return_hypno=True: Tuple of (epochs_object, hypno_dataframe)
                              where hypno_dataframe has columns: 'onset', 'dur', 'label'
                              compatible with score_KCs function
    
    Example:
        >>> raw = create_mne_raw_from_catalog(eeg_df, catalog, dataset, "id.episl.eeg", sfreq=125)
        >>> 
        >>> # Extract N2 epochs 60-65 (without hypnogram)
        >>> n2_epochs_60_65 = extract_epochs_by_sleep_stage(
        ...     raw, sleep_stage=-2, 
        ...     epoch_indices=slice(60, 66),
        ...     return_hypno=False
        ... )
        >>> 
        >>> # Extract all N2/N3 epochs with hypnogram (for score_KCs)
        >>> n2_n3_epochs, hypno = extract_epochs_by_sleep_stage(
        ...     raw, sleep_stage=[-2, -3],
        ...     return_hypno=True
        ... )
        >>> 
        >>> # Use with score_KCs function
        >>> from clas.helpers.external_libs.K_complex_adelaide.KC_algorithm.model import score_KCs
        >>> C3_data = n2_n3_epochs.get_data(picks='C3').squeeze(axis=1)
        >>> peaks, stage_peaks, d, probas = score_KCs(
        ...     C3_data.flatten(), 
        ...     Fs=raw.info['sfreq'], 
        ...     Stages=hypno, 
        ...     sleep_stages=[3, 4]  # N2=3, N3=4 in score_KCs format
        ... )
        >>> 
        >>> # Extract using string labels
        >>> n2_epochs, hypno = extract_epochs_by_sleep_stage(
        ...     raw, sleep_stage='N2', return_hypno=True
        ... )
    """
    if mne is None:
        raise ImportError("MNE is required for this function. Install with: pip install mne")
    
    # Check if Raw object has annotations
    if len(raw.annotations) == 0:
        raise ValueError(
            "Raw object has no annotations. Make sure to create Raw object with "
            "create_mne_raw_from_catalog(..., add_epoch_annotations=True)"
        )
    
    # Normalize sleep_stage to list of strings
    stage_map = {
        -4: 'N4', -3: 'N3', -2: 'N2', -1: 'N1',
        0: 'REM', 1: 'WAKE', 2: 'N2', 3: 'N3', 4: 'N4',
        5: 'REM', -5: 'REM'
    }
    
    if isinstance(sleep_stage, (int, float)):
        # Convert numeric stage to string label
        sleep_stage_labels = [stage_map.get(int(sleep_stage), f'Stage{int(sleep_stage)}')]
    elif isinstance(sleep_stage, str):
        sleep_stage_labels = [sleep_stage]
    elif isinstance(sleep_stage, list):
        sleep_stage_labels = []
        for stage in sleep_stage:
            if isinstance(stage, (int, float)):
                sleep_stage_labels.append(stage_map.get(int(stage), f'Stage{int(stage)}'))
            else:
                sleep_stage_labels.append(str(stage))
    else:
        raise TypeError(f"sleep_stage must be int, str, or list, got {type(sleep_stage)}")
    
    # Get sampling frequency
    sfreq = raw.info['sfreq']
    
    # Parse annotations to extract epoch information
    annotations = raw.annotations
    epoch_info = []
    
    for i, (onset, duration, desc) in enumerate(zip(annotations.onset, annotations.duration, annotations.description)):
        # Parse sleep stage from description (format: "Epoch_N2", "Epoch_N3", etc.)
        if desc.startswith('Epoch_'):
            stage_label = desc.replace('Epoch_', '')
        else:
            # Try to extract stage from description
            stage_label = desc.split('_')[-1] if '_' in desc else desc
        
        # Calculate epoch index from onset time
        epoch_idx = int(onset / epoch_duration_sec)
        
        epoch_info.append({
            'epoch_idx': epoch_idx,
            'onset': onset,
            'duration': duration,
            'stage_label': stage_label,
            'annotation_idx': i
        })
    
    # FIRST: Filter by sleep stage
    sleep_stage_filtered = [ep for ep in epoch_info if ep['stage_label'] in sleep_stage_labels]
    
    if len(sleep_stage_filtered) == 0:
        available_stages = set(ep['stage_label'] for ep in epoch_info)
        raise ValueError(
            f"No epochs found with sleep stage(s) {sleep_stage_labels}. "
            f"Available stages: {sorted(available_stages)}"
        )
    
    # SECOND: Further filter by epoch_indices if specified (only within the sleep stage filtered epochs)
    if epoch_indices is not None:
        if isinstance(epoch_indices, int):
            epoch_indices_list = [epoch_indices]
        elif isinstance(epoch_indices, slice):
            epoch_indices_list = list(range(epoch_indices.start, epoch_indices.stop, epoch_indices.step or 1))
        else:
            epoch_indices_list = list(epoch_indices)
        
        # Filter: keep epochs that match BOTH sleep stage AND are in the requested epoch indices
        # This filters within the already sleep-stage-filtered epochs
        filtered_epochs = [ep for ep in sleep_stage_filtered if ep['epoch_idx'] in epoch_indices_list]
        
        if len(filtered_epochs) == 0:
            # Get available epoch indices for the requested sleep stages
            available_indices = sorted(set(ep['epoch_idx'] for ep in sleep_stage_filtered))
            
            # Show a helpful error message
            if len(available_indices) > 0:
                indices_preview = available_indices[:20]
                preview_str = f"{indices_preview}{'...' if len(available_indices) > 20 else ''}"
                raise ValueError(
                    f"No epochs found with sleep stage(s) {sleep_stage_labels} and indices {epoch_indices_list}. "
                    f"\nAvailable epoch indices for sleep stage(s) {sleep_stage_labels}: {preview_str} "
                    f"(total: {len(available_indices)} epochs)"
                )
            else:
                raise ValueError(
                    f"No epochs found with sleep stage(s) {sleep_stage_labels} and indices {epoch_indices_list}. "
                    f"No epochs with sleep stage(s) {sleep_stage_labels} exist in the data."
                )
    else:
        # No epoch_indices filter, use all sleep-stage-filtered epochs
        filtered_epochs = sleep_stage_filtered
    
    # Get epoch indices
    epoch_idx_list = [ep['epoch_idx'] for ep in filtered_epochs]
    
    if verbose:
        stage_counts = {}
        for ep in filtered_epochs:
            stage_counts[ep['stage_label']] = stage_counts.get(ep['stage_label'], 0) + 1
        
        print(f"Found {len(epoch_idx_list)} epochs matching criteria:")
        print(f"  Sleep stage(s): {sleep_stage_labels}")
        if epoch_indices is not None:
            print(f"  Epoch indices: {epoch_idx_list}")
        print(f"  Sleep stage distribution: {stage_counts}")
    
    # Extract epochs
    epochs_obj = extract_epochs_from_raw(
        raw, 
        epoch_indices=epoch_idx_list,
        epoch_duration_sec=epoch_duration_sec,
        return_epochs=return_epochs
    )
    
    # Create hypnogram DataFrame compatible with score_KCs if requested
    if return_hypno:
        # Simple mapping from current labels to numeric values (keep current system)
        label_to_numeric = {
            'Wake': 1, 'WAKE': 1, 'W': 1,
            'N1': -1,
            'N2': -2,
            'N3': -3, 'N4': -3,
            'REM': 0, 'R': 0
        }
        
        # Create hypnogram DataFrame
        hypno_list = []
        for ep in filtered_epochs:
            stage_label = ep['stage_label']
            # Use label directly if it's already numeric, otherwise map it
            if isinstance(stage_label, (int, float)):
                numeric_label = int(stage_label)
            else:
                numeric_label = label_to_numeric.get(stage_label, 1)  # Default to Wake if unknown
            
            hypno_list.append({
                'onset': ep['onset'],
                'dur': ep['duration'],
                'label': numeric_label
            })
        
        hypno = pd.DataFrame(hypno_list)
        
        if verbose:
            print(f"\nCreated hypnogram DataFrame with {len(hypno)} epochs")
            print(f"  Sleep stage distribution: {hypno['label'].value_counts().to_dict()}")
        
        return epochs_obj, hypno
    else:
        return epochs_obj


def create_hypnogram_from_catalog(
    catalog = None,
    dataset: Optional[str] = None,
    eeg_catalog_name: Optional[str] = None,
    epoch_duration_sec: float = 20.0,
    sfreq: float = 125.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Create a hypnogram DataFrame from epochs catalog with absolute epoch indices.
    
    This function loads the epochs catalog and creates a hypnogram DataFrame that contains
    absolute epoch indices relative to the whole dataset. This allows you to select epochs
    based on criteria (e.g., "5th to 10th N2 epoch") and then load the corresponding data.
    
    Args:
        catalog: Kedro catalog object for loading epochs data
        dataset: Dataset partition name (e.g., 'EPISL_02_W1')
        eeg_catalog_name: Name of the EEG catalog entry (e.g., 'id.episl.eeg')
                        Used to derive epochs catalog name
        epoch_duration_sec: Duration of each epoch in seconds. Default is 20.0 seconds.
        sfreq: Sampling frequency in Hz. Used to calculate epoch start times.
        verbose: Whether to print verbose output
    
    Returns:
        DataFrame with columns:
        - 'epoch_idx': Absolute epoch index (0-indexed, relative to whole dataset)
        - 'sleep_stage': Sleep stage label (numeric: -3=N3, -2=N2, -1=N1, 0=REM, 1=WAKE)
        - 'start_idx': Start sample index in the whole dataset (at sfreq)
        - 'onset': Onset time in seconds (relative to dataset start)
        - 'dur': Duration in seconds (typically epoch_duration_sec)
        - 'label': Sleep stage label for score_KCs compatibility (same as sleep_stage)
    
    Example:
        >>> hypno = create_hypnogram_from_catalog(
        ...     catalog=catalog,
        ...     dataset="EPISL_02_W1",
        ...     eeg_catalog_name="id.episl.eeg",
        ...     epoch_duration_sec=20.0,
        ...     sfreq=125.0
        ... )
        >>> # Select 5th to 10th N2 epochs
        >>> n2_epochs = hypno[hypno['sleep_stage'] == -2]
        >>> selected_indices = n2_epochs.iloc[5:10]['epoch_idx'].tolist()
    """
    if catalog is None or dataset is None or eeg_catalog_name is None:
        raise ValueError("catalog, dataset, and eeg_catalog_name are required")
    
    # Derive epochs catalog name from EEG catalog name
    epochs_catalog_name = f"{eeg_catalog_name}.epochs"
    
    try:
        epochs_df = catalog.load(epochs_catalog_name)[dataset]()
        
        if verbose:
            print(f"Loaded epochs data with {len(epochs_df)} epochs")
        
        # Create hypnogram DataFrame with absolute epoch indices
        hypno_list = []
        
        # Determine which column contains epoch index
        if 'epoch_idx' in epochs_df.columns:
            epoch_idx_col = 'epoch_idx'
        elif 'start_idx' in epochs_df.columns:
            # Calculate epoch_idx from start_idx
            samples_per_epoch = int(epoch_duration_sec * sfreq)
            epochs_df = epochs_df.copy()
            epochs_df['epoch_idx'] = (epochs_df['start_idx'] / samples_per_epoch).astype(int)
            epoch_idx_col = 'epoch_idx'
        else:
            raise ValueError("Epochs DataFrame must have 'epoch_idx' or 'start_idx' column")
        
        # Determine which column contains sleep stage
        if 'sleep_stage' in epochs_df.columns:
            stage_col = 'sleep_stage'
        else:
            raise ValueError("Epochs DataFrame must have 'sleep_stage' column")
        
        # Create hypnogram entries
        for _, row in epochs_df.iterrows():
            epoch_idx = int(row[epoch_idx_col])
            sleep_stage = row[stage_col]
            
            # Calculate start_idx if not present
            if 'start_idx' in epochs_df.columns:
                start_idx = int(row['start_idx'])
            else:
                samples_per_epoch = int(epoch_duration_sec * sfreq)
                start_idx = epoch_idx * samples_per_epoch
            
            # Calculate onset time
            onset = start_idx / sfreq
            
            hypno_list.append({
                'epoch_idx': epoch_idx,
                'sleep_stage': sleep_stage,
                'start_idx': start_idx,
                'onset': onset,
                'dur': epoch_duration_sec,
                'label': sleep_stage  # For score_KCs compatibility
            })
        
        hypno = pd.DataFrame(hypno_list).sort_values('epoch_idx').reset_index(drop=True)
        
        if verbose:
            stage_counts = hypno['sleep_stage'].value_counts().to_dict()
            print(f"Created hypnogram with {len(hypno)} epochs")
            print(f"  Sleep stage distribution: {stage_counts}")
            print(f"  Epoch range: {hypno['epoch_idx'].min()} - {hypno['epoch_idx'].max()}")
        
        return hypno
        
    except Exception as e:
        raise ValueError(
            f"Could not load epochs data from '{epochs_catalog_name}' for dataset '{dataset}'. "
            f"Original error: {e}"
        ) from e


def select_epochs_from_hypnogram(
    hypno: pd.DataFrame,
    sleep_stage: Optional[Union[int, List[int], str, List[str]]] = None,
    epoch_selection: Optional[Union[int, slice, List[int], str]] = None,
    verbose: bool = False
) -> Tuple[List[int], pd.DataFrame]:
    """
    Select epochs from hypnogram based on sleep stage and selection criteria.
    
    This function provides flexible epoch selection from a hypnogram:
    - Select by sleep stage (e.g., all N2 epochs)
    - Select by position within sleep stage (e.g., "5th to 10th N2 epoch")
    - Select consecutive epochs (e.g., "5 consecutive N2 epochs")
    
    Args:
        hypno: Hypnogram DataFrame from create_hypnogram_from_catalog()
        sleep_stage: Sleep stage(s) to filter. Can be:
                    - int: Sleep stage number (e.g., -2 for N2, -3 for N3)
                    - str: Sleep stage label (e.g., 'N2', 'N3')
                    - List: Multiple sleep stages (e.g., [-2, -3] or ['N2', 'N3'])
                    - None: Use all epochs (no filtering)
        epoch_selection: Selection criteria within the filtered epochs. Can be:
                        - int: Nth epoch (0-indexed, e.g., 5 for 6th epoch)
                        - slice: Range of epochs (e.g., slice(5, 10) for epochs 5-9)
                        - List[int]: Specific epoch indices (e.g., [0, 2, 5])
                        - str: Special selection modes:
                               - "first": First epoch
                               - "last": Last epoch
                               - "consecutive:N": N consecutive epochs starting from first
                        - None: Select all epochs matching sleep_stage
        verbose: Whether to print verbose output
    
    Returns:
        Tuple of:
        - selected_epoch_indices: List of absolute epoch indices (relative to whole dataset)
        - selected_hypno: DataFrame with selected epochs
    
    Examples:
        >>> # Select all N2 epochs
        >>> indices, hypno_n2 = select_epochs_from_hypnogram(hypno, sleep_stage=-2)
        
        >>> # Select 5th to 10th N2 epochs
        >>> indices, hypno_n2_5_10 = select_epochs_from_hypnogram(
        ...     hypno, sleep_stage=-2, epoch_selection=slice(5, 10)
        ... )
        
        >>> # Select 5 consecutive N2 epochs
        >>> indices, hypno_n2_5consec = select_epochs_from_hypnogram(
        ...     hypno, sleep_stage=-2, epoch_selection="consecutive:5"
        ... )
        
        >>> # Select all N2 and N3 epochs
        >>> indices, hypno_n2_n3 = select_epochs_from_hypnogram(
        ...     hypno, sleep_stage=[-2, -3]
        ... )
    """
    # Filter by sleep stage if specified
    if sleep_stage is not None:
        # Normalize sleep_stage to list
        stage_map = {
            -4: -4, -3: -3, -2: -2, -1: -1,
            0: 0, 1: 1, 2: -2, 3: -3, 4: -4,  # Map alternative encodings
            'N4': -4, 'N3': -3, 'N2': -2, 'N1': -1,
            'REM': 0, 'WAKE': 1, 'Wake': 1, 'W': 1
        }
        
        if isinstance(sleep_stage, (int, float)):
            sleep_stages = [int(sleep_stage)]
        elif isinstance(sleep_stage, str):
            sleep_stages = [stage_map.get(sleep_stage, sleep_stage)]
        elif isinstance(sleep_stage, list):
            sleep_stages = []
            for stage in sleep_stage:
                if isinstance(stage, (int, float)):
                    sleep_stages.append(int(stage))
                else:
                    sleep_stages.append(stage_map.get(stage, stage))
        else:
            raise TypeError(f"sleep_stage must be int, str, or list, got {type(sleep_stage)}")
        
        # Filter hypnogram
        filtered_hypno = hypno[hypno['sleep_stage'].isin(sleep_stages)].copy().reset_index(drop=True)
        
        if len(filtered_hypno) == 0:
            available_stages = sorted(hypno['sleep_stage'].unique())
            raise ValueError(
                f"No epochs found with sleep stage(s) {sleep_stages}. "
                f"Available stages: {available_stages}"
            )
    else:
        filtered_hypno = hypno.copy()
    
    if verbose:
        stage_counts = filtered_hypno['sleep_stage'].value_counts().to_dict()
        print(f"Filtered to {len(filtered_hypno)} epochs")
        if sleep_stage is not None:
            print(f"  Sleep stage(s): {sleep_stages}")
        print(f"  Sleep stage distribution: {stage_counts}")
    
    # Apply epoch_selection if specified
    if epoch_selection is not None:
        n_filtered = len(filtered_hypno)
        
        if isinstance(epoch_selection, str):
            # Special selection modes
            if epoch_selection == "first":
                selected_indices_in_filtered = [0]
            elif epoch_selection == "last":
                selected_indices_in_filtered = [n_filtered - 1]
            elif epoch_selection.startswith("consecutive:"):
                # Extract number of consecutive epochs
                try:
                    n_consecutive = int(epoch_selection.split(":")[1])
                    selected_indices_in_filtered = list(range(min(n_consecutive, n_filtered)))
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid consecutive format: {epoch_selection}. Use 'consecutive:N'")
            else:
                raise ValueError(f"Unknown epoch_selection string: {epoch_selection}")
        elif isinstance(epoch_selection, int):
            selected_indices_in_filtered = [epoch_selection]
        elif isinstance(epoch_selection, slice):
            selected_indices_in_filtered = list(range(*epoch_selection.indices(n_filtered)))
        elif isinstance(epoch_selection, list):
            selected_indices_in_filtered = list(epoch_selection)
        else:
            raise TypeError(f"epoch_selection must be int, slice, list, str, or None, got {type(epoch_selection)}")
        
        # Validate indices
        if any(idx >= n_filtered or idx < 0 for idx in selected_indices_in_filtered):
            raise ValueError(f"Some epoch_selection indices out of range. Available: 0-{n_filtered-1}")
        
        # Select epochs
        selected_hypno = filtered_hypno.iloc[selected_indices_in_filtered].copy().reset_index(drop=True)
    else:
        selected_hypno = filtered_hypno.copy()
    
    # Get absolute epoch indices
    selected_epoch_indices = selected_hypno['epoch_idx'].tolist()
    
    if verbose:
        print(f"Selected {len(selected_epoch_indices)} epochs")
        print(f"  Absolute epoch indices: {selected_epoch_indices[:10]}{'...' if len(selected_epoch_indices) > 10 else ''}")
    
    return selected_epoch_indices, selected_hypno


def add_events_to_raw(
    raw: 'mne.io.Raw',
    events_dict: Dict[str, Union[pd.DataFrame, Dict[str, Any]]],
    sfreq: Optional[float] = None,
    sample_column: Optional[str] = None,
    channel_column: Optional[str] = None,
    event_type: str = 'discrete',
    verbose: bool = False
) -> 'mne.io.Raw':
    """
    Add multiple types of events to an MNE Raw object.
    
    This function provides a flexible interface for adding various event types (KC labels,
    SW events, SW bursts, etc.) to an MNE Raw object. It handles both discrete point events
    (using MNE's events array) and continuous annotations (using MNE's Annotations).
    
    Args:
        raw: MNE Raw object to add events to
        events_dict: Dictionary mapping event type names to event data.
                    Each value can be:
                    - DataFrame with event information
                    - Dict with 'data' (DataFrame), 'event_type' ('discrete' or 'continuous'),
                      'sample_column', 'channel_column', etc.
        sfreq: Sampling frequency in Hz. If None, uses raw.info['sfreq']
        sample_column: Default column name for sample indices (e.g., 'ndx_trough', 'peak_idx').
                      Can be overridden per event type in events_dict.
        channel_column: Default column name for channel information (e.g., 'channel').
                       Can be overridden per event type in events_dict.
        event_type: Default event type ('discrete' for point events, 'continuous' for periods).
                   Can be overridden per event type in events_dict.
        verbose: Whether to print verbose output
    
    Returns:
        MNE Raw object with events added
    
    Example:
        >>> # Add KC events (discrete point events)
        >>> kc_df = pd.DataFrame({
        ...     'peak_idx': [1000, 2000, 3000],
        ...     'channel': [0, 1, 0],
        ...     'kc_probability': [0.9, 0.85, 0.92]
        ... })
        >>> 
        >>> # Add SW events (discrete: troughs, peaks; continuous: wave periods)
        >>> sw_df = pd.DataFrame({
        ...     'ndx_trough': [500, 1500],
        ...     'wave_start': [400, 1400],
        ...     'wave_stop': [600, 1600],
        ...     'channel': [0, 1]
        ... })
        >>> 
        >>> # Add SW bursts (continuous periods)
        >>> sw_burst_df = pd.DataFrame({
        ...     'burst_start': [1000, 5000],
        ...     'burst_stop': [2000, 6000],
        ...     'burst_count': [5, 8],
        ...     'channel': [0, 1]
        ... })
        >>> 
        >>> events_dict = {
        ...     'KC': {
        ...         'data': kc_df,
        ...         'event_type': 'discrete',
        ...         'sample_column': 'peak_idx',
        ...         'channel_column': 'channel'
        ...     },
        ...     'SW_trough': {
        ...         'data': sw_df,
        ...         'event_type': 'discrete',
        ...         'sample_column': 'ndx_trough',
        ...         'channel_column': 'channel'
        ...     },
        ...     'SW_wave': {
        ...         'data': sw_df,
        ...         'event_type': 'continuous',
        ...         'start_column': 'wave_start',
        ...         'stop_column': 'wave_stop',
        ...         'channel_column': 'channel'
        ...     },
        ...     'SW_burst': {
        ...         'data': sw_burst_df,
        ...         'event_type': 'continuous',
        ...         'start_column': 'burst_start',
        ...         'stop_column': 'burst_stop',
        ...         'channel_column': 'channel'
        ...     }
        ... }
        >>> 
        >>> raw_with_events = add_events_to_raw(raw, events_dict, sfreq=125)
    """
    if mne is None:
        raise ImportError("MNE is required for this function. Install with: pip install mne")
    
    if sfreq is None:
        sfreq = raw.info['sfreq']
    
    # Collect all discrete events and continuous annotations
    all_events = []
    all_annotations = []
    
    # Event ID counter (MNE uses integers > 0 for event IDs)
    event_id_counter = 1
    event_id_map = {}
    
    for event_name, event_data in events_dict.items():
        # Handle both DataFrame and Dict formats
        if isinstance(event_data, pd.DataFrame):
            df = event_data
            evt_type = event_type
            samp_col = sample_column
            ch_col = channel_column
            start_col = None
            stop_col = None
        elif isinstance(event_data, dict):
            df = event_data.get('data')
            if df is None:
                warnings.warn(f"Event '{event_name}' has no 'data' key. Skipping.")
                continue
            evt_type = event_data.get('event_type', event_type)
            samp_col = event_data.get('sample_column', sample_column)
            ch_col = event_data.get('channel_column', channel_column)
            start_col = event_data.get('start_column', None)
            stop_col = event_data.get('stop_column', None)
        else:
            warnings.warn(f"Event '{event_name}' has unsupported format. Skipping.")
            continue
        
        if df.empty:
            if verbose:
                print(f"Skipping empty event type: {event_name}")
            continue
        
        if evt_type == 'discrete':
            # Create discrete point events
            if samp_col is None or samp_col not in df.columns:
                warnings.warn(f"Event '{event_name}': sample column '{samp_col}' not found. Skipping.")
                continue
            
            # Get sample indices
            samples = df[samp_col].values.astype(int)
            
            # Filter valid samples
            max_sample = len(raw.times) * sfreq
            valid_mask = (samples >= 0) & (samples < max_sample)
            samples = samples[valid_mask]
            df_valid = df[valid_mask].copy()
            
            if len(samples) == 0:
                if verbose:
                    print(f"No valid samples for event '{event_name}'. Skipping.")
                continue
            
            # Create event ID for this event type
            if event_name not in event_id_map:
                event_id_map[event_name] = event_id_counter
                event_id_counter += 1
            
            event_id = event_id_map[event_name]
            
            # Create events array: [sample, 0, event_id]
            events = np.zeros((len(samples), 3), dtype=int)
            events[:, 0] = samples
            events[:, 2] = event_id
            
            # Add channel information to description if available
            if ch_col and ch_col in df_valid.columns:
                descriptions = [f"{event_name}_ch{int(ch)}" for ch in df_valid[ch_col].values]
            else:
                descriptions = [event_name] * len(samples)
            
            all_events.append(events)
            
            if verbose:
                print(f"Added {len(samples)} discrete '{event_name}' events (event_id={event_id})")
        
        elif evt_type == 'continuous':
            # Create continuous annotations
            if start_col is None or start_col not in df.columns:
                warnings.warn(f"Event '{event_name}': start column '{start_col}' not found. Skipping.")
                continue
            
            if stop_col is None or stop_col not in df.columns:
                # Try to infer from duration or use single sample
                if 'duration' in df.columns:
                    df = df.copy()
                    df[stop_col] = df[start_col] + df['duration']
                else:
                    warnings.warn(f"Event '{event_name}': stop column not found and no duration. Skipping.")
                    continue
            
            # Get start and stop indices
            starts = df[start_col].values.astype(int)
            stops = df[stop_col].values.astype(int)
            
            # Convert to time (seconds)
            onsets = starts / sfreq
            durations = (stops - starts) / sfreq
            
            # Filter valid annotations
            max_time = raw.times[-1]
            valid_mask = (onsets >= 0) & (onsets < max_time) & (durations > 0)
            onsets = onsets[valid_mask]
            durations = durations[valid_mask]
            df_valid = df[valid_mask].copy()
            
            if len(onsets) == 0:
                if verbose:
                    print(f"No valid annotations for event '{event_name}'. Skipping.")
                continue
            
            # Create descriptions
            if ch_col and ch_col in df_valid.columns:
                descriptions = []
                for idx, row in df_valid.iterrows():
                    ch = int(row[ch_col])
                    desc = f"{event_name}_ch{ch}"
                    # Add additional info if available (e.g., burst count)
                    if 'burst_count' in df_valid.columns:
                        desc += f"_n{int(row['burst_count'])}"
                    descriptions.append(desc)
            else:
                descriptions = [event_name] * len(onsets)
            
            # Create annotations
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            all_annotations.append(annotations)
            
            if verbose:
                print(f"Added {len(onsets)} continuous '{event_name}' annotations")
        
        else:
            warnings.warn(f"Unknown event_type '{evt_type}' for '{event_name}'. Use 'discrete' or 'continuous'.")
    
    # Combine and add events
    if all_events:
        combined_events = np.vstack(all_events)
        # Sort by sample index
        sort_idx = np.argsort(combined_events[:, 0])
        combined_events = combined_events[sort_idx]
        raw.add_events(combined_events, sfreq=sfreq)
        
        if verbose:
            print(f"Added {len(combined_events)} total discrete events")
    
    # Combine and add annotations
    if all_annotations:
        # Combine annotations
        combined_annotations = all_annotations[0]
        for ann in all_annotations[1:]:
            combined_annotations = combined_annotations + ann
        
        # Merge with existing annotations if any
        if len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + combined_annotations)
        else:
            raw.set_annotations(combined_annotations)
        
        if verbose:
            print(f"Added {len(combined_annotations)} total continuous annotations")
    
    return raw