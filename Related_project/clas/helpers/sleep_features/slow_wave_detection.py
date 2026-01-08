import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from typing import Union, List, Dict, Any
from joblib import Parallel, delayed

def slowwave_detection(
    EegData: np.ndarray,
    SampRate: Union[int, float],
    WaveStart: str = 'NegZeroCrossings',
    CleanEpochs: Union[List[int], np.ndarray] = None,
    EpochLength: Union[int, float] = 20,
    QualityCheck: str = 'No',
    min_freq: float = 0.5,  # Minimum slow-wave frequency (Hz); corresponds to a max period of 1/min_freq seconds.
    max_freq: float = 4.0   # Maximum slow-wave frequency (Hz); corresponds to a min period of 1/max_freq seconds.
) -> pd.DataFrame:
    """
    Detect and characterize slow waves in EEG data.

    This refactored version processes each channel in parallel and returns a 
    pandas DataFrame with one row per detected wave (including channel information).
    Only waves with a duration (wave_stop - wave_start) between 1/max_freq and 1/min_freq
    seconds are retained (e.g., between 0.25 and 2 seconds by default).

    Parameters
    ----------
    EegData : np.ndarray
        EEG data with shape (channels, data_points). It is recommended to use 
        bandpass filtered data (e.g. 0.5 to 4 Hz).
    SampRate : int or float
        Sampling rate of the data.
    WaveStart : str, optional
        Method for defining the start of slow waves. One of:
        'Peaks', 'Troughs', 'PosZeroCrossings', or 'NegZeroCrossings'
        (default is 'NegZeroCrossings').
    CleanEpochs : list of int or np.ndarray, optional
        Indices (epoch numbers, starting at 1) of epochs considered to be clean 
        NREM sleep periods. Default is None (no cleaning applied).
    EpochLength : int or float, optional
        Length of each epoch in seconds (default is 30).
    QualityCheck : str, optional
        If 'Yes', a quality-check figure is produced for each channel.
        (default is 'No').
    min_freq : float, optional
        Minimum frequency (Hz) of slow waves (default is 0.5 Hz). Waves with a period 
        longer than 1/min_freq seconds (e.g., 2 s) will be discarded.
    max_freq : float, optional
        Maximum frequency (Hz) of slow waves (default is 4.0 Hz). Waves with a period 
        shorter than 1/max_freq seconds (e.g., 0.25 s) will be discarded.

    Returns
    -------
    pd.DataFrame
        A DataFrame in which each row corresponds to one detected slow wave.
        Columns include wave_start, wave_stop, amplitude measures, indices,
        the chosen mid-crossing (if applicable), and a "channel" column.
    """
    # Validate EEG data orientation.
    EegData = validate_eeg_data(EegData)
    nChans = EegData.shape[0]

    # Prepare clean intervals (each as a tuple (start, end) in sample indices)
    if CleanEpochs is not None and len(CleanEpochs) > 0:
        clean_intervals = np.array([
            ((epoch - 1) * EpochLength * SampRate, epoch * EpochLength * SampRate)
            for epoch in CleanEpochs
        ])
    else:
        clean_intervals = None

    # ---- Helper functions for per-channel processing ----
    def _process_negzerocrossings(data: np.ndarray, poscross: np.ndarray, negcross: np.ndarray) -> List[Dict]:
        waves = []
        # Define wave boundaries as successive negative zero crossings.
        wave_starts = negcross[:-1]
        wave_stops = negcross[1:]
        for ws, we in zip(wave_starts, wave_stops):
            segment = data[ws:we+1]
            amp_peak = np.max(segment)
            amp_trough = np.min(segment)
            indices_max = np.where(segment == amp_peak)[0]
            ndx_peak = ws + indices_max[-1] if indices_max.size > 0 else np.nan
            indices_min = np.where(segment == amp_trough)[0]
            ndx_trough = ws + indices_min[-1] if indices_min.size > 0 else np.nan
            # Pick the first positive zero crossing within the wave as the mid crossing.
            mids = poscross[(poscross > ws) & (poscross < we)]
            mid_xing = mids[0] if mids.size > 0 else np.nan
            waves.append({
                "wave_start": ws,
                "wave_stop": we,
                "amp_peak": amp_peak,
                "amp_trough": amp_trough,
                "ndx_peak": ndx_peak,
                "ndx_trough": ndx_trough,
                "mid_xing": mid_xing
            })
        return waves

    def _process_poszerocrossings(data: np.ndarray, poscross: np.ndarray, negcross: np.ndarray) -> List[Dict]:
        waves = []
        # Define wave boundaries as successive positive zero crossings.
        wave_starts = poscross[:-1]
        wave_stops = poscross[1:]
        for ws, we in zip(wave_starts, wave_stops):
            segment = data[ws:we+1]
            amp_peak = np.max(segment)
            amp_trough = np.min(segment)
            indices_max = np.where(segment == amp_peak)[0]
            ndx_peak = ws + indices_max[-1] if indices_max.size > 0 else np.nan
            indices_min = np.where(segment == amp_trough)[0]
            ndx_trough = ws + indices_min[-1] if indices_min.size > 0 else np.nan
            mids = negcross[(negcross > ws) & (negcross < we)]
            mid_xing = mids[0] if mids.size > 0 else np.nan
            waves.append({
                "wave_start": ws,
                "wave_stop": we,
                "amp_peak": amp_peak,
                "amp_trough": amp_trough,
                "ndx_peak": ndx_peak,
                "ndx_trough": ndx_trough,
                "mid_xing": mid_xing
            })
        return waves

    def _process_peaks(data: np.ndarray, poscross: np.ndarray, negcross: np.ndarray) -> List[Dict]:
        # For the peak-based method, work with positive half-waves.
        poshalfwave_start = poscross.copy()
        poshalfwave_stop = negcross.copy()
        if poshalfwave_stop.size and poshalfwave_start.size and poshalfwave_stop[0] < poshalfwave_start[0]:
            poshalfwave_stop = poshalfwave_stop[1:]
        if poshalfwave_start.size and poshalfwave_stop.size and poshalfwave_start[-1] > poshalfwave_stop[-1]:
            poshalfwave_start = poshalfwave_start[:-1]
        halfwave_peak_ndx = []
        for s, e in zip(poshalfwave_start, poshalfwave_stop):
            segment = data[s:e+1]
            idx = s + (np.where(segment == np.max(segment))[0][-1] if segment.size > 0 else 0)
            halfwave_peak_ndx.append(idx)
        halfwave_peak_ndx = np.array(halfwave_peak_ndx)
        
        waves = []
        for i in range(len(halfwave_peak_ndx) - 1):
            ws = halfwave_peak_ndx[i]
            we = halfwave_peak_ndx[i+1]
            if i < len(poshalfwave_stop) and (i+1) < len(poshalfwave_start):
                neg_start = poshalfwave_stop[i]
                neg_stop = poshalfwave_start[i+1]
                neg_seg = data[neg_start:neg_stop+1]
                if neg_seg.size > 0:
                    amp_trough = np.min(neg_seg)
                    idx_min = neg_start + np.where(neg_seg == np.min(neg_seg))[0][-1]
                else:
                    amp_trough = np.nan
                    idx_min = np.nan
                neg_xing = neg_start
                pos_xing = poshalfwave_start[i+1]
            else:
                amp_trough = np.nan
                idx_min = np.nan
                neg_xing = np.nan
                pos_xing = np.nan
            amp_peak1 = data[int(halfwave_peak_ndx[i])] if not np.isnan(halfwave_peak_ndx[i]) else np.nan
            amp_peak2 = data[int(halfwave_peak_ndx[i+1])] if not np.isnan(halfwave_peak_ndx[i+1]) else np.nan
            waves.append({
                "wave_start": ws,
                "wave_stop": we,
                "amp_peak1": amp_peak1,
                "amp_peak2": amp_peak2,
                "amp_trough": amp_trough,
                "ndx_trough": idx_min,
                "neg_xing": neg_xing,
                "pos_xing": pos_xing
            })
        return waves

    def _process_troughs(data: np.ndarray, poscross: np.ndarray, negcross: np.ndarray) -> List[Dict]:
        # For the trough-based method, work with negative half-waves.
        neghalfwave_start = negcross.copy()
        neghalfwave_stop = poscross.copy()
        if neghalfwave_stop.size and neghalfwave_start.size and neghalfwave_stop[0] < neghalfwave_start[0]:
            neghalfwave_stop = neghalfwave_stop[1:]
        if neghalfwave_start.size and neghalfwave_stop.size and neghalfwave_start[-1] > neghalfwave_stop[-1]:
            neghalfwave_start = neghalfwave_start[:-1]
        halfwave_trough_ndx = []
        for s, e in zip(neghalfwave_start, neghalfwave_stop):
            segment = data[s:e+1]
            idx = s + (np.where(segment == np.min(segment))[0][-1] if segment.size > 0 else 0)
            halfwave_trough_ndx.append(idx)
        halfwave_trough_ndx = np.array(halfwave_trough_ndx)
        
        waves = []
        for i in range(len(halfwave_trough_ndx) - 1):
            ws = halfwave_trough_ndx[i]
            we = halfwave_trough_ndx[i+1]
            if i < len(neghalfwave_stop) and (i+1) < len(neghalfwave_start):
                pos_start = neghalfwave_stop[i]
                pos_stop = neghalfwave_start[i+1]
                pos_seg = data[pos_start:pos_stop+1]
                if pos_seg.size > 0:
                    amp_peak = np.max(pos_seg)
                    idx_max = pos_start + np.where(pos_seg == np.max(pos_seg))[0][-1]
                else:
                    amp_peak = np.nan
                    idx_max = np.nan
                pos_xing = neghalfwave_stop[i]
                neg_xing = neghalfwave_start[i]
            else:
                amp_peak = np.nan
                idx_max = np.nan
                pos_xing = np.nan
                neg_xing = np.nan
            amp_trough1 = data[int(halfwave_trough_ndx[i])] if not np.isnan(halfwave_trough_ndx[i]) else np.nan
            amp_trough2 = data[int(halfwave_trough_ndx[i+1])] if not np.isnan(halfwave_trough_ndx[i+1]) else np.nan
            waves.append({
                "wave_start": ws,
                "wave_stop": we,
                "amp_trough1": amp_trough1,
                "amp_trough2": amp_trough2,
                "amp_peak": amp_peak,
                "ndx_peak": idx_max,
                "pos_xing": pos_xing,
                "neg_xing": neg_xing
            })
        return waves

    def process_channel(ch: int, data: np.ndarray) -> List[Dict]:
        # Compute common quantities.
        pos_bool = data > 0
        diff_bool = np.diff(pos_bool.astype(int))
        poscross = np.where(diff_bool == 1)[0] + 1
        negcross = np.where(diff_bool == -1)[0] + 1

        # Compute derivative-based peaks and troughs (used in some methods).
        deriv = np.diff(data)
        pos_deriv = deriv > 0
        diff_deriv = np.diff(pos_deriv.astype(int))
        peaks = (np.where(diff_deriv == -1)[0] + 2).astype(int)
        troughs = (np.where(diff_deriv == 1)[0] + 2).astype(int)
        if peaks.size:
            peaks = peaks[data[peaks] >= 0]
        if troughs.size:
            troughs = troughs[data[troughs] <= 0]

        # If the channel is flat, skip processing.
        if np.all(data == 0):
            warnings.warn(f"Channel {ch+1} is flat (only zeros), skipping channel.")
            return []

        # Choose the detection method.
        if WaveStart == 'NegZeroCrossings':
            waves = _process_negzerocrossings(data, poscross, negcross)
        elif WaveStart == 'PosZeroCrossings':
            waves = _process_poszerocrossings(data, poscross, negcross)
        elif WaveStart == 'Peaks':
            waves = _process_peaks(data, poscross, negcross)
        elif WaveStart == 'Troughs':
            waves = _process_troughs(data, poscross, negcross)
        else:
            raise ValueError(f"Unknown WaveStart method: {WaveStart}")

        # Apply clean epoch filtering if requested.
        if clean_intervals is not None and len(waves) > 0:
            waves = [
                wave for wave in waves 
                if any((wave["wave_start"] >= interval[0]) and (wave["wave_stop"] < interval[1]) 
                       for interval in clean_intervals)
            ]

        # --- New: Filter waves based on duration constraints ---
        # Duration in samples must be between 1/max_freq and 1/min_freq seconds.
        lower_bound_samples = (1.0 / max_freq) * SampRate  # e.g., 0.25 s if max_freq=4 Hz.
        upper_bound_samples = (1.0 / min_freq) * SampRate  # e.g., 2 s if min_freq=0.5 Hz.
        waves = [wave for wave in waves 
                 if lower_bound_samples <= (wave["wave_stop"] - wave["wave_start"]) <= upper_bound_samples]

        # If quality check is enabled, plot the results.
        if QualityCheck == 'Yes' and len(waves) > 0:
            channel_wave_dict = { key: np.array([w.get(key, np.nan) for w in waves])
                                  for key in waves[0].keys() }
            plot_quality_check(data, SampRate, channel_wave_dict, WaveStart, ch)
        # Add channel information.
        for wave in waves:
            wave["channel"] = ch  # 0-indexed channel number.
        return waves

    # ---- End helper functions ----

    print("\nProcessing channels:", end=" ")
    t_start = time.time()
    # Process each channel in parallel.
    if QualityCheck == 'Yes':
        n_jobs = 1
        print("Quality check enabled. Running on single thread.")
    else:
        n_jobs = -2

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_channel)(ch, EegData[ch, :]) for ch in range(nChans)
    )
    # Flatten list (each channel returns a list of wave dictionaries).
    all_waves = [wave for channel_waves in results for wave in channel_waves]
    df = pd.DataFrame(all_waves)

    t_elapsed = time.time() - t_start
    print(f"\nSlow-wave detection took {t_elapsed/60:.2f} minutes")
    return df

def validate_eeg_data(EegData: np.ndarray) -> np.ndarray:
    """
    Validate and orient EEG data to be in shape (channels, samples).

    If the data is provided as (samples, channels), it will be transposed
    with a warning.
    """
    if EegData.shape[0] > EegData.shape[1]:
        warnings.warn("The data has been transposed so that channels are rows and samples are columns.")
        EegData = EegData.T
    return EegData

def plot_quality_check(data: np.ndarray, SampRate: Union[int, float],
                       wave_struct: dict, WaveStart: str, ch: int) -> None:
    """
    Plot a quality-check figure for the specified channel.

    Parameters
    ----------
    data : np.ndarray
        1D EEG signal for the channel.
    SampRate : int or float
        Sampling rate of the data.
    wave_struct : dict
        Dictionary containing detected wave parameters for the channel.
    WaveStart : str
        The method used for wave detection.
    ch : int
        Channel index (0-indexed) for labeling.
    """
    print(f"\nPlotting quality check for channel {ch+1}...")
    WavesToPlot = np.arange(9, 20)
    if "wave_start" not in wave_struct or len(wave_struct["wave_start"]) < WavesToPlot[-1] + 1:
        return

    wave_start_arr = np.array(wave_struct["wave_start"])
    wave_stop_arr = np.array(wave_struct["wave_stop"])
    PlotEEGStart = int(wave_start_arr[WavesToPlot[0]])
    PlotEEGEnd = int(wave_stop_arr[WavesToPlot[-1]])
    x_vals = np.arange(PlotEEGStart, PlotEEGEnd + 1) / SampRate

    plt.figure()
    plt.plot(x_vals, data[PlotEEGStart:PlotEEGEnd+1], label="EEG Signal")
    
    if WaveStart in ['NegZeroCrossings', 'PosZeroCrossings']:
        marker = '^' if WaveStart=='NegZeroCrossings' else 'v'
        xs = wave_start_arr[WavesToPlot] / SampRate
        ys = data[wave_start_arr[WavesToPlot].astype(int)]
        plt.plot(xs, ys, 'k'+marker, label='Wave start')
        xs2 = wave_stop_arr[WavesToPlot] / SampRate
        ys2 = data[wave_stop_arr[WavesToPlot].astype(int)]
        plt.plot(xs2, ys2, 'rv', label='Wave stop')
    # Additional plotting for other methods can be added similarly.

    plt.axhline(0, color='k', linestyle='--', label='_nolegend_')
    plt.legend()
    plt.title(f"Channel {ch+1} - Wave start: {WaveStart}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show(block=True)
