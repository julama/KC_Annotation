"""Load .mat files and extract EEG data, epochs, and channel information"""

import logging
from contextlib import contextmanager

import scipy.io as sio
import pandas as pd
import numpy as np
from typing import Optional


# mat73 logs ERROR for MATLAB objects it cannot decode (e.g. digitalFilter in EEGLAB structs).
# The fields we need (data, srate, …) still load; suppress only this noisy message.
_SUPPRESS_MAT73_SUBSTR = "MATLAB type not supported"


class _SuppressMat73UnsupportedTypeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return _SUPPRESS_MAT73_SUBSTR not in record.getMessage()
        except Exception:
            return True


@contextmanager
def _silence_mat73_unsupported_type_logs():
    flt = _SuppressMat73UnsupportedTypeFilter()
    root = logging.getLogger()
    handlers = list(root.handlers)
    last_resort = getattr(logging, "lastResort", None)
    if last_resort is not None and last_resort not in handlers:
        handlers.append(last_resort)
    for h in handlers:
        h.addFilter(flt)
    try:
        yield
    finally:
        for h in handlers:
            try:
                h.removeFilter(flt)
            except ValueError:
                pass


class EEGData:
    """Container for loaded EEG data and metadata"""
    
    def __init__(self, data: pd.DataFrame, visnum: np.ndarray, srate: float, 
                 chanlocs: pd.DataFrame, data_id: str, artndxn: Optional[np.ndarray] = None):
        self.data = data  # samples × channels
        self.visnum = visnum  # sleep stages array
        self.srate = srate  # sampling rate
        self.chanlocs = chanlocs  # channel locations DataFrame
        self.data_id = data_id  # dataset identifier
        self.artndxn = artndxn  # artifact mask (epochs × channels), 0=bad, 1=good


def load_mat_file(filepath: str) -> EEGData:
    """
    Load .mat file and extract EEG data, epochs, and channel information.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        EEGData object containing all extracted information
    """
    def _squeeze_scalar(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = np.asarray(x).squeeze()
            if x.size == 1:
                return x.item()
            return x
        if isinstance(x, bytes):
            return x.decode(errors="ignore")
        return x

    def _as_numpy(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _as_1d_int_array(x):
        if x is None:
            return None
        x = _as_numpy(x)
        x = np.asarray(x).squeeze()
        if x.ndim == 0:
            return np.array([int(x)])
        return x.astype(int).reshape(-1)

    def _parse_chanlocs(chanlocs_raw):
        """
        Best-effort conversion of MATLAB chanlocs into a DataFrame.
        If parsing fails, we return an empty DataFrame (the app can still run).
        """
        if chanlocs_raw is None:
            return pd.DataFrame()

        # Typical mat73 representation: list of dicts
        if isinstance(chanlocs_raw, list):
            rows = [c for c in chanlocs_raw if isinstance(c, dict)]
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        # Sometimes dict-of-arrays
        if isinstance(chanlocs_raw, dict):
            # If it's a dict of scalar values, just store one row.
            if all(not isinstance(v, (list, np.ndarray)) for v in chanlocs_raw.values()):
                return pd.DataFrame([chanlocs_raw])

            # If it's a dict of arrays, try to make rows
            n = None
            for v in chanlocs_raw.values():
                arr = np.asarray(v)
                if arr.size > 1:
                    n = int(arr.shape[0])
                    break
            if n is None:
                return pd.DataFrame([chanlocs_raw])

            rows = []
            for i in range(n):
                row = {}
                for k, v in chanlocs_raw.items():
                    arr = np.asarray(v).squeeze()
                    try:
                        row[k] = arr[i]
                    except Exception:
                        row[k] = arr
                rows.append(row)
            return pd.DataFrame(rows)

        # Fallback for object ndarray with dict entries
        arr = _as_numpy(chanlocs_raw)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            flat = arr.reshape(-1)
            rows = [item for item in flat if isinstance(item, dict)]
            if rows:
                return pd.DataFrame(rows)

        return pd.DataFrame()

    # ---- Try scipy (MAT v7 and earlier) first ----
    try:
        mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        eeg_struct = mat["EEG"]  # scipy returns a mat_struct-like object

        data_id = getattr(eeg_struct, "id", "unknown")
        srate = float(getattr(eeg_struct, "srate", 125.0))

        # Original code expects MATLAB layout: channels × samples -> transpose -> samples × channels
        eeg_data = getattr(eeg_struct, "data", None)
        if eeg_data is None:
            raise KeyError("EEG.data missing")

        visnum = getattr(eeg_struct, "visnum", None)
        chanlocs_raw = getattr(eeg_struct, "chanlocs", None)
        artndxn_raw = getattr(eeg_struct, "artndxn", None)

        # Extract channel locations (mat_struct representation)
        chanlocs_list = []
        if hasattr(eeg_struct, "chanlocs") and eeg_struct.chanlocs is not None:
            for chanloc in eeg_struct.chanlocs:
                props = [prop for prop in dir(chanloc) if not prop.startswith("_")]
                chanlocs_list.append({prop: getattr(chanloc, prop) for prop in props})
        chanlocs_df = pd.DataFrame(chanlocs_list) if chanlocs_list else pd.DataFrame()

    except NotImplementedError as e:
        # ---- Fallback: MATLAB v7.3 (HDF5) ----
        if "v7.3" not in str(e).lower() and "hdf" not in str(e).lower():
            raise

        import mat73  # reads MATLAB v7.3 MAT files

        with _silence_mat73_unsupported_type_logs():
            mat = mat73.loadmat(filepath)
        eeg_struct = mat["EEG"]

        # mat73 sometimes wraps the EEG struct in a 1-element list
        if isinstance(eeg_struct, list) and len(eeg_struct) == 1 and isinstance(eeg_struct[0], dict):
            eeg_struct = eeg_struct[0]

        if not isinstance(eeg_struct, dict):
            raise ValueError(f"Unexpected v7.3 EEG struct type: {type(eeg_struct)}")

        data_id = _squeeze_scalar(eeg_struct.get("id", "unknown"))
        srate = float(_squeeze_scalar(eeg_struct.get("srate", 125.0)))

        eeg_data = _as_numpy(eeg_struct.get("data", None))
        if eeg_data is None:
            raise KeyError("EEG.data missing in v7.3 file")

        visnum = eeg_struct.get("visnum", None)
        chanlocs_raw = eeg_struct.get("chanlocs", None)
        artndxn_raw = eeg_struct.get("artndxn", None)

        chanlocs_df = _parse_chanlocs(chanlocs_raw)

    # ---- Common normalization / formatting for both code paths ----
    eeg_data = np.asarray(eeg_data)
    eeg_data = np.squeeze(eeg_data)
    if eeg_data.ndim != 2:
        raise ValueError(f"EEG data has unexpected shape after squeeze: {eeg_data.shape}")

    # Convert channels × samples -> samples × channels
    data_df = pd.DataFrame(eeg_data.T)

    # Normalize visnum
    if visnum is None:
        n_epochs = int(len(data_df) / (20 * srate))  # 20 second epochs
        visnum_arr = np.zeros(n_epochs, dtype=int)
    else:
        if isinstance(visnum, pd.Series):
            visnum_arr = visnum.values.astype(int).reshape(-1)
        else:
            visnum_arr = _as_1d_int_array(visnum)
            if visnum_arr is None:
                n_epochs = int(len(data_df) / (20 * srate))
                visnum_arr = np.zeros(n_epochs, dtype=int)

    # Reorder columns if 'labels' exists
    if not chanlocs_df.empty and "labels" in chanlocs_df.columns:
        cols = chanlocs_df.columns.tolist()
        cols.remove("labels")
        cols = ["labels"] + cols
        chanlocs_df = chanlocs_df[cols]

    artndxn = None
    if artndxn_raw is not None:
        artndxn = np.asarray(artndxn_raw)
        artndxn = np.squeeze(artndxn)
        if artndxn.ndim == 1:
            artndxn = artndxn.reshape(-1, 1)

    print(f"Loaded EEG data for {data_id}")
    print(f"  Data shape: {data_df.shape} (samples × channels)")
    print(f"  Sampling rate: {srate} Hz")
    print(f"  Number of epochs: {len(visnum_arr)}")
    print(f"  Channels: {len(chanlocs_df) if not chanlocs_df.empty else data_df.shape[1]}")
    if artndxn is not None:
        print(f"  artndxn shape: {artndxn.shape}")

    return EEGData(
        data=data_df,
        visnum=visnum_arr,
        srate=srate,
        chanlocs=chanlocs_df,
        data_id=data_id,
        artndxn=artndxn,
    )

