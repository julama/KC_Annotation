from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
import os

# === Loading dataset infos ===========================================================
def dataset_info(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    records = []
    for path, loader in mat_data.items():
        mat = loader()
        props = ['id', 'subject', 'cond', 'date', 'age', 'srate']
        record = {prop: getattr(mat['EEG'], prop) for prop in props}
        print(f'Loading dataset infos for {record["id"]} ...')
        record['mat_file'] = path.split('/')[-1] + '.mat'
        records.append(record)
    return pd.DataFrame(records)

# === Loading EEG data ================================================================
def _create_eeg_loader(mat_loader):
    def load_and_process():
        mat = mat_loader()
        data_id = mat['EEG'].id
        print(f"Loading EEG data for {data_id} ...")
        return pd.DataFrame(mat['EEG'].data.T)
    return load_and_process

def eeg_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    loaders = {}
    for path, loader in mat_data.items():
        loaders[path.split('/')[-2]] = _create_eeg_loader(loader)
    return loaders


# === Loading AXO stimulation data =====================================================
def _create_axo_stim_loader(mat_loader):
    """Creates a lazy loader function for AXO stimulation data from .mat files"""
    def load_and_process():
        mat = mat_loader()
        data_id = "axo_stim"  # Default ID for AXO data
        print(f"Loading AXO stimulation data for {data_id} ...")
        
        # Extract M_VAL (values) and M_HEA (header/column names)
        m_val = mat['M_VAL']  # Shape: (samples, channels)
        m_hea = mat['M_HEA']  # Column names
        
        # Ensure M_HEA is properly formatted as a list of strings
        if isinstance(m_hea, np.ndarray):
            # Handle nested arrays - M_HEA has shape (1, 18) with each element being an array
            if m_hea.ndim == 2:
                # Flatten the array and extract string values
                column_names = []
                for item in m_hea.flatten():
                    if isinstance(item, np.ndarray):
                        column_names.append(str(item.item()))
                    else:
                        column_names.append(str(item))
            else:
                column_names = [str(item) for item in m_hea]
        else:
            column_names = list(m_hea)
        
        # Convert to pandas DataFrame with column names from M_HEA
        df = pd.DataFrame(m_val, columns=column_names)
        
        # Make column names unique (there are duplicates in the original data)
        seen = {}
        new_columns = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_columns.append(col)
        df.columns = new_columns
        
        print(f"AXO data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    return load_and_process

def axo_stim_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Creates lazy loader functions for AXO stimulation data from .mat files.
    
    Args:
        mat_data: Dictionary mapping file paths to .mat loader functions
        
    Returns:
        Dictionary mapping subject IDs to AXO stimulation data loader functions
    """
    loaders = {}
    for path, loader in mat_data.items():
        # Extract subject ID from file path (e.g., "C-006" from the filename)
        filename = os.path.basename(path)
        # Extract subject ID from filename like "C-006_eeg_axo048_visit04_17-09-2025_2.mat"
        subject_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
        loaders[subject_id] = _create_axo_stim_loader(loader)
    return loaders

# === Loading Epochs data ================================================================
def _create_epochs_loader(mat_loader):
    """Creates a lazy loader function for epoch data with sample indices"""
    def load_and_process():
        mat = mat_loader()
        data_id = mat['EEG'].id
        print(f"Loading epochs data for {data_id} ...")
        
        # Extract epoch data (visnum and ndxEpoGoodSleep)
        visnum = getattr(mat['EEG'], 'visnum', None)
        ndx_epo_good_sleep = getattr(mat['EEG'], 'ndxEpoGoodSleep', None)
        
        # Create DataFrame with epoch information
        epochs_df = pd.DataFrame()
        
        if visnum is not None:
            # Convert to pandas Series if it's not already
            if not isinstance(visnum, pd.Series):
                visnum = pd.Series(visnum)
            epochs_df['sleep_stage'] = visnum
            
            # Initialize all epochs as bad (0)
            n_epochs = len(visnum)
            epochs_df['is_good_epoch'] = 0
            
            # Mark good epochs based on ndx_epo_good_sleep
            if ndx_epo_good_sleep is not None:
                # Convert to numpy array if it's not already
                if isinstance(ndx_epo_good_sleep, pd.Series):
                    ndx_epo_good_sleep = ndx_epo_good_sleep.values
                elif not isinstance(ndx_epo_good_sleep, np.ndarray):
                    ndx_epo_good_sleep = np.array(ndx_epo_good_sleep)
                
                # Convert MATLAB indices (1-based) to Python indices (0-based)
                good_indices = ndx_epo_good_sleep.astype(int) - 1
                
                # Mark good epochs with 1
                epochs_df.loc[good_indices, 'is_good_epoch'] = 1
        
        # Add epoch indices (20 seconds per epoch, assuming 125 Hz sampling rate)
        samples_per_epoch = 20 * 125  # 20 seconds * 125 Hz
        n_epochs = len(epochs_df)
        
        # Create epoch indices (0-based Python indexing)
        epochs_df['epoch_idx'] = np.arange(n_epochs)
        epochs_df['start_idx'] = epochs_df['epoch_idx'] * samples_per_epoch
        epochs_df['end_idx'] = (epochs_df['epoch_idx'] + 1) * samples_per_epoch - 1
        
        return epochs_df
    
    return load_and_process

def epochs_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Creates loader functions for epoch data from .mat files.
    
    Args:
        mat_data: Dictionary mapping file paths to loader functions for .mat files
        
    Returns:
        Dictionary mapping subject IDs to epoch data loader functions
    """
    loaders = {}
    for path, loader in mat_data.items():
        loaders[path.split('/')[-2]] = _create_epochs_loader(loader)
    return loaders

# === Loading section infos ================================================================
def _create_section_loader(mat_loader):
    def load_and_process():
        mat = mat_loader()
        data_id = mat['EEG'].id
        print(f"Loading section infos for {data_id} ...")

        sections = []
        for ii, sample in enumerate(mat['WT'].W.ON.samples):
            first_stim = int(mat['WT'].T.ON.samples[mat['WT'].T.ON.is_first[ii]-1])-1  # -1 because of matlab indexing
            sections.append({
                'section_id': f'{data_id}_ON{first_stim:07d}',
                'first_stim': first_stim,
                'cond': mat['EEG'].cond,
                'start': int(sample.min()-1), # -1 because of matlab indexing
                'stop': int(sample.max()-1)+1, # -1 because of matlab indexing
                'length': int(sample.max() - sample.min() + 1),
            })
        return pd.DataFrame(sections)
    return load_and_process

def section_info_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    """Load and process matlab data containing EEG recordings.
    
    Args:
        mat_data: Dictionary mapping file paths to loader functions that return matlab data structures
                 containing EEG recordings and ON section markers
    """

    loaders = {}
    for path, loader in mat_data.items():

        loaders[path.split('/')[-2]] = _create_section_loader(loader)

    return loaders

# === Loading stimulation infos ===========================================================
def _create_stim_loader(mat_loader):
    def load_and_process():
        mat = mat_loader()
        data_id = mat['EEG'].id
        print(f"Loading stimulation infos for {data_id} ...")      

        stim_df = pd.DataFrame()
        stim_df['index'] = pd.Series(mat['WT'].T.ON.samples).astype(int) - 1  # -1 because of matlab indexing
        stim_df['cond'] = mat['EEG'].cond

        # Create section_id
        section_starts = [sample.min()-1 for sample in mat['WT'].W.ON.samples] # -1 because of matlab indexing
        section_bins = section_starts + [mat['EEG'].data.shape[1]]
        first_stims = mat['WT'].T.ON.samples[mat['WT'].T.ON.is_first-1]-1
        section_ids = [f'{data_id}_ON{int(first_stim):07d}' for first_stim in first_stims]
        stim_df['section_id'] = pd.cut(stim_df['index'], bins=section_bins, labels=section_ids)

        # Add number within section
        stim_df['number_in_section'] = stim_df.groupby('section_id', observed=True).cumcount() + 1

        # Create stim_id
        stim_id = stim_df['section_id'].astype(str) + '_S' + stim_df['number_in_section'].astype(str).str.zfill(2)
        stim_df.insert(0, 'stim_id', stim_id)

        return stim_df

    return load_and_process

def stim_info_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    loaders = {}
    for path, loader in mat_data.items():
        loaders[path.split('/')[-2]] = _create_stim_loader(loader)
    return loaders

# === Loading channel infos ================================================================
def _create_channel_loader(mat_loader):
    def load_and_process():
        mat = mat_loader()
        data_id = mat['EEG'].id
        print(f"Loading channel infos for {data_id} ...")

        channels = []
        for chanloc in mat['EEG'].chanlocs:
            props = [prop for prop in dir(chanloc) if not prop.startswith('_')]
            channels.append({prop: getattr(chanloc, prop) for prop in props})

        df = pd.DataFrame(channels)

        # Put 'labels' as first column
        cols = df.columns.tolist()
        if 'labels' in cols:
            labels = df['labels']
            cols.remove('labels')
            cols = ['labels'] + cols
            df = df[cols]

        # Add index column nbr
        df.insert(1, 'nbr', range(1, len(df) + 1))

        return df
    return load_and_process

def channel_info_loaders(mat_data: Dict[str, Callable[[], Dict[str, Any]]]) -> Dict[str, Any]:
    loaders = {}
    for path, loader in mat_data.items():
        loaders[path.split('/')[-2]] = _create_channel_loader(loader)
    return loaders
