"""Handle KC annotations for user-selected regions"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class AnnotationManager:
    """Manages K-complex annotations for user-selected regions"""
    
    def __init__(self, annotation_file: str, sampling_rate: float):
        """
        Initialize annotation manager.
        
        Args:
            annotation_file: Path to CSV file for saving annotations
            sampling_rate: Sampling rate in Hz (for converting time to indices)
        """
        self.annotation_file = annotation_file
        self.sampling_rate = sampling_rate
        
        # Initialize regions dictionary: {region_id: {'start_idx': int, 'stop_idx': int, 'is_kc': int}}
        self.regions: Dict[int, Dict] = {}
        self._next_region_id = 0
        
        # Load existing annotations if file exists
        self._load_annotations()
    
    def _load_annotations(self):
        """Load existing annotations from file if it exists"""
        if Path(self.annotation_file).exists():
            try:
                df = pd.read_csv(self.annotation_file)
                
                # Skip old format files that have event_id (old SW events format)
                if 'event_id' in df.columns:
                    print(f"Warning: Annotation file has old format (event_id). Skipping load. Please use new region-based format.")
                    print(f"  To start fresh, delete or rename: {self.annotation_file}")
                    return
                
                # Also skip if file has is_kc column with values that suggest old format
                # Old format might have is_kc=-1 for unannotated, new format uses 0/1
                if 'is_kc' in df.columns:
                    # Check if any rows have is_kc=-1 (old format indicator)
                    if (df['is_kc'] == -1).any():
                        print(f"Warning: Annotation file appears to have old format (is_kc=-1). Skipping load.")
                        print(f"  To start fresh, delete or rename: {self.annotation_file}")
                        return
                
                # Check if required columns exist
                if 'start_idx' in df.columns and 'stop_idx' in df.columns:
                    # Safety check: if file has thousands of rows, it's likely old SW events data
                    if len(df) > 1000:
                        print(f"Warning: Annotation file has {len(df)} rows, which suggests old SW events data.")
                        print(f"  Skipping load. To start fresh, delete or rename: {self.annotation_file}")
                        return
                    
                    # Load regions - only load if region_id exists (user-selected regions)
                    loaded_count = 0
                    for _, row in df.iterrows():
                        # Skip rows without region_id (might be old format)
                        if 'region_id' not in row or pd.isna(row.get('region_id')):
                            continue
                            
                        try:
                            region_id = int(row['region_id'])
                            start_idx = int(row['start_idx'])
                            stop_idx = int(row['stop_idx'])
                            is_kc = int(row.get('is_kc', 0))  # Default to 0 (unannotated)
                            
                            # Only accept is_kc values 0 or 1
                            if is_kc not in [0, 1]:
                                continue
                            
                            self.regions[region_id] = {
                                'start_idx': start_idx,
                                'stop_idx': stop_idx,
                                'is_kc': is_kc
                            }
                            
                            # Update next region ID
                            if region_id >= self._next_region_id:
                                self._next_region_id = region_id + 1
                            
                            loaded_count += 1
                        except (ValueError, TypeError) as e:
                            # Skip invalid rows
                            continue
                    
                    if loaded_count > 0:
                        print(f"Loaded {loaded_count} user-selected regions from {self.annotation_file}")
                    else:
                        print(f"No valid regions found in {self.annotation_file}")
                else:
                    print(f"Warning: Annotation file exists but missing required columns (start_idx, stop_idx)")
            except Exception as e:
                print(f"Error loading annotations: {e}")
                import traceback
                traceback.print_exc()
    
    def add_region(self, start_idx: int, stop_idx: int, epoch_start_idx: Optional[int] = None) -> int:
        """
        Add a new region.
        
        Args:
            start_idx: Start sample index (absolute or relative to epoch)
            stop_idx: Stop sample index (absolute or relative to epoch)
            epoch_start_idx: If provided, converts relative indices to absolute
            
        Returns:
            region_id: ID of the newly created region
        """
        # Convert to absolute indices if epoch_start_idx is provided
        if epoch_start_idx is not None:
            abs_start = epoch_start_idx + start_idx
            abs_stop = epoch_start_idx + stop_idx
        else:
            abs_start = start_idx
            abs_stop = stop_idx
        
        region_id = self._next_region_id
        self._next_region_id += 1
        
        self.regions[region_id] = {
            'start_idx': abs_start,
            'stop_idx': abs_stop,
            'is_kc': 0  # Default to unannotated
        }
        
        self._save_annotations()
        return region_id
    
    def delete_region(self, region_id: int) -> bool:
        """
        Delete a region.
        
        Args:
            region_id: Region ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if region_id in self.regions:
            del self.regions[region_id]
            self._save_annotations()
            return True
        return False
    
    def update_region(self, region_id: int, start_idx: Optional[int] = None, 
                     stop_idx: Optional[int] = None, is_kc: Optional[int] = None):
        """
        Update a region's properties.
        
        Args:
            region_id: Region ID to update
            start_idx: New start index (optional)
            stop_idx: New stop index (optional)
            is_kc: New KC annotation (0 or 1, optional)
        """
        if region_id not in self.regions:
            raise ValueError(f"Region {region_id} not found")
        
        if start_idx is not None:
            self.regions[region_id]['start_idx'] = start_idx
        if stop_idx is not None:
            self.regions[region_id]['stop_idx'] = stop_idx
        if is_kc is not None:
            self.regions[region_id]['is_kc'] = int(is_kc)
        
        self._save_annotations()
    
    def get_region(self, region_id: int) -> Optional[Dict]:
        """
        Get region data.
        
        Args:
            region_id: Region ID
            
        Returns:
            Dictionary with region data or None if not found
        """
        return self.regions.get(region_id)
    
    def get_regions_for_epoch(self, epoch_start_idx: int, epoch_end_idx: int) -> Dict[int, Dict]:
        """
        Get all regions that overlap with the current epoch.
        
        Args:
            epoch_start_idx: Start sample index of epoch
            epoch_end_idx: End sample index of epoch
            
        Returns:
            Dictionary of {region_id: region_data} for overlapping regions
        """
        overlapping = {}
        for region_id, region in self.regions.items():
            start = region['start_idx']
            stop = region['stop_idx']
            
            # Check if region overlaps with epoch
            if not (stop < epoch_start_idx or start > epoch_end_idx):
                # Convert to relative indices
                rel_start = max(0, start - epoch_start_idx)
                rel_stop = min(epoch_end_idx - epoch_start_idx, stop - epoch_start_idx)
                
                overlapping[region_id] = {
                    'start_idx': start,
                    'stop_idx': stop,
                    'relative_start': rel_start,
                    'relative_stop': rel_stop,
                    'is_kc': region['is_kc']
                }
        
        return overlapping
    
    def set_annotation(self, region_id: int, is_kc: bool):
        """
        Set annotation for a region and save immediately.
        
        Args:
            region_id: Region ID
            is_kc: True for KC, False for unannotated
        """
        if region_id not in self.regions:
            raise ValueError(f"Region {region_id} not found")
        
        self.regions[region_id]['is_kc'] = 1 if is_kc else 0
        self._save_annotations()
    
    def get_annotation_status(self, region_id: int) -> str:
        """
        Get human-readable annotation status.
        
        Args:
            region_id: Region ID
            
        Returns:
            'KC' or 'unannotated'
        """
        if region_id not in self.regions:
            return 'unannotated'
        
        is_kc = self.regions[region_id]['is_kc']
        return 'KC' if is_kc == 1 else 'unannotated'
    
    def _save_annotations(self):
        """Save annotations to CSV file"""
        if not self.regions:
            # Create empty file with headers
            empty_df = pd.DataFrame(columns=['region_id', 'start_idx', 'stop_idx', 'is_kc'])
            empty_df.to_csv(self.annotation_file, index=False)
            return
        
        # Create DataFrame from regions
        data = []
        for region_id, region in self.regions.items():
            data.append({
                'region_id': region_id,
                'start_idx': region['start_idx'],
                'stop_idx': region['stop_idx'],
                'is_kc': region['is_kc']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.annotation_file, index=False)
    
    def get_annotation_count(self) -> Dict[str, int]:
        """Get count of annotations by type"""
        counts = {'KC': 0, 'unannotated': 0}
        
        for region in self.regions.values():
            if region['is_kc'] == 1:
                counts['KC'] += 1
            else:
                counts['unannotated'] += 1
        
        return counts
    
    def get_all_regions(self) -> Dict[int, Dict]:
        """Get all regions"""
        return self.regions.copy()
