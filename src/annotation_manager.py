"""Handle KC/non-KC annotations and save to CSV"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class AnnotationManager:
    """Manages K-complex annotations for SW events"""
    
    def __init__(self, sw_events: pd.DataFrame, annotation_file: str):
        """
        Initialize annotation manager.
        
        Args:
            sw_events: DataFrame with SW events (columns: start_idx, stop_idx, event_id)
            annotation_file: Path to CSV file for saving annotations
        """
        self.sw_events = sw_events.copy()
        self.annotation_file = annotation_file
        
        # Initialize annotation dictionary: {event_id: is_kc (0 or 1)}
        self.annotations: Dict[int, int] = {}
        
        # Load existing annotations if file exists
        self._load_annotations()
    
    def _load_annotations(self):
        """Load existing annotations from file if it exists"""
        if Path(self.annotation_file).exists():
            try:
                df = pd.read_csv(self.annotation_file)
                
                # Check if required columns exist
                if 'event_id' in df.columns and 'is_kc' in df.columns:
                    # Create mapping from event_id to is_kc
                    for _, row in df.iterrows():
                        event_id = int(row['event_id'])
                        is_kc = int(row['is_kc'])
                        self.annotations[event_id] = is_kc
                    
                    print(f"Loaded {len(self.annotations)} existing annotations from {self.annotation_file}")
                else:
                    print(f"Warning: Annotation file exists but missing required columns")
            except Exception as e:
                print(f"Error loading annotations: {e}")
    
    def get_annotation(self, event_id: int) -> Optional[int]:
        """
        Get annotation for a SW event.
        
        Args:
            event_id: Event ID
            
        Returns:
            1 if KC, 0 if non-KC, None if not annotated
        """
        return self.annotations.get(event_id)
    
    def set_annotation(self, event_id: int, is_kc: bool):
        """
        Set annotation for a SW event and save immediately.
        
        Args:
            event_id: Event ID
            is_kc: True for KC, False for non-KC
        """
        self.annotations[event_id] = 1 if is_kc else 0
        self._save_annotations()
    
    def toggle_annotation(self, event_id: int):
        """
        Toggle annotation for a SW event.
        Cycles: None -> KC (1) -> non-KC (0) -> None
        
        Args:
            event_id: Event ID
        """
        current = self.get_annotation(event_id)
        
        if current is None:
            # Not annotated -> mark as KC
            self.set_annotation(event_id, True)
        elif current == 1:
            # KC -> mark as non-KC
            self.set_annotation(event_id, False)
        else:
            # non-KC -> remove annotation
            if event_id in self.annotations:
                del self.annotations[event_id]
            self._save_annotations()
    
    def _save_annotations(self):
        """Save annotations to CSV file"""
        # Create DataFrame with all SW events and their annotations
        result_df = self.sw_events[['start_idx', 'stop_idx', 'event_id']].copy()
        result_df['is_kc'] = result_df['event_id'].map(self.annotations).fillna(-1).astype(int)
        
        # Only save events that have been annotated (is_kc >= 0)
        # Or save all events with -1 for unannotated
        result_df.to_csv(self.annotation_file, index=False)
    
    def get_annotation_status(self, event_id: int) -> str:
        """
        Get human-readable annotation status.
        
        Args:
            event_id: Event ID
            
        Returns:
            'KC', 'non-KC', or 'unannotated'
        """
        annotation = self.get_annotation(event_id)
        if annotation is None:
            return 'unannotated'
        elif annotation == 1:
            return 'KC'
        else:
            return 'non-KC'
    
    def get_annotation_count(self) -> Dict[str, int]:
        """Get count of annotations by type"""
        counts = {'KC': 0, 'non-KC': 0, 'unannotated': 0}
        
        for event_id in self.sw_events['event_id']:
            status = self.get_annotation_status(event_id)
            counts[status] += 1
        
        return counts

