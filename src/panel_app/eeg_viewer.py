"""HoloViews-based EEG Viewer with subcoordinate_y for high-performance display"""

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread
import param
from typing import Optional, List, Dict, Tuple

# Use Bokeh backend for interactivity
hv.extension('bokeh')


# Color mapping for annotation states
ANNOTATION_COLORS = {
    'unannotated': '#808080',  # Gray
    'KC': '#2ecc71',           # Green
    'non-KC': '#e74c3c',       # Red
}


def downsample_for_display(data: np.ndarray, time: np.ndarray, 
                           max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample data for display using LTTB-like approach.
    
    For EEG at 125Hz with 20s epochs = 2500 samples, usually no downsampling needed.
    But for higher sampling rates, this prevents UI lag.
    """
    n_samples = len(data)
    if n_samples <= max_points:
        return time, data
    
    # Simple decimation with min/max preservation for EEG
    step = n_samples // (max_points // 2)
    indices = []
    for i in range(0, n_samples - step, step):
        chunk = data[i:i+step]
        min_idx = i + np.argmin(chunk)
        max_idx = i + np.argmax(chunk)
        indices.extend(sorted([min_idx, max_idx]))
    
    indices = sorted(set(indices))
    return time[indices], data[indices]


class EEGViewer(param.Parameterized):
    """
    High-performance EEG viewer using HoloViews with subcoordinate_y.
    
    Displays:
    - Main overlay plot with all channels using subcoordinate_y
    - Individual focus channel rows below
    - SW event markers as VSpans
    """
    
    # Reactive parameters
    epoch_index = param.Integer(default=0, bounds=(0, None), doc="Current epoch index")
    
    def __init__(self, epoch_manager, sw_events: pd.DataFrame, 
                 annotation_manager, focus_channels: List[int] = None,
                 **params):
        """
        Initialize EEG Viewer.
        
        Args:
            epoch_manager: EpochManager instance
            sw_events: DataFrame with SW events
            annotation_manager: AnnotationManager instance
            focus_channels: List of channel indices for individual row display
        """
        super().__init__(**params)
        
        self.epoch_manager = epoch_manager
        self.sw_events = sw_events
        self.annotation_manager = annotation_manager
        self.sampling_rate = epoch_manager.sampling_rate
        self.focus_channels = focus_channels or [34, 55, 70]
        
        # Update epoch bounds
        max_epochs = max(0, epoch_manager.get_epoch_count() - 1)
        self.param.epoch_index.bounds = (0, max_epochs)
        
        # Keep track of annotation updates
        self._annotation_version = param.Integer(default=0)
    
    def get_current_epoch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get current epoch data and SW events."""
        # Sync epoch manager with our index
        self.epoch_manager.current_epoch_idx = self.epoch_index
        
        # Get epoch data
        epoch_data = self.epoch_manager.get_current_epoch()
        epoch_info = self.epoch_manager.get_current_epoch_info()
        
        # Filter SW events for this epoch
        from src.sw_loader import get_sw_events_for_current_epoch
        current_sw = get_sw_events_for_current_epoch(self.sw_events, self.epoch_manager)
        
        return epoch_data, current_sw
    
    def _create_time_axis(self, n_samples: int) -> np.ndarray:
        """Create time axis in seconds."""
        return np.arange(n_samples) / self.sampling_rate
    
    def _get_channel_names(self, epoch_data: pd.DataFrame) -> List[str]:
        """Get channel names from epoch data."""
        return list(epoch_data.columns)
    
    def create_overlay_plot(self) -> hv.Overlay:
        """
        Create main overlay plot with all channels using vertical offset stacking.
        
        Compatible with HoloViews 1.17.x (uses manual offset instead of subcoordinate_y).
        """
        epoch_data, current_sw = self.get_current_epoch_data()
        
        if epoch_data.empty:
            return hv.Text(0, 0, "No data available").opts(
                width=1200, height=400
            )
        
        n_samples = len(epoch_data)
        time_axis = self._create_time_axis(n_samples)
        channel_names = self._get_channel_names(epoch_data)
        
        # Create curves for each channel with vertical offset stacking
        curves = []
        n_channels = len(channel_names)
        
        # Select a subset of channels for the overlay (every Nth channel if too many)
        if n_channels > 20:
            channel_step = n_channels // 16
            display_indices = list(range(0, n_channels, channel_step))[:20]
        else:
            display_indices = list(range(n_channels))
        
        y_offset = 0
        channel_spacing = 100
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
                  '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b'] * 3
        
        for i, ch_idx in enumerate(display_indices):
            ch_name = channel_names[ch_idx]
            ch_data = epoch_data.iloc[:, ch_idx].values
            
            # Normalize and offset
            ch_std = np.std(ch_data)
            if ch_std > 0:
                normalized = (ch_data - np.mean(ch_data)) / ch_std * 30
            else:
                normalized = ch_data - np.mean(ch_data)
            
            offset_data = normalized + y_offset
            
            # Downsample if needed
            t_disp, d_disp = downsample_for_display(offset_data, time_axis)
            
            # Create curve
            curve = hv.Curve(
                (t_disp, d_disp), 
                kdims=['Time (s)'], 
                vdims=[f'Ch{ch_idx}'],
                label=f'Ch {ch_idx}'
            ).opts(
                line_width=1,
                color=colors[i % len(colors)],
                tools=['xwheel_zoom', 'xpan', 'reset'],
            )
            curves.append(curve)
            y_offset += channel_spacing
        
        # Create overlay of all channel curves
        overlay = hv.Overlay(curves)
        
        # Add SW event VSpans
        sw_vspans = self._create_sw_vspans(current_sw)
        if sw_vspans:
            overlay = overlay * hv.Overlay(sw_vspans)
        
        # Apply styling
        overlay.opts(
            opts.Curve(
                line_width=1,
                tools=['xwheel_zoom', 'xpan', 'reset', 'hover'],
                active_tools=['xwheel_zoom', 'xpan'],
            ),
            opts.Overlay(
                width=1200,
                height=500,
                title=f"EEG Epoch {self.epoch_index + 1}/{self.epoch_manager.get_epoch_count()}",
                xlabel='Time (s)',
                ylabel='Channels (stacked)',
                shared_axes=True,
                show_legend=False,  # Too many channels for legend
            )
        )
        
        return overlay
    
    def create_focus_channel_plot(self, channel_idx: int) -> hv.Overlay:
        """
        Create individual focus channel plot.
        
        Args:
            channel_idx: Channel index to display
        """
        epoch_data, current_sw = self.get_current_epoch_data()
        
        if epoch_data.empty or channel_idx >= len(epoch_data.columns):
            return hv.Text(0, 0, f"Channel {channel_idx} not available").opts(
                width=1200, height=150
            )
        
        n_samples = len(epoch_data)
        time_axis = self._create_time_axis(n_samples)
        channel_names = self._get_channel_names(epoch_data)
        
        ch_name = channel_names[channel_idx]
        ch_data = epoch_data.iloc[:, channel_idx].values
        
        # Downsample if needed
        t_disp, d_disp = downsample_for_display(ch_data, time_axis)
        
        # Create main curve
        curve = hv.Curve(
            (t_disp, d_disp),
            kdims=['Time (s)'],
            vdims=['Amplitude (µV)'],
            label=f'Channel {channel_idx}'
        )
        
        # Add SW event VSpans
        sw_vspans = self._create_sw_vspans(current_sw)
        if sw_vspans:
            overlay = curve * hv.Overlay(sw_vspans)
        else:
            overlay = curve
        
        # Apply styling
        overlay.opts(
            opts.Curve(
                line_width=1.5,
                color='#2c3e50',
                tools=['xwheel_zoom', 'xpan', 'reset', 'hover'],
                active_tools=['xwheel_zoom', 'xpan'],
            ),
            opts.Overlay(
                width=1200,
                height=150,
                title=f"Channel {channel_idx} ({ch_name})",
                xlabel='Time (s)',
                ylabel='µV',
            )
        )
        
        return overlay
    
    def _create_sw_vspans(self, sw_events: pd.DataFrame) -> List[hv.VSpan]:
        """
        Create VSpan elements for SW events.
        
        Args:
            sw_events: DataFrame with SW events for current epoch
            
        Returns:
            List of hv.VSpan elements
        """
        if sw_events.empty:
            return []
        
        vspans = []
        for _, sw_row in sw_events.iterrows():
            event_id = sw_row['event_id']
            
            # Get time boundaries
            rel_start = sw_row.get('relative_start', 0)
            rel_stop = sw_row.get('relative_stop', 0)
            
            start_time = rel_start / self.sampling_rate
            stop_time = rel_stop / self.sampling_rate
            
            # Get annotation status and color
            status = self.annotation_manager.get_annotation_status(event_id)
            color = ANNOTATION_COLORS.get(status, ANNOTATION_COLORS['unannotated'])
            
            # Create VSpan with lower alpha for better visibility
            vspan = hv.VSpan(start_time, stop_time).opts(
                color=color,
                alpha=0.3,
                line_width=0,
            )
            vspans.append(vspan)
        
        return vspans
    
    def create_linked_plots(self) -> hv.Layout:
        """
        Create all plots with linked x-axes.
        
        Returns a layout with:
        - Main overlay plot (top)
        - Focus channel rows (bottom, with linked x-axis)
        """
        # Create main overlay plot
        main_plot = self.create_overlay_plot()
        
        # Create focus channel plots
        focus_plots = []
        for ch_idx in self.focus_channels:
            focus_plot = self.create_focus_channel_plot(ch_idx)
            focus_plots.append(focus_plot)
        
        # Stack plots vertically with linked x-axes
        # Using shared_axes=True in the layout
        all_plots = [main_plot] + focus_plots
        
        layout = hv.Layout(all_plots).cols(1).opts(
            opts.Layout(shared_axes=True)
        )
        
        return layout
    
    def view(self) -> hv.Layout:
        """Get the complete viewer layout."""
        return self.create_linked_plots()

