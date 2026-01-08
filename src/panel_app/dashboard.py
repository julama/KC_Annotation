"""Main Panel Dashboard for EEG Annotation"""

import panel as pn
import holoviews as hv
from holoviews import opts, streams
import param
import numpy as np
import pandas as pd
from typing import Optional, List
from bokeh.models import HoverTool

# Initialize extensions
pn.extension('tabulator', sizing_mode='stretch_width')
hv.extension('bokeh')

# Color mapping for annotation states
ANNOTATION_COLORS = {
    'unannotated': '#7f8c8d',  # Gray
    'KC': '#27ae60',           # Green  
    'non-KC': '#c0392b',       # Red
}


def downsample_minmax(data: np.ndarray, time: np.ndarray, 
                      max_points: int = 4000) -> tuple:
    """
    Downsample preserving min/max for EEG visualization.
    """
    n_samples = len(data)
    if n_samples <= max_points:
        return time, data
    
    # Min-max downsampling preserves peaks
    step = max(1, n_samples // (max_points // 2))
    indices = []
    for i in range(0, n_samples - step, step):
        chunk = data[i:i+step]
        min_idx = i + np.argmin(chunk)
        max_idx = i + np.argmax(chunk)
        if min_idx <= max_idx:
            indices.extend([min_idx, max_idx])
        else:
            indices.extend([max_idx, min_idx])
    
    indices = sorted(set(indices))
    if len(indices) == 0:
        return time, data
    return time[indices], data[indices]


class EEGDashboard(param.Parameterized):
    """
    Panel-based EEG Dashboard with HoloViews visualization.
    
    Features:
    - Multi-channel overlay with subcoordinate_y
    - Focus channel individual rows
    - Linked x-axes for synchronized zooming/panning
    - SW event VSpans with click-to-annotate
    - Navigation buttons
    """
    
    # Reactive parameters
    epoch_index = param.Integer(default=0, bounds=(0, None), doc="Current epoch index")
    update_trigger = param.Integer(default=0, doc="Trigger for annotation updates")
    
    def __init__(self, epoch_manager, sw_events: pd.DataFrame,
                 annotation_manager, focus_channels: List[int] = None,
                 **params):
        super().__init__(**params)
        
        self.epoch_manager = epoch_manager
        self.sw_events = sw_events
        self.annotation_manager = annotation_manager
        self.sampling_rate = epoch_manager.sampling_rate
        self.focus_channels = focus_channels or [34, 55, 70]
        
        # Update epoch bounds
        max_epochs = max(0, epoch_manager.get_epoch_count() - 1)
        self.param.epoch_index.bounds = (0, max_epochs)
        
        # Shared x-range for linking (will be connected via Bokeh)
        self._x_range = None
        
        # Cache current epoch data
        self._current_epoch_data = None
        self._current_sw_events = None
    
    def _get_epoch_data(self):
        """Get current epoch data and SW events (with caching)."""
        # Sync epoch manager
        self.epoch_manager.current_epoch_idx = self.epoch_index
        
        # Get data
        self._current_epoch_data = self.epoch_manager.get_current_epoch()
        
        # Filter SW events for this epoch
        from src.sw_loader import get_sw_events_for_current_epoch
        self._current_sw_events = get_sw_events_for_current_epoch(
            self.sw_events, self.epoch_manager
        )
        
        return self._current_epoch_data, self._current_sw_events
    
    def _create_time_axis(self, n_samples: int) -> np.ndarray:
        """Create time axis in seconds."""
        return np.arange(n_samples) / self.sampling_rate
    
    def _create_sw_vspans(self, sw_events: pd.DataFrame, alpha: float = 0.25) -> hv.Overlay:
        """Create VSpan elements for SW events."""
        if sw_events.empty:
            return hv.Overlay([])
        
        vspans = []
        for _, sw_row in sw_events.iterrows():
            event_id = sw_row['event_id']
            
            rel_start = sw_row.get('relative_start', 0)
            rel_stop = sw_row.get('relative_stop', 0)
            
            start_time = rel_start / self.sampling_rate
            stop_time = rel_stop / self.sampling_rate
            
            status = self.annotation_manager.get_annotation_status(event_id)
            color = ANNOTATION_COLORS.get(status, ANNOTATION_COLORS['unannotated'])
            
            vspan = hv.VSpan(start_time, stop_time).opts(
                color=color,
                alpha=alpha,
                line_width=1,
                line_color=color,
                line_alpha=0.6,
            )
            vspans.append(vspan)
        
        return hv.Overlay(vspans) if vspans else hv.Overlay([])
    
    @param.depends('epoch_index', 'update_trigger')
    def create_main_plot(self) -> hv.Overlay:
        """
        Create main overlay plot with all channels using vertical offset stacking.
        
        Compatible with HoloViews 1.17.x (uses manual offset instead of subcoordinate_y).
        """
        epoch_data, current_sw = self._get_epoch_data()
        
        if epoch_data is None or epoch_data.empty:
            return hv.Text(0, 0, "No data available").opts(
                width=1200, height=400
            )
        
        n_samples = len(epoch_data)
        time_axis = self._create_time_axis(n_samples)
        n_channels = len(epoch_data.columns)
        
        # Select channels for display (limit to ~20 for performance)
        if n_channels > 20:
            step = n_channels // 16
            display_indices = list(range(0, n_channels, step))[:20]
        else:
            display_indices = list(range(n_channels))
        
        # Calculate vertical spacing for channel stacking
        # Normalize each channel and apply offset
        curves = []
        y_offset = 0
        channel_spacing = 100  # Spacing between channels in normalized units
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
                  '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b'] * 3
        
        for i, idx in enumerate(display_indices):
            ch_data = epoch_data.iloc[:, idx].values
            
            # Normalize to ~[-50, 50] range and apply offset
            ch_std = np.std(ch_data)
            if ch_std > 0:
                normalized = (ch_data - np.mean(ch_data)) / ch_std * 30
            else:
                normalized = ch_data - np.mean(ch_data)
            
            offset_data = normalized + y_offset
            t_disp, d_disp = downsample_minmax(offset_data, time_axis)
            
            curve = hv.Curve(
                (t_disp, d_disp),
                kdims=['Time (s)'],
                vdims=['Amplitude'],
                label=f'Ch{idx}'
            ).opts(
                line_width=1,
                color=colors[i % len(colors)],
            )
            curves.append(curve)
            
            # Update offset for next channel
            y_offset += channel_spacing
        
        # Create channel overlay
        channel_overlay = hv.Overlay(curves)
        
        # Add SW event VSpans
        sw_overlay = self._create_sw_vspans(current_sw, alpha=0.2)
        
        # Combine
        combined = sw_overlay * channel_overlay
        
        # Get epoch info for title
        epoch_info = self.epoch_manager.get_current_epoch_info()
        sleep_stage = epoch_info.get('sleep_stage', '?')
        total_epochs = epoch_info.get('total_filtered', 0)
        
        combined.opts(
            opts.Overlay(
                width=1200,
                height=500,
                title=f"Epoch {self.epoch_index + 1}/{total_epochs} | Sleep Stage: {sleep_stage}",
                xlabel='Time (s)',
                ylabel='Channels (stacked)',
                tools=['xwheel_zoom', 'xpan', 'reset', 'tap'],
                active_tools=['xwheel_zoom', 'xpan'],
                shared_axes=True,
                show_legend=False,  # Too cluttered with many channels
            ),
            opts.VSpan(
                apply_ranges=False,  # Don't affect y-axis range
            )
        )
        
        return combined
    
    @param.depends('epoch_index', 'update_trigger')
    def create_focus_plot(self, channel_idx: int) -> hv.Overlay:
        """Create individual focus channel plot."""
        epoch_data, current_sw = self._get_epoch_data()
        
        if epoch_data is None or epoch_data.empty:
            return hv.Text(0, 0, f"No data").opts(width=1200, height=150)
        
        if channel_idx >= len(epoch_data.columns):
            return hv.Text(0, 0, f"Channel {channel_idx} not found").opts(
                width=1200, height=150
            )
        
        n_samples = len(epoch_data)
        time_axis = self._create_time_axis(n_samples)
        
        ch_data = epoch_data.iloc[:, channel_idx].values
        t_disp, d_disp = downsample_minmax(ch_data, time_axis)
        
        # Create main curve
        curve = hv.Curve(
            (t_disp, d_disp),
            kdims=['Time (s)'],
            vdims=['Amplitude (µV)'],
        )
        
        # Add SW VSpans
        sw_overlay = self._create_sw_vspans(current_sw, alpha=0.3)
        
        combined = sw_overlay * curve
        
        combined.opts(
            opts.Curve(
                line_width=1.5,
                color='#34495e',
                tools=['xwheel_zoom', 'xpan', 'reset', 'tap', 'hover'],
                active_tools=['xwheel_zoom', 'xpan'],
            ),
            opts.Overlay(
                width=1200,
                height=150,
                title=f"Channel {channel_idx}",
                xlabel='Time (s)',
                ylabel='µV',
                shared_axes=True,
            ),
            opts.VSpan(
                apply_ranges=False,
            )
        )
        
        return combined
    
    def next_epoch(self, event=None):
        """Navigate to next epoch."""
        max_idx = self.epoch_manager.get_epoch_count() - 1
        if self.epoch_index < max_idx:
            self.epoch_index += 1
    
    def prev_epoch(self, event=None):
        """Navigate to previous epoch."""
        if self.epoch_index > 0:
            self.epoch_index -= 1
    
    def _get_status_text(self) -> str:
        """Get status bar text."""
        epoch_info = self.epoch_manager.get_current_epoch_info()
        counts = self.annotation_manager.get_annotation_count()
        
        return (
            f"**Epoch**: {self.epoch_index + 1} / {epoch_info.get('total_filtered', 0)} | "
            f"**Sleep Stage**: {epoch_info.get('sleep_stage', '?')} | "
            f"**KC**: {counts['KC']} | **non-KC**: {counts['non-KC']} | "
            f"**Unannotated**: {counts['unannotated']}"
        )
    
    @param.depends('epoch_index', 'update_trigger')
    def status_bar(self) -> pn.pane.Markdown:
        """Create reactive status bar."""
        return pn.pane.Markdown(
            self._get_status_text(),
            styles={'font-size': '14px', 'padding': '10px', 'background': '#ecf0f1'}
        )
    
    def create_sw_buttons(self) -> pn.Column:
        """Create annotation buttons for SW events in current epoch."""
        _, current_sw = self._get_epoch_data()
        
        if current_sw.empty:
            return pn.Column(
                pn.pane.Markdown("*No SW events in this epoch*"),
                sizing_mode='stretch_width'
            )
        
        buttons = []
        # Limit to first 30 events for UI
        for _, sw_row in current_sw.head(30).iterrows():
            event_id = sw_row['event_id']
            status = self.annotation_manager.get_annotation_status(event_id)
            
            # Button styling based on status
            if status == 'KC':
                button_type = 'success'
                label = f'SW {event_id}: KC'
            elif status == 'non-KC':
                button_type = 'danger'
                label = f'SW {event_id}: non-KC'
            else:
                button_type = 'default'
                label = f'SW {event_id}: ?'
            
            btn = pn.widgets.Button(
                name=label,
                button_type=button_type,
                width=120,
                height=35,
            )
            
            # Bind click handler
            def on_click(event, eid=event_id):
                self.annotation_manager.toggle_annotation(eid)
                self.update_trigger += 1  # Trigger refresh
            
            btn.on_click(on_click)
            buttons.append(btn)
        
        if len(current_sw) > 30:
            buttons.append(
                pn.pane.Markdown(f"*(Showing 30 of {len(current_sw)} events)*")
            )
        
        return pn.FlexBox(*buttons, flex_wrap='wrap', align_items='flex-start')
    
    @param.depends('epoch_index', 'update_trigger')
    def sw_buttons_view(self):
        """Reactive SW buttons panel."""
        return self.create_sw_buttons()
    
    def view(self) -> pn.Column:
        """Create the complete dashboard layout."""
        
        # Navigation buttons
        prev_btn = pn.widgets.Button(
            name='◀ Previous', 
            button_type='primary',
            width=120,
        )
        prev_btn.on_click(self.prev_epoch)
        
        next_btn = pn.widgets.Button(
            name='Next ▶',
            button_type='primary', 
            width=120,
        )
        next_btn.on_click(self.next_epoch)
        
        # Epoch slider for quick navigation
        epoch_slider = pn.widgets.IntSlider.from_param(
            self.param.epoch_index,
            name='Epoch',
            width=400,
        )
        
        # Navigation row
        nav_row = pn.Row(
            prev_btn,
            pn.Spacer(width=20),
            epoch_slider,
            pn.Spacer(width=20),
            next_btn,
            align='center',
        )
        
        # Create linked plots using DynamicMap for reactivity
        main_dmap = hv.DynamicMap(self.create_main_plot)
        
        # Focus channel plots
        focus_dmaps = []
        for ch_idx in self.focus_channels:
            dmap = hv.DynamicMap(
                lambda idx=ch_idx: self.create_focus_plot(idx)
            )
            focus_dmaps.append(dmap)
        
        # Stack all plots with shared axes
        all_plots = [main_dmap] + focus_dmaps
        
        # Create a linked layout using HoloViews
        # Note: We'll use Panel's Column for layout since we want shared x-axes
        plot_layout = pn.Column(
            pn.pane.HoloViews(main_dmap, linked_axes=True, sizing_mode='stretch_width'),
            *[pn.pane.HoloViews(dmap, linked_axes=True, sizing_mode='stretch_width') 
              for dmap in focus_dmaps],
            sizing_mode='stretch_width',
        )
        
        # Full dashboard layout
        dashboard = pn.Column(
            pn.pane.Markdown(
                "# 🧠 EEG K-Complex Annotation Tool",
                styles={'text-align': 'center'}
            ),
            self.status_bar,
            pn.layout.Divider(),
            nav_row,
            pn.layout.Divider(),
            pn.pane.Markdown("### SW Event Annotations"),
            pn.pane.Markdown("*Click buttons to cycle: Unknown → KC → non-KC → Unknown*"),
            self.sw_buttons_view,
            pn.layout.Divider(),
            pn.pane.Markdown("### EEG Signals"),
            plot_layout,
            sizing_mode='stretch_width',
        )
        
        return dashboard


def create_dashboard(epoch_manager, sw_events: pd.DataFrame,
                    annotation_manager, focus_channels: List[int] = None) -> pn.Column:
    """
    Factory function to create the EEG dashboard.
    
    Args:
        epoch_manager: EpochManager instance
        sw_events: DataFrame with SW events
        annotation_manager: AnnotationManager instance
        focus_channels: List of channel indices for focus rows
        
    Returns:
        Panel Column layout
    """
    dashboard = EEGDashboard(
        epoch_manager=epoch_manager,
        sw_events=sw_events,
        annotation_manager=annotation_manager,
        focus_channels=focus_channels,
    )
    
    return dashboard.view()

