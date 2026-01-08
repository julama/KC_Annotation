"""Reusable Plotly plot components for EEG visualization"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import mne
from typing import List, Optional, Tuple
import sys
from pathlib import Path
# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def create_raw_plot(epoch_data: pd.DataFrame, sw_events: pd.DataFrame, 
                    annotation_manager, sampling_rate: float) -> go.Figure:
    """
    Create Plotly figure for raw epoch data with SW event markers.
    
    Args:
        epoch_data: DataFrame with shape (samples × channels)
        sw_events: DataFrame with SW events for current epoch
        annotation_manager: AnnotationManager instance
        sampling_rate: Sampling rate (Hz)
        
    Returns:
        Plotly Figure
    """
    print(f"[DEBUG PLOT] Creating raw plot with {len(sw_events)} SW events")
    fig = go.Figure()
    
    # Create time axis
    n_samples = len(epoch_data)
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Plot a subset of channels (or all if few channels)
    n_channels = len(epoch_data.columns)
    if n_channels > 20:
        # Plot every 5th channel to avoid clutter
        channels_to_plot = epoch_data.columns[::5]
    else:
        channels_to_plot = epoch_data.columns
    
    # Add offset for each channel
    offset = 0
    channel_offset_map = {}
    
    for ch_idx, ch_name in enumerate(channels_to_plot):
        channel_data = epoch_data[ch_name].values
        channel_offset_map[ch_idx] = offset
        
        # Add channel trace
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=channel_data + offset,
            mode='lines',
            name=f'Ch {ch_idx}',
            line=dict(width=1),
            hovertemplate=f'Channel {ch_idx}<br>Time: %{{x:.2f}}s<br>Amplitude: %{{y:.2f}}<extra></extra>'
        ))
        
        # Update offset for next channel
        if len(channel_data) > 0:
            offset += np.max(channel_data) - np.min(channel_data) + np.std(channel_data) * 2
    
    # Add SW event markers (limit to first 50 for performance)
    if not sw_events.empty:
        sw_events_limited = sw_events.head(50)  # Limit to first 50 events for performance
        if len(sw_events) > 50:
            print(f"[DEBUG PLOT] Limiting SW events from {len(sw_events)} to 50 for performance")
        for _, sw_row in sw_events_limited.iterrows():
            event_id = sw_row['event_id']
            rel_start = sw_row.get('relative_start', sw_row['start_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
            rel_stop = sw_row.get('relative_stop', sw_row['stop_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
            
            start_time = rel_start / sampling_rate
            stop_time = rel_stop / sampling_rate
            
            # Get annotation status
            annotation = annotation_manager.get_annotation(event_id)
            if annotation is None:
                color = 'blue'
                name = f'SW {event_id} (unannotated)'
            elif annotation == 1:
                color = 'green'
                name = f'SW {event_id} (KC)'
            else:
                color = 'red'
                name = f'SW {event_id} (non-KC)'
            
            # Add vertical lines for SW event boundaries
            fig.add_vline(
                x=start_time,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                annotation_text=f"SW {event_id} start"
            )
            fig.add_vline(
                x=stop_time,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                annotation_text=f"SW {event_id} stop"
            )
            
            # Add shaded region for SW event (clickable via time-based detection)
            fig.add_vrect(
                x0=start_time,
                x1=stop_time,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=1,
                line_color=color,
                annotation_text=name
            )
    
    fig.update_layout(
        title="Raw Epoch Data",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (offset by channel)",
        hovermode='closest',
        height=400
    )
    
    print("[DEBUG PLOT] Raw plot created")
    return fig


def create_channel_plots(epoch_data: pd.DataFrame, sw_events: pd.DataFrame,
                        annotation_manager, channel_indices: List[int],
                        sampling_rate: float) -> go.Figure:
    """
    Create subplot figure with individual channel plots.
    
    Args:
        epoch_data: DataFrame with shape (samples × channels)
        sw_events: DataFrame with SW events for current epoch
        annotation_manager: AnnotationManager instance
        channel_indices: List of channel indices to plot
        sampling_rate: Sampling rate (Hz)
        
    Returns:
        Plotly Figure with subplots
    """
    print(f"[DEBUG PLOT] Creating channel plots with {len(sw_events)} SW events")
    n_channels = len(channel_indices)
    fig = make_subplots(
        rows=n_channels,
        cols=1,
        subplot_titles=[f'Channel {idx}' for idx in channel_indices],
        vertical_spacing=0.1
    )
    
    # Create time axis
    n_samples = len(epoch_data)
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Plot each channel
    for plot_idx, ch_idx in enumerate(channel_indices):
        if ch_idx < len(epoch_data.columns):
            ch_name = epoch_data.columns[ch_idx]
            channel_data = epoch_data[ch_name].values
            
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=channel_data,
                    mode='lines',
                    name=f'Ch {ch_idx}',
                    line=dict(width=1.5),
                    showlegend=False
                ),
                row=plot_idx + 1,
                col=1
            )
            
            # Add SW event markers for this channel (limit to first 50 for performance)
            if not sw_events.empty:
                sw_events_limited = sw_events.head(50)  # Limit for performance
                for _, sw_row in sw_events_limited.iterrows():
                    event_id = sw_row['event_id']
                    rel_start = sw_row.get('relative_start', sw_row['start_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
                    rel_stop = sw_row.get('relative_stop', sw_row['stop_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
                    
                    start_time = rel_start / sampling_rate
                    stop_time = rel_stop / sampling_rate
                    
                    # Get annotation status
                    annotation = annotation_manager.get_annotation(event_id)
                    if annotation is None:
                        color = 'blue'
                    elif annotation == 1:
                        color = 'green'
                    else:
                        color = 'red'
                    
                    # Add shaded region
                    fig.add_vrect(
                        x0=start_time,
                        x1=stop_time,
                        fillcolor=color,
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        row=plot_idx + 1,
                        col=1
                    )
    
    fig.update_layout(
        title="Individual Channel Plots",
        height=300 * n_channels,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (s)", row=n_channels, col=1)
    fig.update_yaxes(title_text="Amplitude", row=n_channels, col=1)
    
    print("[DEBUG PLOT] Channel plots created")
    return fig


def create_topoplot(epoch_data: pd.DataFrame, sw_event: pd.Series,
                    chanlocs: pd.DataFrame, sampling_rate: float,
                    filter_range: Tuple[float, float] = None) -> go.Figure:
    """
    Create topoplot for a SW event using MNE (converted to Plotly).
    
    Args:
        epoch_data: DataFrame with shape (samples × channels)
        sw_event: Series with SW event info (start_idx, stop_idx, etc.)
        chanlocs: DataFrame with channel locations
        sampling_rate: Sampling rate (Hz)
        filter_range: Tuple of (low_freq, high_freq) for filtering, or None
        
    Returns:
        Plotly Figure with topoplot (as image)
    """
    print(f"[DEBUG TOPO] Creating topoplot for event {sw_event.get('event_id', '?')}")
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import io
    import base64
    # Import preprocessing function
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.preprocessing import apply_bandpass_filter
    
    # Extract SW event window
    if 'relative_start' in sw_event and 'relative_stop' in sw_event:
        start_idx = int(sw_event['relative_start'])
        stop_idx = int(sw_event['relative_stop'])
    else:
        # Fallback: use absolute indices
        start_idx = int(sw_event['start_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
        stop_idx = int(sw_event['stop_idx'] - epoch_data.index[0] if hasattr(epoch_data.index, '__len__') else 0)
    
    # Extract data window
    sw_data = epoch_data.iloc[start_idx:stop_idx+1].copy()
    
    # Apply bandpass filter if specified
    if filter_range is None:
        filter_range = (config.SW_FILTER_LOW, config.SW_FILTER_HIGH)
    
    low_freq, high_freq = filter_range
    filtered_data = apply_bandpass_filter(sw_data, low_freq, high_freq, sampling_rate)
    
    # Average across time window
    mean_data = filtered_data.mean(axis=0).values
    
    # Get channel positions
    if not chanlocs.empty:
        if 'X' in chanlocs.columns and 'Y' in chanlocs.columns:
            pos = np.array([chanlocs['X'].values, chanlocs['Y'].values]).T
        elif 'x' in chanlocs.columns and 'y' in chanlocs.columns:
            pos = np.array([chanlocs['x'].values, chanlocs['y'].values]).T
        else:
            # Fallback: create circular layout
            n_chans = len(mean_data)
            angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
            pos = np.array([np.cos(angles), np.sin(angles)]).T
    else:
        # Fallback: create circular layout
        n_chans = len(mean_data)
        angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
        pos = np.array([np.cos(angles), np.sin(angles)]).T
    
    # Create topoplot using MNE
    fig_mne, ax = plt.subplots(figsize=(4, 4))
    try:
        mne.viz.plot_topomap(
            mean_data,
            pos,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(None, None)
        )
        
        # Convert to base64 image
        buf = io.BytesIO()
        fig_mne.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig_mne)
        
        # Create Plotly figure with image
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_str}',
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top"
            )
        )
        fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False)
        fig.update_layout(
            title=f"SW Event {sw_event.get('event_id', '?')} ({low_freq}-{high_freq} Hz)",
            height=300,
            width=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        print(f"[DEBUG TOPO] Topoplot completed for event {sw_event.get('event_id', '?')}")
        return fig
    except Exception as e:
        print(f"[ERROR TOPO] Error creating topoplot: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close(fig_mne)
        except:
            pass
        # Return placeholder on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating topoplot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title=f"SW Event {sw_event.get('event_id', '?')}", height=300, width=300)
        return fig


def create_topoplots_row(epoch_data: pd.DataFrame, sw_events: pd.DataFrame,
                        chanlocs: pd.DataFrame, annotation_manager,
                        sampling_rate: float, max_topoplots: int = 5) -> List[go.Figure]:
    """
    Create a list of topoplot figures, one for each SW event in current epoch.
    
    Args:
        epoch_data: DataFrame with shape (samples × channels)
        sw_events: DataFrame with SW events for current epoch
        chanlocs: DataFrame with channel locations
        annotation_manager: AnnotationManager instance
        sampling_rate: Sampling rate (Hz)
        max_topoplots: Maximum number of topoplots to create (default: 10)
        
    Returns:
        List of Plotly Figures (one per SW event, limited to max_topoplots)
    """
    print(f"[DEBUG PLOT] Creating topoplots for {len(sw_events)} SW events (max: {max_topoplots})")
    
    if len(sw_events) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No SW events in this epoch",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Topoplots (0.5-2 Hz filtered)", height=300)
        return [fig]
    
    # Limit number of topoplots to avoid performance issues
    sw_events_limited = sw_events.head(max_topoplots)
    
    if len(sw_events) > max_topoplots:
        print(f"[DEBUG PLOT] Limiting topoplots from {len(sw_events)} to {max_topoplots}")
    
    # Create one topoplot per SW event
    topoplots = []
    for idx, (_, sw_row) in enumerate(sw_events_limited.iterrows()):
        print(f"[DEBUG PLOT] Creating topoplot {idx+1}/{len(sw_events_limited)}")
        topo_fig = create_topoplot(epoch_data, sw_row, chanlocs, sampling_rate)
        topoplots.append(topo_fig)
    
    print("[DEBUG PLOT] All topoplots created")
    return topoplots

