"""Main Dash application for EEG annotation tool"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional
from src.gui.components import create_raw_plot, create_channel_plots, create_topoplots_row
from src.epoch_manager import EpochManager
from src.sw_loader import get_sw_events_for_current_epoch
from src.annotation_manager import AnnotationManager
import sys
from pathlib import Path
# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def create_app(eeg_data: pd.DataFrame, visnum: np.ndarray, sampling_rate: float,
               chanlocs: pd.DataFrame, sw_events: pd.DataFrame,
               annotation_manager: AnnotationManager,
               epoch_manager: EpochManager,
               channel_indices: list = None):
    """
    Create and configure Dash application.
    
    Args:
        eeg_data: Preprocessed EEG data (samples × channels)
        visnum: Sleep stage array
        sampling_rate: Sampling rate (Hz)
        chanlocs: Channel locations DataFrame
        sw_events: All SW events DataFrame
        annotation_manager: AnnotationManager instance
        epoch_manager: EpochManager instance
        channel_indices: List of channel indices to display individually
    """
    if channel_indices is None:
        channel_indices = config.DEFAULT_CHANNEL_INDICES
    
    app = dash.Dash(__name__)
    
    # Store data in app server for callbacks
    app.server.eeg_data = eeg_data
    app.server.visnum = visnum
    app.server.sampling_rate = sampling_rate
    app.server.chanlocs = chanlocs
    app.server.sw_events = sw_events
    app.server.annotation_manager = annotation_manager
    app.server.epoch_manager = epoch_manager
    app.server.channel_indices = channel_indices
    
    # Initial plots - create empty placeholders to speed up startup
    # Real plots will be created by callbacks
    initial_raw_plot = go.Figure()
    initial_raw_plot.add_annotation(text="Loading...", xref="paper", yref="paper", 
                                    x=0.5, y=0.5, showarrow=False)
    initial_raw_plot.update_layout(title="Raw Epoch Data", height=400)
    
    initial_channel_plot = go.Figure()
    initial_channel_plot.add_annotation(text="Loading...", xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    initial_channel_plot.update_layout(title="Individual Channel Plots", height=600)
    
    initial_topoplots = [go.Figure()]
    
    # Layout
    app.layout = html.Div([
        html.Div([
            html.H1("EEG K-Complex Annotation Tool", style={'textAlign': 'center'}),
            html.Div(id='status-bar', style={'textAlign': 'center', 'margin': '10px'}),
        ]),
        
        html.Div([
            html.Button('Previous Epoch', id='prev-btn', n_clicks=0,
                       style={'margin': '5px', 'padding': '10px 20px'}),
            html.Button('Next Epoch', id='next-btn', n_clicks=0,
                       style={'margin': '5px', 'padding': '10px 20px'}),
            html.Div(id='epoch-info', style={'display': 'inline-block', 'margin': '10px'}),
        ], style={'textAlign': 'center', 'margin': '10px'}),
        
        html.Div([
            html.Div([
                html.Label("Click on plot to annotate SW events, or use buttons below:"),
                html.Div(id='sw-event-buttons', style={'margin': '10px'}),
            ], style={'textAlign': 'center', 'margin': '10px'}),
        ]),
        
        html.Div([
            dcc.Graph(id='raw-plot', figure=initial_raw_plot, style={'height': '400px'})
        ]),
        
        html.Div([
            dcc.Graph(id='channel-plots', figure=initial_channel_plot)
        ]),
        
        html.Div([
            html.Div(id='topoplots-container')
        ]),
        
        dcc.Store(id='current-epoch-store', data=0),
        dcc.Store(id='sw-events-store', data=[]),
        dcc.Location(id='url', refresh=False),  # Use Location to trigger on page load
    ])
    
    # Callbacks
    @app.callback(
        [Output('raw-plot', 'figure'),
         Output('channel-plots', 'figure'),
         Output('topoplots-container', 'children'),
         Output('status-bar', 'children'),
         Output('epoch-info', 'children'),
         Output('current-epoch-store', 'data'),
         Output('sw-events-store', 'data'),
         Output('sw-event-buttons', 'children')],
        [Input('prev-btn', 'n_clicks'),
         Input('next-btn', 'n_clicks'),
         Input('url', 'pathname')],  # Trigger on initial page load
        [State('current-epoch-store', 'data')]
    )
    def update_display(prev_clicks, next_clicks, pathname, current_epoch_data):
        """Update all plots and status when epoch changes"""
        print(f"\n[DEBUG] Callback triggered!")
        print(f"[DEBUG] Triggered: {callback_context.triggered}")
        print(f"[DEBUG] pathname: {pathname}")
        
        try:
            ctx = callback_context
            epoch_manager = app.server.epoch_manager
            annotation_manager = app.server.annotation_manager
            eeg_data = app.server.eeg_data
            sampling_rate = app.server.sampling_rate
            chanlocs = app.server.chanlocs
            sw_events = app.server.sw_events
            channel_indices = app.server.channel_indices
            
            print(f"[DEBUG] Epoch manager has {len(epoch_manager.filtered_epochs_df)} filtered epochs")
            
            # Handle navigation or initial load
            # Always ensure we have a valid epoch index
            if len(epoch_manager.filtered_epochs_df) == 0:
                print("[DEBUG] No epochs available!")
                # No epochs available
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No epochs available", xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
                return (empty_fig, empty_fig, [html.Div("No epochs available")], 
                       "No epochs available", "", 0, [], [])
            
            # Handle triggers
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id']
                print(f"[DEBUG] Trigger ID: {trigger_id}")
                if 'prev-btn' in trigger_id:
                    epoch_manager.prev_epoch()
                elif 'next-btn' in trigger_id:
                    epoch_manager.next_epoch()
                elif 'url' in trigger_id:
                    # Initial page load - show first epoch
                    print("[DEBUG] Initial page load detected")
                    epoch_manager.current_epoch_idx = 0
                elif 'raw-plot' in trigger_id and click_data:
                    # Handle SW event click
                    # Extract event ID from click data if possible
                    # For now, we'll handle this in a separate callback
                    pass
            else:
                # Fallback: ensure we're at a valid epoch index
                print("[DEBUG] No trigger detected, using fallback")
                if epoch_manager.current_epoch_idx >= len(epoch_manager.filtered_epochs_df):
                    epoch_manager.current_epoch_idx = 0
            
            print(f"[DEBUG] Current epoch index: {epoch_manager.current_epoch_idx}")
            
            # Get current epoch data
            print("[DEBUG] Getting current epoch data...")
            current_epoch_data = epoch_manager.get_current_epoch()
            print(f"[DEBUG] Epoch data shape: {current_epoch_data.shape}")
            
            current_epoch_info = epoch_manager.get_current_epoch_info()
            print(f"[DEBUG] Epoch info: {current_epoch_info}")
            
            print(f"[DEBUG] Filtering SW events (total: {len(sw_events)})...")
            current_sw_events = get_sw_events_for_current_epoch(sw_events, epoch_manager)
            print(f"[DEBUG] Filtered SW events: {len(current_sw_events)}")
            
            print("[DEBUG] Creating plots...")
            
            # Create plots
            raw_fig = create_raw_plot(
                current_epoch_data, current_sw_events, annotation_manager, sampling_rate
            )
            print("[DEBUG] Raw plot created")
            
            channel_fig = create_channel_plots(
                current_epoch_data, current_sw_events, annotation_manager,
                channel_indices, sampling_rate
            )
            print("[DEBUG] Channel plots created")
            
            topoplots = create_topoplots_row(
                current_epoch_data, current_sw_events, chanlocs,
                annotation_manager, sampling_rate
            )
            print("[DEBUG] Topoplots created")
            
            # Create topoplots container
            topoplots_children = []
            for i, topo_fig in enumerate(topoplots):
                topoplots_children.append(
                    html.Div([
                        dcc.Graph(figure=topo_fig, style={'display': 'inline-block', 'width': '300px'})
                    ], style={'display': 'inline-block', 'margin': '10px'})
                )
            
            # Status bar
            status_text = f"Epoch {current_epoch_info.get('current_idx', 0) + 1} / {current_epoch_info.get('total_filtered', 0)} | "
            status_text += f"Sleep Stage: {current_epoch_info.get('sleep_stage', '?')} | "
            counts = annotation_manager.get_annotation_count()
            status_text += f"KC: {counts['KC']}, non-KC: {counts['non-KC']}, Unannotated: {counts['unannotated']}"
            
            # Epoch info
            epoch_info_text = f"Epoch {current_epoch_info.get('current_idx', 0) + 1} / {current_epoch_info.get('total_filtered', 0)}"
            
            # Create buttons for each SW event in current epoch (limited to first 20 for UI)
            sw_buttons = []
            if not current_sw_events.empty:
                sw_events_for_buttons = current_sw_events.head(20)  # Limit to 20 buttons
                for _, sw_row in sw_events_for_buttons.iterrows():
                    event_id = sw_row['event_id']
                    annotation_status = annotation_manager.get_annotation_status(event_id)
                    
                    # Determine button color and text
                    if annotation_status == 'KC':
                        btn_color = 'green'
                        btn_text = f'SW {event_id}: KC'
                    elif annotation_status == 'non-KC':
                        btn_color = 'red'
                        btn_text = f'SW {event_id}: non-KC'
                    else:
                        btn_color = 'lightblue'
                        btn_text = f'SW {event_id}: Unannotated'
                    
                    sw_buttons.append(
                        html.Button(
                            btn_text,
                            id={'type': 'sw-annotate-btn', 'index': event_id},
                            n_clicks=0,
                            style={
                                'margin': '3px',
                                'padding': '5px 10px',
                                'backgroundColor': btn_color,
                                'color': 'white' if annotation_status != 'unannotated' else 'black',
                                'border': '1px solid black',
                                'cursor': 'pointer'
                            }
                        )
                    )
                
                if len(current_sw_events) > 20:
                    sw_buttons.append(html.Div(f"(Showing first 20 of {len(current_sw_events)} events)", 
                                               style={'margin': '5px', 'fontStyle': 'italic'}))
            
            print("[DEBUG] Callback completed successfully!")
            return (raw_fig, channel_fig, topoplots_children, status_text, epoch_info_text,
                    current_epoch_info.get('current_idx', 0),
                    current_sw_events.to_dict('records') if not current_sw_events.empty else [],
                    sw_buttons)
        
        except Exception as e:
            print(f"[ERROR] Exception in callback: {e}")
            import traceback
            traceback.print_exc()
            # Return error figure
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper",
                                    x=0.5, y=0.5, showarrow=False)
            error_fig.update_layout(title="Error", height=400)
            return (error_fig, error_fig, [html.Div(f"Error: {str(e)}")], 
                   f"Error: {str(e)}", "", 0, [], [])
    
    @app.callback(
        [Output('raw-plot', 'figure', allow_duplicate=True),
         Output('channel-plots', 'figure', allow_duplicate=True),
         Output('topoplots-container', 'children', allow_duplicate=True),
         Output('sw-event-buttons', 'children', allow_duplicate=True)],
        [Input('raw-plot', 'clickData'),
         Input({'type': 'sw-annotate-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
        [State('sw-events-store', 'data'),
         State({'type': 'sw-annotate-btn', 'index': dash.dependencies.ALL}, 'id')],
        prevent_initial_call=True
    )
    def handle_sw_click(click_data, button_clicks, sw_events_data, button_ids):
        """Handle clicks on SW event markers or annotation buttons"""
        ctx = callback_context
        epoch_manager = app.server.epoch_manager
        annotation_manager = app.server.annotation_manager
        eeg_data = app.server.eeg_data
        sampling_rate = app.server.sampling_rate
        chanlocs = app.server.chanlocs
        sw_events = app.server.sw_events
        channel_indices = app.server.channel_indices
        
        # Get current SW events
        current_sw_events = get_sw_events_for_current_epoch(sw_events, epoch_manager)
        
        if current_sw_events.empty:
            return dash.no_update, dash.no_update, dash.no_update, []
        
        event_id_to_toggle = None
        
        # Check if a button was clicked
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id']
            if 'sw-annotate-btn' in trigger_id:
                # Button was clicked - extract event ID from button ID
                try:
                    # The trigger_id looks like: "{'type': 'sw-annotate-btn', 'index': 123}.n_clicks"
                    # Find which button was clicked by checking button_clicks
                    if button_clicks and button_ids:
                        for idx, (clicks, bid) in enumerate(zip(button_clicks, button_ids)):
                            if clicks and clicks > 0 and bid and bid.get('type') == 'sw-annotate-btn':
                                event_id_to_toggle = bid['index']
                                print(f"[DEBUG CLICK] Button clicked for event {event_id_to_toggle}")
                                break
                except Exception as e:
                    print(f"[DEBUG CLICK] Error parsing button click: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Check if plot was clicked
        if click_data and event_id_to_toggle is None:
            try:
                # click_data structure: {'points': [{'x': time_value, 'y': amplitude, ...}]}
                # Note: clicked_time is already relative to epoch (0-20 seconds)
                clicked_time = click_data['points'][0]['x']  # Time in seconds (relative to epoch)
                relative_sample = int(clicked_time * sampling_rate)  # Sample index within epoch
                
                print(f"[DEBUG CLICK] Plot clicked at time: {clicked_time:.3f}s (relative sample: {relative_sample})")
                print(f"[DEBUG CLICK] Current SW events relative ranges:")
                if not current_sw_events.empty:
                    for _, row in current_sw_events.iterrows():
                        print(f"  Event {row['event_id']}: {row['relative_start']}-{row['relative_stop']}")
                
                # Find overlapping SW events
                matching_events = current_sw_events[
                    (current_sw_events['relative_start'] <= relative_sample) & 
                    (current_sw_events['relative_stop'] >= relative_sample)
                ]
                
                if len(matching_events) > 0:
                    # If multiple events overlap, use the one with the smallest duration (most specific)
                    matching_events = matching_events.copy()
                    matching_events['duration'] = matching_events['relative_stop'] - matching_events['relative_start']
                    selected_event = matching_events.loc[matching_events['duration'].idxmin()]
                    event_id_to_toggle = selected_event['event_id']
                    print(f"[DEBUG CLICK] Found matching event {event_id_to_toggle}")
                else:
                    print(f"[DEBUG CLICK] No SW event found at sample {relative_sample}. Try clicking directly on a colored region or use the buttons below.")
            except (KeyError, IndexError, ValueError) as e:
                print(f"[DEBUG CLICK] Error parsing click data: {e}")
                import traceback
                traceback.print_exc()
        
        # Toggle annotation if we found an event
        if event_id_to_toggle is not None:
            print(f"[DEBUG CLICK] Toggling annotation for event {event_id_to_toggle}...")
            annotation_manager.toggle_annotation(event_id_to_toggle)
            print(f"[DEBUG CLICK] Annotation updated: {annotation_manager.get_annotation_status(event_id_to_toggle)}")
        else:
            # No valid click - return no update
            return dash.no_update, dash.no_update, dash.no_update, []
        
        # Get current epoch data
        current_epoch_data = epoch_manager.get_current_epoch()
        current_sw_events = get_sw_events_for_current_epoch(sw_events, epoch_manager)
        
        # Recreate plots
        raw_fig = create_raw_plot(
            current_epoch_data, current_sw_events, annotation_manager, sampling_rate
        )
        channel_fig = create_channel_plots(
            current_epoch_data, current_sw_events, annotation_manager,
            channel_indices, sampling_rate
        )
        topoplots = create_topoplots_row(
            current_epoch_data, current_sw_events, chanlocs,
            annotation_manager, sampling_rate
        )
        
        # Create topoplots container
        topoplots_children = []
        for topo_fig in topoplots:
            topoplots_children.append(
                html.Div([
                    dcc.Graph(figure=topo_fig, style={'display': 'inline-block', 'width': '300px'})
                ], style={'display': 'inline-block', 'margin': '10px'})
            )
        
        # Recreate buttons with updated annotation status
        sw_buttons = []
        if not current_sw_events.empty:
            sw_events_for_buttons = current_sw_events.head(20)  # Limit to 20 buttons
            for _, sw_row in sw_events_for_buttons.iterrows():
                event_id = sw_row['event_id']
                annotation_status = annotation_manager.get_annotation_status(event_id)
                
                # Determine button color and text
                if annotation_status == 'KC':
                    btn_color = 'green'
                    btn_text = f'SW {event_id}: KC'
                elif annotation_status == 'non-KC':
                    btn_color = 'red'
                    btn_text = f'SW {event_id}: non-KC'
                else:
                    btn_color = 'lightblue'
                    btn_text = f'SW {event_id}: Unannotated'
                
                sw_buttons.append(
                    html.Button(
                        btn_text,
                        id={'type': 'sw-annotate-btn', 'index': event_id},
                        n_clicks=0,
                        style={
                            'margin': '3px',
                            'padding': '5px 10px',
                            'backgroundColor': btn_color,
                            'color': 'white' if annotation_status != 'unannotated' else 'black',
                            'border': '1px solid black',
                            'cursor': 'pointer'
                        }
                    )
                )
            
            if len(current_sw_events) > 20:
                sw_buttons.append(html.Div(f"(Showing first 20 of {len(current_sw_events)} events)", 
                                           style={'margin': '5px', 'fontStyle': 'italic'}))
        
        return raw_fig, channel_fig, topoplots_children, sw_buttons
    
    return app

