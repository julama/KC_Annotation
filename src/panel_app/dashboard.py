import panel as pn
import holoviews as hv
from holoviews import opts, streams
import param
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import warnings
import threading
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for MNE topoplots
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from colorsys import hls_to_rgb

# Suppress HoloViews FutureWarning about pd.unique
# This is an internal HoloViews issue when processing mixed-type data
warnings.filterwarnings('ignore', category=FutureWarning, module='holoviews.core.util')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available. Topoplots will be disabled.")

# Initialize extensions
pn.extension('tabulator', sizing_mode='stretch_width')
hv.extension('bokeh')

# --- CONFIGURATION ---
ANNOTATION_COLORS = {
    'unannotated': '#95a5a6',  # Gray
    'KC': '#27ae60',           # Green  
}

def downsample_minmax(data: np.ndarray, time: np.ndarray, max_points: int = 4000) -> tuple:
    n = len(data)
    if n <= max_points: return time, data
    step = max(1, n // (max_points // 2))
    idx = []
    for i in range(0, n - step, step):
        chunk = data[i:i+step]
        idx.extend([i + np.argmin(chunk), i + np.argmax(chunk)])
    idx = sorted(set(idx))
    return time[idx], data[idx]

def xyz_to_rgb(x, y, z):
    """
    Convert (X, Y, Z) coordinates on the unit sphere into an RGB color.
    From Related_project/clas/helpers/viz.py
    """
    # Compute Hue from X, Z (atan2 maps angle from [-pi, pi])
    hue = -(np.arctan2(z, x) / (2 * np.pi)) % 1  # Normalize to [0, 1]
    
    # Lightness directly from Y (rescale from [-1, 1] to [0, 1])
    lightness = ((1+y)/2)
    
    if hasattr(hue, '__iter__') or hasattr(lightness, '__iter__'):
        return [hls_to_rgb(h, l, 1) for h, l in zip(hue, lightness)]
    else:
        return hls_to_rgb(hue, lightness, 1)  # Saturation fixed at 1

def load_channel_colors(channels_file: str) -> Dict[int, Tuple[float, float, float]]:
    """Load channel colors from channels.csv file."""
    channel_colors = {}
    
    if not Path(channels_file).exists():
        print(f"Warning: Channel file not found: {channels_file}")
        return channel_colors
    
    try:
        df = pd.read_csv(channels_file)
        if 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
            for idx, row in df.iterrows():
                x, y, z = row['X'], row['Y'], row['Z']
                rgb = xyz_to_rgb(x, y, z)
                # Convert to 0-255 range for display
                channel_colors[idx] = tuple(int(c * 255) for c in rgb)
        elif 'nbr' in df.columns:
            # Use nbr column as index
            for _, row in df.iterrows():
                if 'X' in row and 'Y' in row and 'Z' in row:
                    nbr = int(row['nbr']) - 1  # Convert to 0-based index
                    x, y, z = row['X'], row['Y'], row['Z']
                    rgb = xyz_to_rgb(x, y, z)
                    channel_colors[nbr] = tuple(int(c * 255) for c in rgb)
    except Exception as e:
        print(f"Error loading channel colors: {e}")
    
    return channel_colors

# --- KEYBOARD LISTENER COMPONENT ---
class KeyboardListener(pn.reactive.ReactiveHTML):
    """
    Native Panel component that listens to document-level keydown events.
    Invisible but active.
    """
    key = param.String(default="")
    _template = '<div id="kb-listener" style="width:0; height:0; overflow:hidden"></div>'
    _scripts = {
        'render': """
            // Remove previous listener if exists
            if (window.kb_handler) document.removeEventListener('keydown', window.kb_handler);
            
            // Define new handler
            window.kb_handler = (e) => {
                // Ignore if user is typing in a real text input
                if (e.target.matches('input, textarea')) return;
                
                let k = e.key.toLowerCase();
                // Sync specific keys to Python (KC: k/c, unannotated: u/y, delete: d/delete)
                if (['k', 'c', 'u', 'y', 'd', 'delete'].includes(k)) { 
                    data.key = k; 
                }
            };
            
            // Attach to document
            document.addEventListener('keydown', window.kb_handler);
        """
    }

# Box selection will use BoundsXY stream from HoloViews

# --- DASHBOARD CLASS ---
class EEGDashboard(param.Parameterized):
    # Reactive Params
    epoch_index = param.Integer(default=0, bounds=(0, None))
    selected_region_id = param.Integer(default=-1)
    current_label = param.Selector(objects=['KC', 'unannotated'], default='unannotated')
    update_trigger = param.Integer(default=0)
    focus_plot_trigger = param.Integer(default=0)  # Separate trigger for focus plots (only on epoch/region changes)
    topoplot_popup = param.Parameter(default=None)  # For storing topoplot popup
    topoplot_trigger = param.Integer(default=0) # Trigger for delayed topoplot generation
    
    def __init__(self, epoch_manager, annotation_manager, 
                 focus_channels: List[int] = None,
                 main_plot_channels: List[int] = None,
                 chanlocs: pd.DataFrame = None, 
                 channels_file: Optional[str] = None,
                 exclude_channels: List[int] = None, **params):
        
        # --- FIX: Set attributes BEFORE super().__init__ ---
        # This prevents "AttributeError" if watchers fire during init
        self.epoch_manager = epoch_manager
        self.annotation_manager = annotation_manager
        self.focus_channels = focus_channels or [34, 55, 70]
        self.main_plot_channels = main_plot_channels
        self.exclude_channels = exclude_channels or []
        self.sampling_rate = epoch_manager.sampling_rate
        self.chanlocs = chanlocs if chanlocs is not None else pd.DataFrame()
        
        # Load channel colors
        if channels_file is None:
            # Try to find channels.csv in Data directory
            possible_paths = [
                Path("Data/channels.csv"),
                Path("Data/EPISL_01_W1/channels.csv"),
            ]
            for path in possible_paths:
                if path.exists():
                    channels_file = str(path)
                    break
        
        self.channel_colors = load_channel_colors(channels_file) if channels_file else {}
        
        # Topoplot cache: {epoch_index: {region_id: base64_image}}
        self._topoplot_cache = {}
        self._current_epoch_for_cache = -1
        
        # Debounce mechanism for update_trigger
        self._update_timer = None
        self._pending_update = False
        
        # Debounce for topoplot
        self._topoplot_timer = None
        
        # Cache for curves to avoid recreation when epoch unchanged
        self._cached_curves = None
        self._cached_epoch_index = -1
        self._cached_time_axis = None
        self._cached_y_range = (-200, 200)
        
        # Cache for epoch_data to avoid redundant calls to _get_epoch_data()
        self._cached_epoch_data = None
        self._cached_epoch_data_index = -1
        self._cached_regions = None
        self._cached_epoch_start = None
        
        # Cache for focus plots to avoid recreation when only selection changes
        self._cached_focus_plots = {}  # Maps channel_idx -> plot
        self._cached_focus_epoch_index = -1
        self._cached_focus_selected_region = -1
        
        # Click handler debouncing
        self._last_click_time = 0
        self._last_click_region = None
        self._click_debounce_time = 0.3  # 300ms debounce for clicks
        self._processing_click = False
        
        # Now call super, which might trigger watchers immediately
        super().__init__(**params)
        
        self.param.epoch_index.bounds = (0, max(0, epoch_manager.get_epoch_count() - 1))
        
        # Initialize Tap stream for selecting existing regions
        self.tap_stream = streams.Tap(transient=True)
        self.tap_stream.add_subscriber(self._on_plot_click)

        # Initialize Keyboard Listener
        self.kb_listener = KeyboardListener()
        self.kb_listener.param.watch(self._handle_kb_event, 'key')
        
        # Watch selected_region_id to trigger delayed topoplot update
        self.param.watch(self._schedule_topoplot_update, 'selected_region_id')

    def _debounced_update(self, delay=0.01):
        """Debounced update trigger to prevent rapid-fire plot recreations."""
        import time as time_module
        
        # Cancel any pending update
        if self._update_timer is not None:
            self._update_timer.cancel()
        
        # Mark that we have a pending update
        self._pending_update = True
        update_request_time = time_module.time()
        
        # Schedule update after delay (reduced to 10ms for faster response)
        def do_update():
            import time as time_module
            if self._pending_update:
                t_before = time_module.time()
                self.update_trigger += 1
                t_after = time_module.time()
                print(f"⏱️ update_trigger incremented: delay={((t_before-update_request_time)*1000):.1f}ms, increment_time={((t_after-t_before)*1000):.1f}ms")
                self._pending_update = False
            self._update_timer = None
        
        self._update_timer = threading.Timer(delay, do_update)
        self._update_timer.daemon = True
        self._update_timer.start()

    def _schedule_topoplot_update(self, event=None):
        """Schedule a delayed update for the topoplot to avoid blocking UI."""
        if self._topoplot_timer is not None:
            self._topoplot_timer.cancel()
            
        def do_update():
            self.topoplot_trigger += 1
            
        # 300ms delay to allow UI to update first
        self._topoplot_timer = threading.Timer(0.3, do_update)
        self._topoplot_timer.daemon = True
        self._topoplot_timer.start()

    def _get_epoch_data(self):
        """Get current epoch data and regions (uses cache to avoid redundant calls)."""
        # Check if we already have cached epoch_data for this epoch_index
        epoch_data_cached = (self._cached_epoch_data_index == self.epoch_index and 
                            self._cached_epoch_data is not None)
        
        if epoch_data_cached and self._cached_regions is not None:
            # Both epoch_data and regions are cached
            return self._cached_epoch_data, self._cached_regions, self._cached_epoch_start
        
        # Need to fetch epoch_data (cache miss or epoch changed)
        if not epoch_data_cached:
            self.epoch_manager.current_epoch_idx = self.epoch_index
            epoch_data = self.epoch_manager.get_current_epoch()
            epoch_info = self.epoch_manager.get_current_epoch_info()
            
            # Cache epoch_data
            self._cached_epoch_data = epoch_data
            self._cached_epoch_data_index = self.epoch_index
            self._cached_epoch_start = int(epoch_info['start_idx'])
        else:
            # Epoch_data cached, but regions need refresh
            epoch_info = self.epoch_manager.get_current_epoch_info()
            epoch_data = self._cached_epoch_data
        
        # Get regions for current epoch (always refresh if not cached)
        if self._cached_regions is None:
            epoch_start = int(epoch_info['start_idx'])
            epoch_end = int(epoch_info['end_idx'])
            current_regions = self.annotation_manager.get_regions_for_epoch(epoch_start, epoch_end)
            self._cached_regions = current_regions
        else:
            current_regions = self._cached_regions
        
        return epoch_data, current_regions, self._cached_epoch_start
    
    def _create_time_axis(self, n): 
        return np.arange(n) / self.sampling_rate

    # --- INTERACTION LOGIC ---
    def _on_box_select(self, bounds):
        """Handle box selection (drag to select region)."""
        if bounds is None:
            return
        
        # Prevent recursive calls
        if not hasattr(self, '_processing_box_select'):
            self._processing_box_select = False
        
        if self._processing_box_select:
            return
        
        # Prevent processing if bounds haven't actually changed (stale callback)
        if hasattr(self, '_last_bounds') and self._last_bounds == bounds:
            return
        
        self._processing_box_select = True
        
        try:
            # BoundsXY returns (x0, y0, x1, y1) tuple
            try:
                x0, y0, x1, y1 = bounds
            except (TypeError, ValueError):
                return
            
            # Use time range (x-axis) only, ignore y-axis
            start_time = min(x0, x1)
            end_time = max(x0, x1)
            
            # Get epoch data
            epoch_data, _, epoch_start_idx = self._get_epoch_data()
            if epoch_data is None:
                return
            
            # Convert time to sample indices (relative to epoch)
            start_idx = int(start_time * self.sampling_rate)
            stop_idx = int(end_time * self.sampling_rate)
            
            # Clamp to epoch boundaries
            start_idx = max(0, min(start_idx, len(epoch_data) - 1))
            stop_idx = max(0, min(stop_idx, len(epoch_data) - 1))
            
            if start_idx >= stop_idx:
                print("⚠️ Invalid selection: start >= stop")
                return
            
            print(f"📦 Box selected: {start_time:.3f}s - {end_time:.3f}s (samples {start_idx}-{stop_idx})")
            
            # Store last bounds to prevent duplicate processing
            self._last_bounds = bounds
            
            # Add new region
            region_id = self.annotation_manager.add_region(start_idx, stop_idx, epoch_start_idx)
            self.selected_region_id = region_id
            
            # Invalidate regions cache (epoch_data cache can stay)
            self._cached_regions = None
            # Invalidate focus plot cache (regions changed)
            self._cached_focus_plots = {}
            
            # REMOVED: Synchronous topoplot generation to improve responsiveness
            # The topoplot will be generated asynchronously via _schedule_topoplot_update
            
            # Update label based on annotation status
            status = self.annotation_manager.get_annotation_status(region_id)
            self.current_label = status
            
            # Trigger update (this will cause create_main_plot to be called, but bounds won't be processed)
            self._debounced_update()
        finally:
            self._processing_box_select = False

    def _on_plot_click(self, x, y):
        """Handle clicks on the plot (select existing region)."""
        import time as time_module
        
        if x is None:
            return
        
        # Debounce: ignore rapid repeated clicks on the same region
        current_time = time_module.time()
        click_time = x
        click_sample = int(click_time * self.sampling_rate)
        
        # Check if this is a duplicate click
        if (self._last_click_region is not None and 
            abs(click_time - self._last_click_time) < self._click_debounce_time and
            abs(click_sample - self._last_click_sample) < 10):  # Within 10 samples
            return  # Ignore duplicate click
        
        # Prevent concurrent processing
        if self._processing_click:
            return
        
        self._processing_click = True
        
        try:
            t0 = time_module.time()
            
            _, current_regions, _ = self._get_epoch_data()
            t1 = time_module.time()
            
            if not current_regions:
                return

            # Find region that contains the click point
            t2 = time_module.time()
            for region_id, region in current_regions.items():
                rel_start = region.get('relative_start', region['start_idx'])
                rel_stop = region.get('relative_stop', region['stop_idx'])
                
                if rel_start <= click_sample <= rel_stop:
                    # Check if already selected (avoid unnecessary updates)
                    if self.selected_region_id == region_id:
                        return
                    
                    print(f"🎯 Clicked region: {region_id} (get_epoch_data: {(t1-t0)*1000:.1f}ms, find_region: {(t2-t1)*1000:.1f}ms)")
                    
                    t3 = time_module.time()
                    self.selected_region_id = region_id
                    
                    # REMOVED: Synchronous topoplot generation to improve responsiveness
                    # The topoplot will be generated asynchronously via _schedule_topoplot_update
                    
                    # Sync label
                    status = self.annotation_manager.get_annotation_status(region_id)
                    self.current_label = status
                    t6 = time_module.time()
                    
                    # Store click info for debouncing
                    self._last_click_time = click_time
                    self._last_click_region = region_id
                    self._last_click_sample = click_sample
                    
                    print(f"   Timing: set_selected={((t3-t2)*1000):.1f}ms, label={((t6-t3)*1000):.1f}ms")
                    
                    click_complete_time = time_module.time()
                    self._debounced_update()
                    t7 = time_module.time()
                    print(f"   Total click handler: {((t7-t0)*1000):.1f}ms (debounce scheduled at {((t7-click_complete_time)*1000):.1f}ms)")
                    
                    # Store click time to compare with plot creation
                    if not hasattr(self, '_click_times'):
                        self._click_times = []
                    self._click_times.append(click_complete_time)
                    if len(self._click_times) > 10:
                        self._click_times.pop(0)
                    
                    return
        finally:
            self._processing_click = False

    def _handle_kb_event(self, event):
        """Handle keyboard input from the ReactiveHTML component."""
        key = event.new
        if not key or self.selected_region_id == -1:
            return
        
        print(f"⌨️ Key Press: {key}")
        
        # Map keys to actions
        if key in ['k', 'c']:
            # Mark as KC
            self.annotation_manager.set_annotation(self.selected_region_id, True)
            self.current_label = 'KC'
        elif key in ['u', 'y']:
            # Mark as unannotated
            self.annotation_manager.set_annotation(self.selected_region_id, False)
            self.current_label = 'unannotated'
        elif key in ['d', 'delete']:
                # Delete region
                if self.annotation_manager.delete_region(self.selected_region_id):
                    print(f"🗑️ Deleted region {self.selected_region_id}")
                    self.selected_region_id = -1
                    self.current_label = 'unannotated'
                    # Invalidate regions cache
                    self._cached_regions = None
                    # Invalidate focus plot cache (regions changed)
                    self._cached_focus_plots = {}
                    self.focus_plot_trigger += 1  # Update focus plots
                    self._debounced_update()
        
        # Reset listener so repeated keys work
        self.kb_listener.key = ""

    @param.depends('current_label', watch=True)
    def _on_label_change(self):
        """Update annotation when label changes (via Key or UI)."""
        if self.selected_region_id == -1:
            return
        
        if self.current_label == 'KC':
            self.annotation_manager.set_annotation(self.selected_region_id, True)
        else:
            self.annotation_manager.set_annotation(self.selected_region_id, False)
        
        # Invalidate regions cache (annotation status changed)
        self._cached_regions = None
        # Invalidate focus plot cache (regions changed)
        self._cached_focus_plots = {}
        self.focus_plot_trigger += 1  # Update focus plots
        self._debounced_update()

    # --- STATUS BAR ---
    @param.depends('epoch_index', 'update_trigger')
    def status_bar(self):
        """Create reactive status bar with epoch info and annotation counts."""
        import time as time_module
        t0 = time_module.time()
        
        epoch_info = self.epoch_manager.get_current_epoch_info()
        sleep_stage = epoch_info.get('sleep_stage', '?')
        current_idx = epoch_info.get('current_idx', 0)
        total_filtered = epoch_info.get('total_filtered', 0)
        
        # Get regions for current epoch (only user-selected regions)
        _, current_regions, _ = self._get_epoch_data()
        t1 = time_module.time()
        regions_in_epoch = len(current_regions)
        
        # Get annotation counts (only user-selected regions)
        counts = self.annotation_manager.get_annotation_count()
        total_regions = len(self.annotation_manager.get_all_regions())
        
        status_text = (
            f"**Epoch**: {current_idx + 1}/{total_filtered} | "
            f"**Sleep Stage**: {sleep_stage} | "
            f"**Regions in Epoch**: {regions_in_epoch} | "
            f"**Total Regions**: {total_regions} | "
            f"**KC**: {counts['KC']} | **Unannotated**: {counts['unannotated']}"
        )
        
        t2 = time_module.time()
        result = pn.pane.Markdown(
            status_text,
            styles={'font-size': '12px', 'padding': '5px 10px', 'background': '#ecf0f1', 'border-radius': '3px', 'margin': '0px'}
        )
        t3 = time_module.time()
        
        total_time = (t3 - t0) * 1000
        if total_time > 50:  # Log if > 50ms
            print(f"⏱️ status_bar: get_epoch_data={((t1-t0)*1000):.1f}ms, create_markdown={((t2-t1)*1000):.1f}ms, TOTAL={total_time:.1f}ms")
        
        return result

    # --- TOPOPLOT GENERATION ---
    def _create_topoplot_image(self, region_id: int, epoch_data: pd.DataFrame, 
                               start_idx: int, stop_idx: int) -> Optional[str]:
        """Create base64-encoded topoplot image for a selected region."""
        if not MNE_AVAILABLE:
            return None
        
        try:
            # Extract data window
            if start_idx >= len(epoch_data) or stop_idx >= len(epoch_data):
                return None
            region_data = epoch_data.iloc[start_idx:stop_idx+1].copy()
            
            # Apply bandpass filter (0.5-2 Hz for slow waves)
            from src.preprocessing import apply_bandpass_filter
            filtered_data = apply_bandpass_filter(region_data, 0.5, 2.0, self.sampling_rate)
            
            # Average across time window
            mean_data = filtered_data.mean(axis=0).values
            
            # Get channel positions
            if not self.chanlocs.empty:
                if 'X' in self.chanlocs.columns and 'Y' in self.chanlocs.columns:
                    # Rotate 90 degrees counter-clockwise: X (Nose) points Up instead of Right
                    # New X = -Old Y, New Y = Old X
                    pos = np.array([-self.chanlocs['Y'].values, self.chanlocs['X'].values]).T
                elif 'x' in self.chanlocs.columns and 'y' in self.chanlocs.columns:
                    pos = np.array([-self.chanlocs['y'].values, self.chanlocs['x'].values]).T
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
            
            # Filter out excluded channels from topoplot
            if self.exclude_channels:
                n_chans = len(mean_data)
                # Create mask: True for channels to KEEP
                mask = np.ones(n_chans, dtype=bool)
                for idx in self.exclude_channels:
                    if 0 <= idx < n_chans:
                        mask[idx] = False
                
                mean_data = mean_data[mask]
                pos = pos[mask]
            
            # Create topoplot using MNE
            fig, ax = plt.subplots(figsize=(3, 3))
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
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            
            return f'data:image/png;base64,{img_str}'
        except Exception as e:
            print(f"[ERROR TOPO] Error creating topoplot for region {region_id}: {e}")
            return None

    def _show_topoplot_popup(self, region_id: int, epoch_data: pd.DataFrame, 
                            start_idx: int, stop_idx: int):
        """Show topoplot as popup for selected region."""
        # Check cache first
        if (self.epoch_index in self._topoplot_cache and 
            region_id in self._topoplot_cache[self.epoch_index]):
            topo_img = self._topoplot_cache[self.epoch_index][region_id]
        else:
            topo_img = self._create_topoplot_image(region_id, epoch_data, start_idx, stop_idx)
            if topo_img:
                # Cache it
                if self.epoch_index not in self._topoplot_cache:
                    self._topoplot_cache[self.epoch_index] = {}
                self._topoplot_cache[self.epoch_index][region_id] = topo_img
        
        if topo_img:
            # Create popup HTML
            status = self.annotation_manager.get_annotation_status(region_id)
            popup_html = f"""
            <div style="padding: 10px; background: white; border: 2px solid #3498db; border-radius: 5px;">
                <h4>Region {region_id} - {status}</h4>
                <img src="{topo_img}" style="width: 300px; height: 300px; display: block; margin: 10px auto;">
                <p style="text-align: center; font-size: 12px;">Time: {start_idx/self.sampling_rate:.3f}s - {stop_idx/self.sampling_rate:.3f}s</p>
            </div>
            """
            self.topoplot_popup = pn.pane.HTML(popup_html, width=350, height=400)
        else:
            self.topoplot_popup = None

    # --- PLOTTING ---
    def _get_topoplot_on_demand(self, region_id: int, epoch_data: pd.DataFrame, 
                                rel_start: int, rel_stop: int) -> str:
        """Get topoplot image on-demand, checking cache first, generating if needed."""
        if not MNE_AVAILABLE or epoch_data is None:
            return ''
        
        # Check cache first
        if (self.epoch_index in self._topoplot_cache and 
            region_id in self._topoplot_cache[self.epoch_index]):
            topo_img = self._topoplot_cache[self.epoch_index][region_id]
            return f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'
        
        # Generate topoplot
        topo_img = self._create_topoplot_image(region_id, epoch_data, rel_start, rel_stop)
        if topo_img:
            # Cache it
            if self.epoch_index not in self._topoplot_cache:
                self._topoplot_cache[self.epoch_index] = {}
            self._topoplot_cache[self.epoch_index][region_id] = topo_img
            return f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'
        
        return ''
    
    @param.depends('topoplot_trigger')
    def topoplot_panel(self):
        """Reactive panel for displaying topoplot of selected region."""
        if self.selected_region_id == -1:
            return pn.Spacer(width=200, height=200)
        
        # Get epoch data (cached)
        epoch_data, current_regions, _ = self._get_epoch_data()
        
        if not current_regions or self.selected_region_id not in current_regions:
             return pn.Spacer(width=200, height=200)
        
        region = current_regions[self.selected_region_id]
        rel_start = region.get('relative_start', region['start_idx'])
        rel_stop = region.get('relative_stop', region['stop_idx'])
        
        html = self._get_topoplot_on_demand(self.selected_region_id, epoch_data, rel_start, rel_stop)
        
        if not html:
             return pn.pane.Markdown("Topoplot unavailable", styles={'font-size': '12px', 'color': 'gray'})
             
        return pn.pane.HTML(html, width=200, height=200)

    def _create_hover_tool_with_callback(self, epoch_data: Optional[pd.DataFrame]):
        """Create HoverTool - topoplots shown if cached, otherwise basic info."""
        from bokeh.models import HoverTool
        
        # Store epoch_data reference for potential future use
        if epoch_data is not None:
            if not hasattr(self, '_hover_epoch_data'):
                self._hover_epoch_data = {}
            self._hover_epoch_data[self.epoch_index] = epoch_data
        
        hover = HoverTool(
            tooltips="""
            <div style="font-size: 12px;">
                <strong>Region @region_id</strong><br>
                Status: @status<br>
                Time: @x0{0.00}s - @x1{0.00}s<br>
                @topo_html{safe}
            </div>
            """,
            point_policy='follow_mouse',
            attachment='above'
        )
        
        return hover
    
    def _create_selected_regions(self, regions: dict, y_range: Tuple[float, float], 
                                epoch_data: Optional[pd.DataFrame] = None):
        """Create rectangles for selected regions."""
        if not regions:
            return hv.Rectangles([], kdims=['x0', 'y0', 'x1', 'y1'])
        
        y_min, y_max = y_range
        rect_data = []
        
        # Initialize cache if epoch changed (but don't generate topoplots yet - lazy loading)
        if epoch_data is not None and MNE_AVAILABLE:
            # Clear cache if epoch changed
            if self._current_epoch_for_cache != self.epoch_index:
                self._topoplot_cache = {}
                self._current_epoch_for_cache = self.epoch_index
            
            # Cache structure for this epoch
            if self.epoch_index not in self._topoplot_cache:
                self._topoplot_cache[self.epoch_index] = {}
        
        for region_id, region in regions.items():
            # Get relative indices for display
            rel_start = region.get('relative_start', region['start_idx'])
            rel_stop = region.get('relative_stop', region['stop_idx'])
            
            status = self.annotation_manager.get_annotation_status(region_id)
            is_sel = (region_id == self.selected_region_id)
            
            # Color scheme: show label color even when selected, with brighter yellow outline
            if status == 'KC':
                fill = ANNOTATION_COLORS['KC']
                base_lc = ANNOTATION_COLORS['KC']
            else:
                fill = ANNOTATION_COLORS['unannotated']
                base_lc = ANNOTATION_COLORS['unannotated']
            
            if is_sel:
                # When selected: use label color for fill, bright yellow for outline
                lc = '#FFD700'  # Bright gold/yellow outline
                lw = 4  # Thicker border when selected
                alpha = 0.25  # Slightly more visible when selected
            else:
                lc = base_lc
                lw = 2
                alpha = 0.15  # Transparent so EEG shows through
            
            # Convert sample indices to time
            start_time = rel_start / self.sampling_rate
            stop_time = rel_stop / self.sampling_rate
            
            # Check if topoplot is cached (lazy loading - only show if already generated)
            topo_html = ''
            if epoch_data is not None and MNE_AVAILABLE:
                if (self.epoch_index in self._topoplot_cache and 
                    region_id in self._topoplot_cache[self.epoch_index]):
                    topo_img = self._topoplot_cache[self.epoch_index][region_id]
                    topo_html = f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'
            
            # Store region metadata for lazy topoplot generation
            rect_data.append({
                'x0': start_time, 'y0': y_min,
                'x1': stop_time, 'y1': y_max,
                'region_id': region_id, 'status': status, 'fill_color': fill, 
                'line_color': lc, 'line_width': lw, 'alpha': alpha,
                'topo_html': topo_html,  # Show if cached, empty otherwise
                'rel_start': rel_start,  # Store for lazy generation
                'rel_stop': rel_stop  # Store for lazy generation
            })
        
        df_rects = pd.DataFrame(rect_data)
        rects = hv.Rectangles(df_rects, kdims=['x0','y0','x1','y1'], 
                             vdims=['region_id','status','fill_color','line_color','line_width','alpha','topo_html','rel_start','rel_stop'])
        
        # Create hover tool - topoplots will be generated on-demand via hook
        hover = self._create_hover_tool_with_callback(epoch_data)
        
        return rects.opts(
            color='fill_color', 
            line_color='line_color', 
            line_width='line_width', 
            alpha='alpha', 
            tools=[hover, 'tap'],
            nonselection_alpha='alpha',
            nonselection_line_alpha=0.9,
            nonselection_color='fill_color',
            nonselection_line_color='line_color'
        )

    def _get_channel_color(self, channel_idx: int) -> str:
        """Get color for a channel, using channel colors if available."""
        if channel_idx in self.channel_colors:
            r, g, b = self.channel_colors[channel_idx]
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#34495e'  # Default dark gray
    
    @param.depends('epoch_index', 'update_trigger')
    def create_main_plot(self, bounds=None, **kwargs):
        """Create butterfly plot with all channels overlaid."""
        import time as time_module
        t0 = time_module.time()
        plot_start_time = t0
        
        # Clear last bounds when epoch changes to allow new selections
        if hasattr(self, '_last_bounds'):
            delattr(self, '_last_bounds')
        
        # Clear epoch_data cache if epoch changed
        if self._cached_epoch_data_index != self.epoch_index:
            self._cached_epoch_data = None
            self._cached_regions = None
            self._cached_epoch_start = None
            # Also clear focus plot cache
            self._cached_focus_plots = {}
            self._cached_focus_epoch_index = -1
            # Trigger focus plot update
            self.focus_plot_trigger += 1
        
        t1 = time_module.time()
        epoch_data, current_regions, _ = self._get_epoch_data()
        t2 = time_module.time()
        
        if epoch_data is None or len(epoch_data) == 0:
            if epoch_data is None:
                print(f"⚠️ epoch_data is None in create_main_plot")
            else:
                print(f"⚠️ Empty epoch_data in create_main_plot")
            return hv.Spacer()
        
        # Cache curves if epoch hasn't changed (only regions changed)
        epoch_changed = (self._cached_epoch_index != self.epoch_index)
        
        if epoch_changed or self._cached_curves is None:
            # Recreate curves when epoch changes
            t = self._create_time_axis(len(epoch_data))
            self._cached_time_axis = t
            
            # Butterfly plot: Subset of channels for performance
            curves = []
            first_curve = None
            
            # Track min/max for dynamic scaling
            y_data_min = 0
            y_data_max = 0
            
            # Determine which channels to plot
            if self.main_plot_channels is not None:
                # Use configured channels, filtering out any that are out of bounds
                n_total = len(epoch_data.columns)
                indices = [i for i in self.main_plot_channels if 0 <= i < n_total]
                if not indices:
                    print("Warning: No valid channels in main_plot_channels. Falling back to default.")
                    indices = list(range(min(30, n_total)))
            else:
                # Limit to ~30 channels max to keep UI responsive
                n_channels = len(epoch_data.columns)
                target_n_channels = 30
                if n_channels > target_n_channels:
                    step = max(1, n_channels // target_n_channels)
                    indices = list(range(0, n_channels, step))
                else:
                    indices = list(range(n_channels))
            
            for i_idx, i in enumerate(indices):
                col = epoch_data.columns[i]
                d = epoch_data[col].values
                # Normalize but don't offset (all on same scale)
                d_normalized = (d - np.mean(d)) / (np.std(d) or 1) * 30
                
                # Update min/max
                current_min = np.min(d_normalized)
                current_max = np.max(d_normalized)
                
                if i_idx == 0:
                    y_data_min = current_min
                    y_data_max = current_max
                else:
                    y_data_min = min(y_data_min, current_min)
                    y_data_max = max(y_data_max, current_max)
                
                tt, dd = downsample_minmax(d_normalized, t)
                color = self._get_channel_color(i)
                
                if i_idx == 0:
                    # First curve: enable box_select and store reference
                    first_curve = hv.Curve((tt, dd), label=f'Ch {i}').opts(
                        color=color,
                        line_width=1,
                        alpha=0.7,
                        tools=['box_select', 'tap', 'hover']
                    )
                    curves.append(first_curve)
                else:
                    curve = hv.Curve((tt, dd), label=f'Ch {i}').opts(
                        color=color,
                        line_width=1,
                        alpha=0.7
                    )
                    curves.append(curve)
            
            # Calculate dynamic range with padding and clipping
            y_pad = (y_data_max - y_data_min) * 0.1
            if y_pad == 0: y_pad = 10
            
            y_limit_min = max(-300, y_data_min - y_pad)
            y_limit_max = min(300, y_data_max + y_pad)
            
            # Cache curves and epoch index
            self._cached_curves = curves
            self._cached_epoch_index = self.epoch_index
            self._cached_y_range = (y_limit_min, y_limit_max)
        else:
            # Reuse cached curves
            curves = self._cached_curves
            first_curve = curves[0] if curves else None
            y_limit_min, y_limit_max = self._cached_y_range
        
        # Update bounds stream source to first curve for box selection
        # Always update to ensure it's connected to the current plot
        if first_curve is not None:
            if not hasattr(self, '_bounds_stream') or self._bounds_stream is None:
                # Create bounds stream if it doesn't exist
                self._bounds_stream = streams.BoundsXY(source=first_curve)
                self._bounds_stream.add_subscriber(self._on_box_select)
            else:
                # Update source to current first curve - this reconnects the stream
                try:
                    self._bounds_stream.source = first_curve
                except Exception as e:
                    # If update fails, recreate the stream
                    print(f"Warning: Failed to update bounds stream source: {e}")
                    self._bounds_stream = streams.BoundsXY(source=first_curve)
                    self._bounds_stream.add_subscriber(self._on_box_select)
        
        # Dynamic y_range for butterfly plot
        y_range = (y_limit_min, y_limit_max)
        
        t3 = time_module.time()
        # Add region rectangles
        regions_rects = self._create_selected_regions(current_regions, y_range, epoch_data)
        t4 = time_module.time()
        
        # Create combined plot
        combined = hv.Overlay(curves) * regions_rects
        t5 = time_module.time()
        
        # Performance logging
        total_time = (t5 - t0) * 1000
        
        # Calculate time since last click (if available)
        time_since_click = None
        if hasattr(self, '_click_times') and self._click_times:
            time_since_click = (plot_start_time - self._click_times[-1]) * 1000
        
        if total_time > 50 or True:  # Always log for debugging
            click_info = f", time_since_click={time_since_click:.1f}ms" if time_since_click is not None else ""
            print(f"⏱️ create_main_plot START: cache_check={((t1-t0)*1000):.1f}ms, get_epoch_data={((t2-t1)*1000):.1f}ms, curves={((t3-t2)*1000):.1f}ms, regions={((t4-t3)*1000):.1f}ms, overlay={((t5-t4)*1000):.1f}ms, TOTAL={total_time:.1f}ms{click_info}")
        
        # Store plot creation time for tracking
        if not hasattr(self, '_plot_creation_times'):
            self._plot_creation_times = []
        self._plot_creation_times.append((plot_start_time, total_time))
        # Keep only last 10
        if len(self._plot_creation_times) > 10:
            self._plot_creation_times.pop(0)
        
        # Set plot options with y_range customization hook
        def set_y_range(plot, element):
            """Hook to set y-axis range dynamically"""
            # Access the y_range directly from handles
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range_handle = plot.handles['y_range']
                y_range_handle.start = y_limit_min
                y_range_handle.end = y_limit_max
        
        # Hook to ensure box_select tool is enabled after plot updates
        def ensure_box_select(plot, element):
            """Ensure box_select tool is enabled on the plot"""
            if hasattr(plot, 'state') and hasattr(plot.state, 'toolbar'):
                # Find box_select tool and ensure it's active
                for tool in plot.state.toolbar.tools:
                    if hasattr(tool, 'name') and tool.name == 'box_select':
                        if not hasattr(tool, 'active') or not tool.active:
                            tool.active = True
                        break
        
        # Hook to optimize selection tool (hit-testing)
        def optimize_selection(plot, element):
            """
            Hook to restrict BoxSelectTool to only the first renderer.
            This ensures hit-testing is done against only 1 curve instead of all of them,
            drastically improving performance when many channels are displayed.
            """
            from bokeh.models import BoxSelectTool
            
            # Find the BoxSelectTool
            box_select = None
            if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                for tool in plot.state.tools:
                    if isinstance(tool, BoxSelectTool):
                        box_select = tool
                        break
            
            # Restrict renderers to the first one (first curve)
            if box_select and hasattr(plot, 'handles') and 'glyph_renderers' in plot.handles:
                renderers = plot.handles['glyph_renderers']
                if renderers:
                    # The first renderer corresponds to the first curve added to the overlay
                    box_select.renderers = [renderers[0]]
        
        plot_opts = opts.Overlay(
            width=1200, height=500, shared_axes=True, show_legend=False, 
            tools=['tap', 'hover', 'xwheel_zoom', 'xpan','box_select'], 
            active_tools=['tap', 'box_select'],
            hooks=[set_y_range, ensure_box_select, optimize_selection],
            #ylim=(y_limit_min, y_limit_max),
            #framewise=True
        )
        
        return combined.opts(plot_opts)s

    @param.depends('epoch_index', 'focus_plot_trigger')
    def create_focus_plot(self, channel_idx, **kwargs):
        """Create focus channel plot (cached when epoch and selection unchanged)."""
        import time as time_module
        t0 = time_module.time()
        
        # Check cache: reuse if epoch and selected_region_id haven't changed
        cache_key = (self.epoch_index, self.selected_region_id)
        if (self._cached_focus_epoch_index == self.epoch_index and 
            channel_idx in self._cached_focus_plots and
            cache_key == getattr(self, '_cached_focus_cache_key', None)):
            cached_plot = self._cached_focus_plots[channel_idx]
            t1 = time_module.time()
            if (t1 - t0) * 1000 > 1:  # Only log if cache check took time
                print(f"⏱️ create_focus_plot Ch{channel_idx}: CACHED ({(t1-t0)*1000:.1f}ms)")
            return cached_plot
        
        epoch_data, current_regions, _ = self._get_epoch_data()
        t1 = time_module.time()
        
        if epoch_data is None:
            return hv.Spacer()
        if channel_idx >= len(epoch_data.columns):
            return hv.Spacer()
        
        d = epoch_data.iloc[:, channel_idx].values
        t = self._create_time_axis(len(d))
        tt, dd = downsample_minmax(d, t)
        color = self._get_channel_color(channel_idx)
        # Use fixed y_range of -200 to +200 for focus plots
        y_range = (-200, 200)
        curve = hv.Curve((tt, dd)).opts(color=color, line_width=1.5)
        regions_rects = self._create_selected_regions(current_regions, y_range, epoch_data)
        
        # Hook to set y-axis range
        def set_y_range_focus(plot, element):
            """Hook to set y-axis range to -200 to +200 for focus plots"""
            # Access the y_range directly from handles
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range = plot.handles['y_range']
                y_range.start = -200
                y_range.end = 200
        
        t2 = time_module.time()
        result = (curve * regions_rects).opts(
            opts.Overlay(width=1200, height=150, shared_axes=True, xaxis=None, ylabel=f"Ch {channel_idx}", hooks=[set_y_range_focus])
        )
        t3 = time_module.time()
        
        # Cache the result
        self._cached_focus_plots[channel_idx] = result
        self._cached_focus_epoch_index = self.epoch_index
        self._cached_focus_cache_key = cache_key
        
        total_time = (t3 - t0) * 1000
        if total_time > 50 or True:  # Always log for debugging
            print(f"⏱️ create_focus_plot Ch{channel_idx}: get_epoch_data={((t1-t0)*1000):.1f}ms, plot_creation={((t2-t1)*1000):.1f}ms, opts={((t3-t2)*1000):.1f}ms, TOTAL={total_time:.1f}ms")
        
        return result

    def view(self) -> pn.Column:
        """Create the dashboard view."""
        btn_prev = pn.widgets.Button(name='◀ Prev', width=80)
        btn_next = pn.widgets.Button(name='Next ▶', width=80)
        btn_prev.on_click(lambda e: setattr(self, 'epoch_index', max(0, self.epoch_index - 1)))
        btn_next.on_click(lambda e: setattr(self, 'epoch_index', self.epoch_index + 1))
        
        # Radio Buttons (only KC and unannotated)
        radio_group = pn.widgets.RadioButtonGroup(
            name='Annotation', options=['KC', 'unannotated'], 
            button_type='default', value=self.current_label
        )
        self.param.watch(lambda e: setattr(radio_group, 'value', e.new), 'current_label')
        radio_group.param.watch(lambda e: setattr(self, 'current_label', e.new), 'value')
        
        # Removed view mode toggle - only butterfly plot now

        info = pn.bind(
            lambda rid: pn.pane.Markdown(f"**Selected Region:** {rid}" if rid != -1 else "**No Region Selected**", 
                                        styles={'font-size': '12px', 'margin': '0px'}), 
            rid=self.param.selected_region_id
        )
        
        # Topoplot is now shown in hover tooltip, no need for fixed display

        style = pn.pane.HTML("""<style>
        .bk-btn-group .bk-btn:nth-child(1) { background-color: #27ae60 !important; color: white !important; }
        .bk-btn-group .bk-btn:nth-child(2) { background-color: #95a5a6 !important; color: white !important; }
        </style>""")

        # Create bounds stream for box selection - source will be set in create_main_plot
        # Initialize it here but source will be updated when plot is created
        if not hasattr(self, '_bounds_stream') or self._bounds_stream is None:
            self._bounds_stream = streams.BoundsXY()
            self._bounds_stream.add_subscriber(self._on_box_select)
        
        # Create main plot with both tap and bounds streams
        main_dmap = hv.DynamicMap(self.create_main_plot, streams=[self.tap_stream, self._bounds_stream])
        
        # Fix focus plots: use functools.partial or proper lambda to capture channel index
        from functools import partial
        
        # Explicitly define streams for focus plots since partial() hides param.depends metadata
        focus_streams = [hv.streams.Params(self, ['epoch_index', 'focus_plot_trigger'])]
        
        focus_dmaps = []
        for ch_idx in self.focus_channels:
            # Create a bound method that properly captures the channel index
            focus_dmap = hv.DynamicMap(partial(self.create_focus_plot, channel_idx=ch_idx), streams=focus_streams)
            focus_dmaps.append(focus_dmap)
        focus_col = pn.Column(*[pn.pane.HoloViews(dmap) for dmap in focus_dmaps])
        
        # Compact layout with controls in a single row
        controls_row = pn.Row(
            btn_prev, btn_next,
            pn.Spacer(width=10),
            info,
            pn.Spacer(width=10),
            radio_group,
            pn.Spacer(width=10),
            pn.pane.Markdown("**Keys:** `K`/`C` = KC, `U`/`Y` = unannotated, `D` = delete | **Drag to select** | **Hover for topoplot**",
                           styles={'font-size': '11px', 'margin': '0px'}),
            align='center',
            sizing_mode='stretch_width',
            margin=(5, 0)
        )
        
        # Add topoplot panel to the right of focus plots
        bottom_row = pn.Row(
            focus_col,
            pn.Spacer(width=20),
            self.topoplot_panel,
            sizing_mode='stretch_width'
        )
        
        return pn.Column(
            self.status_bar,  # Status bar at top
            controls_row,  # Compact controls row
            pn.pane.HoloViews(main_dmap, sizing_mode='stretch_width'),
            bottom_row,
            self.kb_listener,  # <--- INVISIBLE LISTENER COMPONENT
            style,
            sizing_mode='stretch_width',
            margin=(0, 10)
        )

def create_dashboard(epoch_manager, annotation_manager, focus_channels=None, main_plot_channels=None, chanlocs=None, channels_file=None, exclude_channels=None):
    """Create dashboard instance."""
    db = EEGDashboard(epoch_manager, annotation_manager, focus_channels, main_plot_channels, chanlocs, channels_file, exclude_channels)
    return db.view()
