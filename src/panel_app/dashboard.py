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
from scipy import signal
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

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
# === CHANGE ANNOTATION COLORS ===
# Modify these to change the color of annotated regions
ANNOTATION_COLORS = {
    'unannotated': '#95a5a6',  # Gray - color for unannotated regions
    'KC': '#27ae60',           # Green - color for KC-labeled regions
}

# === CHANGE CONTEXT DISPLAY SETTINGS ===
# Display settings: show extra context around each epoch
# Epoch progression/hop remains controlled by EpochManager.epoch_length_sec (e.g. 20s).
DISPLAY_CONTEXT_BEFORE_SEC = 5.0   # Seconds of context shown before epoch
DISPLAY_CONTEXT_AFTER_SEC = 5.0    # Seconds of context shown after epoch
CONTEXT_BACKGROUND_COLOR = '#cfe8ff'  # Light blue - background color for context regions

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
                 exclude_channels: List[int] = None,
                 plot_width: int = 1200,
                 plot_height: int = 500,
                 **params):
        
        # --- FIX: Set attributes BEFORE super().__init__ ---
        # This prevents "AttributeError" if watchers fire during init
        self.epoch_manager = epoch_manager
        self.annotation_manager = annotation_manager
        self.focus_channels = focus_channels or [34, 55, 70]
        self.main_plot_channels = main_plot_channels
        self.exclude_channels = exclude_channels or []
        self.plot_width = int(plot_width)
        self.plot_height = int(plot_height)
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
        # Cache for spectral power data (0.5-4 Hz) per epoch
        # Maps epoch_index -> DataFrame (samples × channels) with power values
        self._spectral_power_cache = {}
        
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
        
        # Track when epoch actually changes (for box selection clearing)
        self._previous_epoch_index = -1
        self._should_clear_box_selection = False
        self._should_clear_box_selection_focus = False
        
        # Now call super, which might trigger watchers immediately
        super().__init__(**params)
        
        self.param.epoch_index.bounds = (0, max(0, epoch_manager.get_epoch_count() - 1))
        
        # Initialize previous epoch index to current value
        self._previous_epoch_index = self.epoch_index
        
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
            # IMPORTANT: Regions are epoch-dependent. If epoch_data changed, any cached
            # regions from the previous epoch are invalid and must be refreshed.
            self._cached_regions = None
            
            # Compute spectral power (0.5-4 Hz) for the entire epoch
            self._compute_epoch_spectral_power(epoch_data)
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

    def _get_display_window_data(self) -> Tuple[pd.DataFrame, int, int, float, float]:
        """
        Get a display window around the current epoch with context:
        - DISPLAY_CONTEXT_BEFORE_SEC seconds before epoch start
        - DISPLAY_CONTEXT_AFTER_SEC seconds after epoch end

        Returns:
            display_df: DataFrame of samples in the display window
            pre_samples: number of samples shown before the epoch start (may be < target near recording start)
            post_samples: number of samples shown after the epoch end (may be < target near recording end)
            center_start_time_s: time (s) in display window where the *epoch* starts
            center_end_time_s: time (s) in display window where the *epoch* ends
        """
        if (hasattr(self, '_cached_display_epoch_index') and
            self._cached_display_epoch_index == self.epoch_index and
            getattr(self, '_cached_display_df', None) is not None):
            return (self._cached_display_df,
                    self._cached_display_pre_samples,
                    self._cached_display_post_samples,
                    self._cached_display_center_start_time_s,
                    self._cached_display_center_end_time_s)

        # Ensure epoch manager is on the current filtered epoch
        self.epoch_manager.current_epoch_idx = self.epoch_index
        epoch_info = self.epoch_manager.get_current_epoch_info()
        if not epoch_info:
            empty = pd.DataFrame()
            return empty, 0, 0, 0.0, 0.0

        epoch_start = int(epoch_info['start_idx'])
        epoch_end = int(epoch_info['end_idx'])  # inclusive

        sr = float(self.sampling_rate)
        pre_target = int(DISPLAY_CONTEXT_BEFORE_SEC * sr)
        post_target = int(DISPLAY_CONTEXT_AFTER_SEC * sr)

        n_samples_total = len(self.epoch_manager.eeg_data)
        win_start = max(0, epoch_start - pre_target)
        win_end = min(n_samples_total - 1, epoch_end + post_target)  # inclusive

        display_df = self.epoch_manager.eeg_data.iloc[win_start:win_end + 1].copy()

        pre_samples = int(epoch_start - win_start)
        post_samples = int(win_end - epoch_end)

        epoch_len_samples = int(epoch_end - epoch_start + 1)
        center_start_time_s = pre_samples / sr
        center_end_time_s = (pre_samples + epoch_len_samples) / sr

        # Cache
        self._cached_display_epoch_index = self.epoch_index
        self._cached_display_df = display_df
        self._cached_display_pre_samples = pre_samples
        self._cached_display_post_samples = post_samples
        self._cached_display_center_start_time_s = center_start_time_s
        self._cached_display_center_end_time_s = center_end_time_s

        return display_df, pre_samples, post_samples, center_start_time_s, center_end_time_s
    
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
            
            # Convert time to sample indices (relative to DISPLAY WINDOW),
            # then map to indices relative to the EPOCH by subtracting pre-context.
            _, pre_samples, _, center_start_time_s, center_end_time_s = self._get_display_window_data()
            start_idx_display = int(start_time * self.sampling_rate)
            stop_idx_display = int(end_time * self.sampling_rate)

            start_idx = start_idx_display - pre_samples
            stop_idx = stop_idx_display - pre_samples
            
            # Clamp to epoch boundaries
            start_idx = max(0, min(start_idx, len(epoch_data) - 1))
            stop_idx = max(0, min(stop_idx, len(epoch_data) - 1))
            
            if start_idx >= stop_idx:
                print("⚠️ Invalid selection: start >= stop")
                return
            
            print(
                f"📦 Box selected (display): {start_time:.3f}s - {end_time:.3f}s | "
                f"(epoch-relative samples {start_idx}-{stop_idx}, epoch window {center_start_time_s:.1f}s-{center_end_time_s:.1f}s)"
            )
            
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
        click_sample_display = int(click_time * self.sampling_rate)
        _, pre_samples, _, _, _ = self._get_display_window_data()
        click_sample = click_sample_display - pre_samples
        
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

            # Ignore clicks in the context-only areas (outside the current epoch window)
            epoch_data, current_regions, _ = self._get_epoch_data()
            if epoch_data is None or len(epoch_data) == 0:
                return
            t1 = time_module.time()

            if not current_regions:
                return

            epoch_len_samples = len(epoch_data)
            if click_sample < 0 or click_sample > epoch_len_samples:
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
    def _compute_epoch_spectral_power(self, epoch_data: pd.DataFrame):
        """
        Compute spectral power (0.5-4 Hz) for the entire epoch using sliding window approach.
        Stores power values in _spectral_power_cache[epoch_index] as DataFrame (samples × channels).
        """
        if epoch_data is None or len(epoch_data) == 0:
            return
        
        try:
            # Parameters for spectral power computation
            low_freq = 0.5  # Hz
            high_freq = 4.0  # Hz
            window_length_sec = 2.0  # 2-second windows for power computation
            overlap_sec = 0.5  # 1-second overlap
            
            sr = self.sampling_rate
            window_length_samples = int(window_length_sec * sr)
            overlap_samples = int(overlap_sec * sr)
            hop_samples = window_length_samples - overlap_samples
            
            n_samples, n_channels = epoch_data.shape
            power_data = np.zeros((n_samples, n_channels))
            
            # Compute power for each channel using sliding window
            for ch_idx in range(n_channels):
                channel_data = epoch_data.iloc[:, ch_idx].values
                
                # Use Welch's method with sliding windows
                # For each window position, compute power in 0.5-4 Hz band
                for start_idx in range(0, n_samples - window_length_samples + 1, hop_samples):
                    end_idx = start_idx + window_length_samples
                    window_data = channel_data[start_idx:end_idx]
                    
                    # Compute power spectral density using Welch's method
                    freqs, psd = signal.welch(
                        window_data,
                        fs=sr,
                        nperseg=min(window_length_samples, len(window_data)),
                        noverlap=overlap_samples // 2 if overlap_samples > 0 else None
                    )
                    
                    # Integrate power in 0.5-4 Hz band
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):
                        band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    else:
                        band_power = 0.0
                    
                    # Assign power value to all samples in this window
                    power_data[start_idx:end_idx, ch_idx] = band_power
                
                # Handle edge cases: fill samples at the beginning/end that weren't covered
                if n_samples > 0:
                    power_col = power_data[:, ch_idx]
                    nonzero_mask = power_col > 0
                    if np.any(nonzero_mask):
                        # Forward fill from first computed value
                        first_nonzero_idx = np.where(nonzero_mask)[0][0]
                        if first_nonzero_idx > 0:
                            power_data[:first_nonzero_idx, ch_idx] = power_data[first_nonzero_idx, ch_idx]
                        
                        # Backward fill from last computed value
                        last_nonzero_idx = np.where(nonzero_mask)[0][-1]
                        if last_nonzero_idx < n_samples - 1:
                            power_data[last_nonzero_idx+1:, ch_idx] = power_data[last_nonzero_idx, ch_idx]
            
            # Store as DataFrame
            power_df = pd.DataFrame(power_data, columns=epoch_data.columns, index=epoch_data.index)
            self._spectral_power_cache[self.epoch_index] = power_df
            
        except Exception as e:
            print(f"[ERROR SPECTRAL] Error computing spectral power for epoch {self.epoch_index}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_topoplot_image(self, region_id: int, epoch_data: pd.DataFrame, 
                               start_idx: int, stop_idx: int) -> Optional[str]:
        """Create base64-encoded topoplot image for a selected region using spectral power."""
        if not MNE_AVAILABLE:
            return None
        
        try:
            # Check if we have cached spectral power for this epoch
            if self.epoch_index not in self._spectral_power_cache:
                print(f"[WARNING TOPO] No spectral power cache for epoch {self.epoch_index}, computing now...")
                self._compute_epoch_spectral_power(epoch_data)
            
            power_df = self._spectral_power_cache.get(self.epoch_index)
            if power_df is None or len(power_df) == 0:
                print(f"[WARNING TOPO] Empty spectral power cache for epoch {self.epoch_index}")
                return None
            
            # Extract power values for the selected time window
            if start_idx >= len(power_df) or stop_idx >= len(power_df):
                return None
            
            region_power = power_df.iloc[start_idx:stop_idx+1]
            
            # Average power across time window for each channel
            mean_power = region_power.mean(axis=0).values
            
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
                    n_chans = len(mean_power)
                    angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
                    pos = np.array([np.cos(angles), np.sin(angles)]).T
            else:
                # Fallback: create circular layout
                n_chans = len(mean_power)
                angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
                pos = np.array([np.cos(angles), np.sin(angles)]).T
            
            # Normalize positions to fit within unit circle (MNE expects positions roughly in -1 to 1 range)
            # This ensures the head outline matches the electrode positions
            pos_original = pos.copy()  # Keep original for focus channel markers
            max_radius = np.max(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))
            if max_radius > 0:
                pos = pos / max_radius * 0.9  # Scale to 90% of unit circle
            
            # Filter out excluded channels from topoplot
            # Track which indices are kept for focus channel marking
            kept_indices = np.arange(len(mean_power))
            if self.exclude_channels:
                n_chans = len(mean_power)
                # Create mask: True for channels to KEEP
                mask = np.ones(n_chans, dtype=bool)
                for idx in self.exclude_channels:
                    if 0 <= idx < n_chans:
                        mask[idx] = False
                
                mean_power = mean_power[mask]
                pos = pos[mask]
                pos_original = pos_original[mask]
                kept_indices = kept_indices[mask]
            
            # Create topoplot using MNE (showing spectral power)
            fig, ax = plt.subplots(figsize=(3, 3))
            mne.viz.plot_topomap(
                mean_power,
                pos,
                axes=ax,
                show=False,
                cmap='Reds',  # Use Reds colormap for power (all positive values)
                vlim=(None, None)
            )
            
            # Mark the 3 focus channels on the topoplot
            if self.focus_channels:
                for focus_ch in self.focus_channels:
                    # Find where this channel is in the kept_indices
                    match_idx = np.where(kept_indices == focus_ch)[0]
                    if len(match_idx) > 0:
                        idx = match_idx[0]
                        # Plot marker at this position
                        ax.plot(pos[idx, 0], pos[idx, 1], 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
                        ax.plot(pos[idx, 0], pos[idx, 1], 'k+', markersize=6, markeredgewidth=1.5)
            
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
                                epoch_data: Optional[pd.DataFrame] = None,
                                time_offset_seconds: float = 0.0):
        """Create rectangles for selected regions."""
        if not regions:
            return hv.Rectangles([], kdims=['x0', 'y0', 'x1', 'y1'])
        
        y_min, y_max = y_range
        rect_data = []
        
        # Initialize cache if epoch changed and generate topoplots for all regions upfront
        if epoch_data is not None and MNE_AVAILABLE:
            # Clear cache if epoch changed
            if self._current_epoch_for_cache != self.epoch_index:
                self._topoplot_cache = {}
                self._current_epoch_for_cache = self.epoch_index
                # Spectral power cache is managed per epoch, no need to clear here
            
            # Cache structure for this epoch
            if self.epoch_index not in self._topoplot_cache:
                self._topoplot_cache[self.epoch_index] = {}
            
            # Generate topoplots for all regions upfront (for hover tooltip)
            for region_id, region in regions.items():
                rel_start = region.get('relative_start', region['start_idx'])
                rel_stop = region.get('relative_stop', region['stop_idx'])
                
                # Generate topoplot if not already cached
                if region_id not in self._topoplot_cache[self.epoch_index]:
                    topo_img = self._create_topoplot_image(region_id, epoch_data, rel_start, rel_stop)
                    if topo_img:
                        self._topoplot_cache[self.epoch_index][region_id] = topo_img
        
        for region_id, region in regions.items():
            # Get relative indices for display
            rel_start = region.get('relative_start', region['start_idx'])
            rel_stop = region.get('relative_stop', region['stop_idx'])
            
            status = self.annotation_manager.get_annotation_status(region_id)
            is_sel = (region_id == self.selected_region_id)
            
            # === CHANGE REGION SELECTION STYLING ===
            # Color scheme: show label color even when selected, with brighter yellow outline
            if status == 'KC':
                fill = ANNOTATION_COLORS['KC']
                base_lc = ANNOTATION_COLORS['KC']
            else:
                fill = ANNOTATION_COLORS['unannotated']
                base_lc = ANNOTATION_COLORS['unannotated']
            
            if is_sel:
                # Selected region styling - modify lc (outline color), lw (line width), alpha
                lc = '#FFD700'  # Bright gold/yellow outline when selected
                lw = 4          # Thicker border when selected
                alpha = 0.25    # Slightly more visible when selected
            else:
                # Non-selected region styling
                lc = base_lc
                lw = 2          # Border thickness when not selected
                alpha = 0.15    # Transparent so EEG shows through
            
            # Convert sample indices to time, shifting by time_offset_seconds (e.g. pre-context)
            start_time = (rel_start / self.sampling_rate) + time_offset_seconds
            stop_time = (rel_stop / self.sampling_rate) + time_offset_seconds
            
            # Get topoplot HTML from cache (now pre-generated)
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
        
        # Store epoch_data for hover tool callback
        if epoch_data is not None:
            if not hasattr(self, '_hover_epoch_data'):
                self._hover_epoch_data = {}
            self._hover_epoch_data[self.epoch_index] = epoch_data
        
        print(f"[HOVER DEBUG] _create_selected_regions: Creating {len(df_rects)} rectangles for hover tool hook")
        
        # Note: HoverTool will be created/configured by the _configure_hover_tool_hook
        # Note: clone=False ensures each Rectangles element is unique (not cached by HoloViews)
        # Note: We include 'hover' in tools to force HoloViews to include extra columns (like topo_html)
        #       in the data source, even though we might replace/configure the actual tool in the hook.
        return rects.opts(
            color='fill_color', 
            line_color='line_color', 
            line_width='line_width', 
            alpha='alpha', 
            tools=['hover', 'tap'],  # 'hover' ensures columns are preserved
            nonselection_alpha='alpha',
            nonselection_line_alpha=0.9,
            nonselection_color='fill_color',
            nonselection_line_color='line_color',
            clone=False  # Prevent HoloViews from caching this element
        )

    # === CHANGE DEFAULT CHANNEL COLOR ===
    def _get_channel_color(self, channel_idx: int) -> str:
        """Get color for a channel, using channel colors if available."""
        if channel_idx in self.channel_colors:
            r, g, b = self.channel_colors[channel_idx]
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#34495e'  # Default dark gray - change this for default EEG color
    
    def _configure_hover_tool_hook(self, plot, element):
        """Create and configure HoverTool for topoplot tooltips on rectangle renderers."""
        from bokeh.models import HoverTool
        from bokeh.models.glyphs import Quad
        
        # print(f"[HOVER DEBUG] _configure_hover_tool_hook CALLED for plot {id(plot)}")
        
        if not hasattr(plot, 'state'):
            # print(f"[HOVER DEBUG] Hook: No plot.state")
            return
        
        def configure():
            """Find Quad renderers and create/update HoverTool for them"""
            # Find Quad (rectangle) renderers
            rect_renderers = []
            renderer_types = []
            if hasattr(plot.state, 'renderers'):
                for renderer in plot.state.renderers:
                    if hasattr(renderer, 'glyph'):
                        renderer_types.append(type(renderer.glyph).__name__)
                        if isinstance(renderer.glyph, Quad):
                            rect_renderers.append(renderer)
            
            if not rect_renderers:
                # print(f"[HOVER DEBUG] Hook: No Quad renderers yet. Found: {renderer_types}")
                return False
            
            # Find existing HoverTool for rectangles or create a new one
            hover_tool = None
            if hasattr(plot.state, 'tools'):
                for tool in plot.state.tools:
                    if isinstance(tool, HoverTool):
                        hover_tool = tool
                        break
            
            if hover_tool:
                # Update existing HoverTool to point to current rectangle renderers and ensure tooltips are correct
                hover_tool.renderers = rect_renderers
                hover_tool.tooltips = """
                    <div style="font-size: 12px;">
                        <strong>Region @region_id</strong><br>
                        Status: @status<br>
                        Time: @x0{0.00}s - @x1{0.00}s<br>
                        @topo_html{safe}
                    </div>
                """
                # print(f"[HOVER DEBUG] Hook SUCCESS: Updated HoverTool for {len(rect_renderers)} Quad renderers (plot {id(plot)})")
            else:
                # Create a new HoverTool specifically for rectangles
                hover_tool = HoverTool(
                    tooltips="""
                        <div style="font-size: 12px;">
                            <strong>Region @region_id</strong><br>
                            Status: @status<br>
                            Time: @x0{0.00}s - @x1{0.00}s<br>
                            @topo_html{safe}
                        </div>
                    """,
                    renderers=rect_renderers,
                    point_policy='follow_mouse',
                    attachment='above'
                )
                plot.state.add_tools(hover_tool)
                # print(f"[HOVER DEBUG] Hook SUCCESS: Added HoverTool for {len(rect_renderers)} Quad renderers (plot {id(plot)})")
            
            return True
        
        # Try immediately
        if configure():
            return
        
        # If failed, retry on next tick
        # print(f"[HOVER DEBUG] Hook: First attempt failed, scheduling retry")
        if hasattr(plot.state, 'document') and plot.state.document:
            plot.state.document.add_next_tick_callback(lambda: configure())
    
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
        epoch_changed_for_cache = (self._cached_epoch_data_index != self.epoch_index)
        # Check if epoch_index param actually changed (not just update_trigger)
        epoch_index_changed = (self._previous_epoch_index != self.epoch_index)
        
        if epoch_changed_for_cache:
            self._cached_epoch_data = None
            self._cached_regions = None
            self._cached_epoch_start = None
            # Also clear focus plot cache
            self._cached_focus_plots = {}
            self._cached_focus_epoch_index = -1
            # Trigger focus plot update
            self.focus_plot_trigger += 1
        
        # Set flag to clear box selection only when epoch_index actually changes
        if epoch_index_changed:
            self._should_clear_box_selection = True
            self._should_clear_box_selection_focus = True
            self._previous_epoch_index = self.epoch_index
            # Clear box selection bounds when epoch changes
            if hasattr(self, '_bounds_stream') and self._bounds_stream is not None:
                try:
                    # Reset bounds stream to clear visual selection
                    self._bounds_stream.event(bounds=None)
                except:
                    pass
            if hasattr(self, '_bounds_stream_focus') and self._bounds_stream_focus is not None:
                try:
                    # Reset focus bounds stream to clear visual selection
                    self._bounds_stream_focus.event(bounds=None)
                except:
                    pass
        
        t1 = time_module.time()
        # Central epoch data (for annotations/topoplots) + display window (for plotting)
        epoch_data, current_regions, _ = self._get_epoch_data()
        display_df, pre_samples, post_samples, center_start_time_s, center_end_time_s = self._get_display_window_data()
        t2 = time_module.time()
        
        if epoch_data is None or len(epoch_data) == 0 or display_df is None or len(display_df) == 0:
            if epoch_data is None:
                print(f"⚠️ epoch_data is None in create_main_plot")
            else:
                print(f"⚠️ Empty epoch_data in create_main_plot")
            return hv.Spacer()
        
        # Cache curves if epoch hasn't changed (only regions changed)
        epoch_changed = (self._cached_epoch_index != self.epoch_index)
        
        if epoch_changed or self._cached_curves is None:
            # Recreate curves when epoch changes
            t = self._create_time_axis(len(display_df))
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
                n_total = len(display_df.columns)
                indices = [i for i in self.main_plot_channels if 0 <= i < n_total]
                if not indices:
                    print("Warning: No valid channels in main_plot_channels. Falling back to default.")
                    indices = list(range(min(30, n_total)))
            else:
                # Limit to ~30 channels max to keep UI responsive
                n_channels = len(display_df.columns)
                target_n_channels = 30
                if n_channels > target_n_channels:
                    step = max(1, n_channels // target_n_channels)
                    indices = list(range(0, n_channels, step))
                else:
                    indices = list(range(n_channels))
            
            for i_idx, i in enumerate(indices):
                col = display_df.columns[i]
                d = display_df[col].values
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
                
                # === CHANGE EEG LINES (BUTTERFLY PLOT) ===
                # Modify line_width, alpha, color below to change EEG curve appearance
                if i_idx == 0:
                    # First curve: enable box_select and store reference
                    first_curve = hv.Curve((tt, dd), label=f'Ch {i}').opts(
                        color=color,
                        line_width=0.6,      # EEG line thickness (butterfly)
                        alpha=0.7,           # EEG line transparency (butterfly)
                        tools=['box_select', 'tap']
                    )
                    curves.append(first_curve)
                else:
                    curve = hv.Curve((tt, dd), label=f'Ch {i}').opts(
                        color=color,
                        line_width=0.6,      # EEG line thickness (butterfly)
                        alpha=0.7            # EEG line transparency (butterfly)
                    )
                    curves.append(curve)
            
            # === CHANGE Y-AXIS RANGE (BUTTERFLY PLOT) ===
            # Use fixed range for butterfly plot (requested: ~-230 to +230)
            # The normalized data is already scaled to ~30 units, so we use a fixed range
            y_limit_min = -230   # Y-axis minimum (butterfly)
            y_limit_max = 230    # Y-axis maximum (butterfly)
            
            # Cache curves and epoch index
            self._cached_curves = curves
            self._cached_epoch_index = self.epoch_index
            self._cached_y_range = (y_limit_min, y_limit_max)
        else:
            # Reuse cached curves
            curves = self._cached_curves
            first_curve = curves[0] if curves else None
            y_limit_min, y_limit_max = self._cached_y_range
        
        # Dynamic y_range for butterfly plot
        y_range = (y_limit_min, y_limit_max)
        
        t3 = time_module.time()
        # Add region rectangles
        regions_rects = self._create_selected_regions(
            current_regions,
            y_range,
            epoch_data,  # central epoch for topoplot indexing
            time_offset_seconds=center_start_time_s
        )
        t4 = time_module.time()
        
        # Background shading for context regions (pre/post)
        vspans = []
        if pre_samples > 0 and center_start_time_s > 0:
            vspans.append(
                hv.VSpan(0.0, center_start_time_s).opts(
                    color=CONTEXT_BACKGROUND_COLOR, alpha=0.25, line_width=0
                )
            )
        display_duration_s = len(display_df) / self.sampling_rate
        if post_samples > 0 and center_end_time_s < display_duration_s:
            vspans.append(
                hv.VSpan(center_end_time_s, display_duration_s).opts(
                    color=CONTEXT_BACKGROUND_COLOR, alpha=0.25, line_width=0
                )
            )

        # === CHANGE REFERENCE LINES (BUTTERFLY PLOT) ===
        # Add zero line and threshold line for butterfly plot
        reference_lines = []
        # Zero line - modify color, line_width, line_dash, alpha
        zero_line = hv.HLine(0).opts(color='gray', line_width=1, line_dash='dashed', alpha=0.5)
        reference_lines.append(zero_line)
        
        # Threshold line (if configured in config.py AMPLITUDE_THRESHOLD)
        threshold = getattr(config, 'AMPLITUDE_THRESHOLD', None)
        if threshold is not None:
            # Threshold line - modify color, line_width, line_dash, alpha
            threshold_line = hv.HLine(threshold).opts(color='red', line_width=1, line_dash='dashed', alpha=0.7)
            reference_lines.append(threshold_line)

        # Create combined plot (background spans behind curves, then regions, then reference lines)
        elements = []
        elements.extend(vspans)
        elements.extend(curves)
        elements.extend(reference_lines)
        combined = hv.Overlay(elements) * regions_rects
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
        # Use local variables to ensure this plot's range is independent
        butterfly_y_min = y_limit_min
        butterfly_y_max = y_limit_max
        
        def set_y_range(plot, element):
            """Hook to set y-axis range dynamically and lock it - BUTTERFLY PLOT ONLY"""
            # Access the y_range directly from handles
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range_handle = plot.handles['y_range']
                # Use local variables to ensure independence from other plots
                # Force set to butterfly plot range only
                y_range_handle.start = butterfly_y_min
                y_range_handle.end = butterfly_y_max
                # Prevent auto-scaling by setting bounds
                y_range_handle.bounds = (butterfly_y_min, butterfly_y_max)
                # Disable auto-range
                if hasattr(y_range_handle, 'reset_start'):
                    y_range_handle.reset_start = butterfly_y_min
                    y_range_handle.reset_end = butterfly_y_max
                # Ensure range doesn't auto-update
                if hasattr(y_range_handle, 'auto_range'):
                    y_range_handle.auto_range = False
        
        # Additional hook to re-lock range after any update
        def lock_y_range(plot, element):
            """Re-lock y-axis range after plot updates - BUTTERFLY PLOT ONLY"""
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range_handle = plot.handles['y_range']
                # Always force range back to butterfly plot values - use local variables
                # Don't check, just force it to be correct
                y_range_handle.start = butterfly_y_min
                y_range_handle.end = butterfly_y_max
                y_range_handle.bounds = (butterfly_y_min, butterfly_y_max)
                if hasattr(y_range_handle, 'auto_range'):
                    y_range_handle.auto_range = False
                if hasattr(y_range_handle, 'reset_start'):
                    y_range_handle.reset_start = butterfly_y_min
                    y_range_handle.reset_end = butterfly_y_max
        
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
            from bokeh.models.glyphs import Line

            def apply_box_select_renderers():
                box_select = None
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            box_select = tool
                            break
                if not box_select:
                    return
                # Prefer handles (HoloViews), fallback to state.renderers (reliable after reload)
                renderers = []
                if hasattr(plot, 'handles') and 'glyph_renderers' in plot.handles:
                    renderers = list(plot.handles.get('glyph_renderers') or [])
                if not renderers and hasattr(plot, 'state') and hasattr(plot.state, 'renderers'):
                    renderers = list(plot.state.renderers)
                if not renderers:
                    return
                line_renderer = None
                for r in renderers:
                    try:
                        if hasattr(r, 'glyph') and isinstance(r.glyph, Line):
                            line_renderer = r
                            break
                    except Exception:
                        continue
                if line_renderer is None:
                    line_renderer = renderers[0]
                box_select.renderers = [line_renderer]

            apply_box_select_renderers()
            # Re-apply after next tick so box_select has correct renderers on reload
            # (handles can be empty when hook runs on first paint)
            if hasattr(plot, 'state') and hasattr(plot.state, 'document') and plot.state.document:
                try:
                    plot.state.document.add_next_tick_callback(apply_box_select_renderers)
                except Exception:
                    pass

        # Manually push BoxSelectTool overlay to stream (HoloViews link can break on reload)
        def link_box_select_to_stream(plot, element):
            from bokeh.models import BoxSelectTool
            if not hasattr(self, '_bounds_stream') or self._bounds_stream is None:
                return

            def attach():
                box_select = None
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            box_select = tool
                            break
                if not box_select or not hasattr(box_select, 'overlay') or box_select.overlay is None:
                    return
                stream_ref = self._bounds_stream

                def push_bounds_to_stream():
                    o = box_select.overlay
                    left = getattr(o, 'left', None)
                    right = getattr(o, 'right', None)
                    top = getattr(o, 'top', None)
                    bottom = getattr(o, 'bottom', None)
                    if left is not None and right is not None and top is not None and bottom is not None:
                        x0, x1 = min(left, right), max(left, right)
                        y0, y1 = min(top, bottom), max(top, bottom)
                        try:
                            stream_ref.event(bounds=(x0, y0, x1, y1))
                        except Exception:
                            pass

                for prop in ('left', 'right', 'top', 'bottom'):
                    try:
                        box_select.overlay.on_change(prop, lambda a, o, n: push_bounds_to_stream())
                    except Exception:
                        pass

            attach()
            if hasattr(plot, 'state') and hasattr(plot.state, 'document') and plot.state.document:
                try:
                    plot.state.document.add_next_tick_callback(attach)
                except Exception:
                    pass
        
        # Hook to clear box selection when epoch changes
        def clear_box_selection(plot, element):
            """Clear BoxSelectTool visual selection - BUTTERFLY PLOT"""
            from bokeh.models import BoxSelectTool
            
            # Only clear if epoch_index actually changed (not just update_trigger)
            if self._should_clear_box_selection:
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            # Clear the selection overlay (visual box)
                            if hasattr(tool, 'overlay'):
                                tool.overlay.left = None
                                tool.overlay.right = None
                                tool.overlay.top = None
                                tool.overlay.bottom = None
                            # Clear any selection on renderers
                            if hasattr(tool, 'renderers') and tool.renderers:
                                for renderer in tool.renderers:
                                    if hasattr(renderer, 'data_source'):
                                        # Clear selection indices
                                        if hasattr(renderer.data_source, 'selected'):
                                            renderer.data_source.selected.indices = []
                            break
                # Clear the flag after clearing
                self._should_clear_box_selection = False
        
        # X range: exact data range (0 to display_duration_s) with no extra padding
        display_duration_s = len(display_df) / self.sampling_rate
        
        # Build tools list
        tools_list = ['tap', 'xwheel_zoom', 'xpan', 'box_select']
        
        plot_opts = opts.Overlay(
            height=self.plot_height, shared_axes=True, show_legend=False, 
            responsive=True,  # Make plot responsive to container width
            tools=tools_list, 
            active_tools=['tap', 'box_select'],
            hooks=[set_y_range, lock_y_range, clear_box_selection, ensure_box_select, optimize_selection, link_box_select_to_stream, self._configure_hover_tool_hook],
            ylim=(butterfly_y_min, butterfly_y_max),  # Explicitly set ylim to prevent auto-scaling - BUTTERFLY PLOT ONLY
            xlim=(0, display_duration_s),  # No white space before 0 or after display end
            framewise=False,  # Prevent auto-scaling on updates
        )
        
        return combined.opts(plot_opts)

    @param.depends('epoch_index', 'update_trigger')
    def create_focus_channels_plot(self, bounds=None, **kwargs):
        """Create stacked plot with 3 focus channels displayed above each other."""
        import time as time_module
        t0 = time_module.time()
        
        # Clear last bounds when epoch changes to allow new selections
        if hasattr(self, '_last_bounds'):
            delattr(self, '_last_bounds')
        
        # Check if epoch_index actually changed (not just update_trigger)
        # The flag is set in create_main_plot, so we just check it here
        epoch_changed_for_focus = (self._cached_epoch_data_index != self.epoch_index)
        
        epoch_data, current_regions, _ = self._get_epoch_data()  # central epoch (for annotations/topoplots)
        display_df, pre_samples, post_samples, center_start_time_s, center_end_time_s = self._get_display_window_data()
        t1 = time_module.time()
        
        if epoch_data is None or len(epoch_data) == 0 or display_df is None or len(display_df) == 0:
            return hv.Spacer()
        
        # Check if we have valid focus channels
        if not self.focus_channels or len(self.focus_channels) == 0:
            return hv.Spacer()
        
        # Validate channel indices
        n_channels = len(display_df.columns)
        valid_channels = [ch for ch in self.focus_channels if 0 <= ch < n_channels]
        if not valid_channels:
            return hv.Spacer()
        
        t = self._create_time_axis(len(display_df))
        
        # Create curves for each focus channel with vertical offsets
        # Normalize each channel similar to butterfly plot, then apply offsets
        curves = []
        first_curve = None
        offset_per_channel = 500  # Vertical spacing between channels
        # === CHANGE Y-AXIS SCALE (3-CHANNEL PLOT) ===
        channel_range = 200  # Y-axis range per channel: -200..200
        
        for i_idx, channel_idx in enumerate(valid_channels):
            col = display_df.columns[channel_idx]
            d = display_df[col].values
            
            # Normalize channel similar to butterfly plot (scale to ~30 units)
            d_normalized = (d - np.mean(d)) / (np.std(d) or 1) * 30
            
            # Apply vertical offset: first channel (i_idx=0) at TOP, last channel at BOTTOM
            # This ensures config order [Fz, 6, Pz] displays as Fz on top, Pz at bottom
            offset = (len(valid_channels) - 1 - i_idx) * offset_per_channel
            d_offset = d_normalized + offset
            
            tt, dd = downsample_minmax(d_offset, t)
            color = self._get_channel_color(channel_idx)
            
            # === CHANGE EEG LINES (3-CHANNEL STACKED PLOT) ===
            # Create curve with explicit y-range to prevent auto-scaling
            curve_opts = {
                'color': color,
                'line_width': 0.5,   # EEG line thickness (3-channel)
                'alpha': 1,        # EEG line transparency (3-channel)
            }
            
            if i_idx == 0:
                # First curve: enable box_select and store reference
                curve_opts['tools'] = ['box_select', 'tap']
                first_curve = hv.Curve((tt, dd), label=f'Ch {channel_idx}').opts(**curve_opts)
                curves.append(first_curve)
            else:
                curve = hv.Curve((tt, dd), label=f'Ch {channel_idx}').opts(**curve_opts)
                curves.append(curve)
        
        # Calculate y_range to cover all stacked channels
        # Each normalized channel spans roughly -250 to +250, with offsets
        y_min = -channel_range
        y_max = (len(valid_channels) - 1) * offset_per_channel + channel_range
        y_range = (y_min, y_max)
        
        t2 = time_module.time()
        # Add region rectangles spanning the full y-range (shifted by pre-context)
        regions_rects = self._create_selected_regions(
            current_regions,
            y_range,
            epoch_data,  # central epoch for topoplot indexing
            time_offset_seconds=center_start_time_s
        )
        t3 = time_module.time()
        
        # Background shading for context regions (pre/post)
        vspans = []
        if pre_samples > 0 and center_start_time_s > 0:
            vspans.append(
                hv.VSpan(0.0, center_start_time_s).opts(
                    color=CONTEXT_BACKGROUND_COLOR, alpha=0.25, line_width=0
                )
            )
        display_duration_s = len(display_df) / self.sampling_rate
        if post_samples > 0 and center_end_time_s < display_duration_s:
            vspans.append(
                hv.VSpan(center_end_time_s, display_duration_s).opts(
                    color=CONTEXT_BACKGROUND_COLOR, alpha=0.25, line_width=0
                )
            )

        # === CHANGE REFERENCE LINES (3-CHANNEL STACKED PLOT) ===
        # Add zero lines and threshold lines for each channel (with offsets)
        reference_lines = []
        threshold = getattr(config, 'AMPLITUDE_THRESHOLD', None)
        
        for i_idx, channel_idx in enumerate(valid_channels):
            # Match curve offset: first channel at TOP, last at BOTTOM
            offset = (len(valid_channels) - 1 - i_idx) * offset_per_channel
            
            # Zero line for this channel - modify color, line_width, line_dash, alpha
            zero_line = hv.HLine(offset).opts(color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
            reference_lines.append(zero_line)
            
            # Threshold lines for this channel (if configured in config.py AMPLITUDE_THRESHOLD)
            if threshold is not None:
                # Negative threshold line (e.g., -35)
                threshold_neg_y = offset + threshold
                threshold_neg_line = hv.HLine(threshold_neg_y).opts(color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
                reference_lines.append(threshold_neg_line)
                # Positive threshold line (e.g., +35)
                threshold_pos_y = offset - threshold  # threshold is negative, so -threshold gives positive
                threshold_pos_line = hv.HLine(threshold_pos_y).opts(color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
                reference_lines.append(threshold_pos_line)
        
        # Create combined plot (background spans behind curves, then regions, then reference lines)
        elements = []
        elements.extend(vspans)
        elements.extend(curves)
        elements.extend(reference_lines)
        combined = hv.Overlay(elements) * regions_rects
        # Set the range directly on the element to ensure it's preserved
        combined = combined.redim.range(y=(y_min, y_max))
        t4 = time_module.time()
        
        # Hook to set y-axis range dynamically and prevent auto-scaling
        # Use local variables to ensure this plot's range is independent from butterfly plot
        focus_y_min = y_min
        focus_y_max = y_max

        def apply_stacked_yaxis_format(plot):
            """Apply repeated -range..range tick labels for stacked channels."""
            if hasattr(plot, 'state') and hasattr(plot.state, 'yaxis') and plot.state.yaxis:
                from bokeh.models import FixedTicker, FuncTickFormatter

                ticks = []
                for i_stack in range(len(valid_channels)):
                    base = i_stack * offset_per_channel
                    for v in (-channel_range, 0, channel_range):
                        ticks.append(base + v)

                plot.state.yaxis[0].ticker = FixedTicker(ticks=ticks)
                
                # Get channel names from config (reversed to match display order: top to bottom)
                ch_names = getattr(config, 'CH_NAMES', None)
                if ch_names and len(ch_names) >= len(valid_channels):
                    # Reverse to match: i_stack=0 is bottom (last config entry), i_stack=N-1 is top (first config entry)
                    ch_names_reversed = list(reversed(ch_names[:len(valid_channels)]))
                    ch_names_js = str(ch_names_reversed)  # Convert to JS array string
                else:
                    ch_names_js = "null"
                
                # === CHANGE CHANNEL NAMES (3-CHANNEL PLOT Y-AXIS) ===
                plot.state.yaxis[0].formatter = FuncTickFormatter(code=f"""
                    const offset = {offset_per_channel};
                    const ch_names = {ch_names_js};
                    const n_channels = {len(valid_channels)};
                    let i_stack = Math.floor(tick / offset);
                    let v = tick - i_stack * offset;
                    if (v > offset/2) {{ v -= offset; i_stack += 1; }}
                    // At zero line, show channel name
                    if (Math.abs(v) < 1 && ch_names && i_stack >= 0 && i_stack < n_channels) {{
                        return ch_names[i_stack] + ": 0";
                    }}
                    return v.toFixed(0);
                """)
                plot.state.yaxis[0].axis_label = "Amplitude (normalized)"
        
        def set_y_range(plot, element):
            """Hook to set y-axis range dynamically and lock it - 3-CHANNEL PLOT ONLY"""
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range_handle = plot.handles['y_range']
                # Force set the range immediately - use local variables for 3-channel plot
                # This ensures independence from butterfly plot
                y_range_handle.start = focus_y_min
                y_range_handle.end = focus_y_max
                y_range_handle.bounds = (focus_y_min, focus_y_max)
                # Disable auto-range - this is critical
                if hasattr(y_range_handle, 'auto_range'):
                    y_range_handle.auto_range = False
                if hasattr(y_range_handle, 'reset_start'):
                    y_range_handle.reset_start = focus_y_min
                    y_range_handle.reset_end = focus_y_max
                
                # Add a post-render callback to ensure range stays locked
                # This runs after Bokeh renders the plot
                if hasattr(plot, 'state') and hasattr(plot.state, 'document'):
                    # Use a document callback to lock range after render
                    def lock_after_render():
                        try:
                            y_range_handle.start = focus_y_min
                            y_range_handle.end = focus_y_max
                            y_range_handle.bounds = (focus_y_min, focus_y_max)
                            if hasattr(y_range_handle, 'auto_range'):
                                y_range_handle.auto_range = False
                        except:
                            pass
                    
                    # Schedule callback after next render
                    if hasattr(plot.state.document, 'add_next_tick_callback'):
                        plot.state.document.add_next_tick_callback(lock_after_render)
                
                # Also set on plot state if available
                if hasattr(plot, 'state') and hasattr(plot.state, 'y_range'):
                    plot.state.y_range.start = focus_y_min
                    plot.state.y_range.end = focus_y_max
                    plot.state.y_range.bounds = (focus_y_min, focus_y_max)
                    if hasattr(plot.state.y_range, 'auto_range'):
                        plot.state.y_range.auto_range = False
                # Ensure tick labels remain correct even after plot updates
                apply_stacked_yaxis_format(plot)
        
        # Additional hook to re-lock range after any update - runs after set_y_range
        # This ensures the range stays locked even if something tries to change it
        def lock_y_range(plot, element):
            """Re-lock y-axis range after plot updates - 3-CHANNEL PLOT ONLY"""
            if hasattr(plot, 'handles') and 'y_range' in plot.handles:
                y_range_handle = plot.handles['y_range']
                # Always force range back to 3-channel plot values - use local variables
                # Don't check, just force it to be correct for this plot only
                y_range_handle.start = focus_y_min
                y_range_handle.end = focus_y_max
                y_range_handle.bounds = (focus_y_min, focus_y_max)
                if hasattr(y_range_handle, 'auto_range'):
                    y_range_handle.auto_range = False
                if hasattr(y_range_handle, 'reset_start'):
                    y_range_handle.reset_start = focus_y_min
                    y_range_handle.reset_end = focus_y_max
            # Also update plot state if available
            if hasattr(plot, 'state') and hasattr(plot.state, 'y_range'):
                plot.state.y_range.start = focus_y_min
                plot.state.y_range.end = focus_y_max
                plot.state.y_range.bounds = (focus_y_min, focus_y_max)
                if hasattr(plot.state.y_range, 'auto_range'):
                    plot.state.y_range.auto_range = False
            # Re-apply stacked y-axis formatting; Bokeh can reset it after selection updates
            apply_stacked_yaxis_format(plot)
        
        # Hook to ensure box_select tool is enabled
        def ensure_box_select(plot, element):
            """Ensure box_select tool is enabled on the plot"""
            if hasattr(plot, 'state') and hasattr(plot.state, 'toolbar'):
                for tool in plot.state.toolbar.tools:
                    if hasattr(tool, 'name') and tool.name == 'box_select':
                        if not hasattr(tool, 'active') or not tool.active:
                            tool.active = True
                        break
        
        # Hook to optimize selection tool
        def optimize_selection(plot, element):
            """Hook to restrict BoxSelectTool to only the first renderer."""
            from bokeh.models import BoxSelectTool
            from bokeh.models.glyphs import Line

            def apply_box_select_renderers():
                box_select = None
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            box_select = tool
                            break
                if not box_select:
                    return
                renderers = []
                if hasattr(plot, 'handles') and 'glyph_renderers' in plot.handles:
                    renderers = list(plot.handles.get('glyph_renderers') or [])
                if not renderers and hasattr(plot, 'state') and hasattr(plot.state, 'renderers'):
                    renderers = list(plot.state.renderers)
                if not renderers:
                    return
                line_renderer = None
                for r in renderers:
                    try:
                        if hasattr(r, 'glyph') and isinstance(r.glyph, Line):
                            line_renderer = r
                            break
                    except Exception:
                        continue
                if line_renderer is None:
                    line_renderer = renderers[0]
                box_select.renderers = [line_renderer]

            apply_box_select_renderers()
            if hasattr(plot, 'state') and hasattr(plot.state, 'document') and plot.state.document:
                try:
                    plot.state.document.add_next_tick_callback(apply_box_select_renderers)
                except Exception:
                    pass

        # Manually push BoxSelectTool overlay to stream (HoloViews link can break on reload)
        def link_box_select_to_stream_focus(plot, element):
            from bokeh.models import BoxSelectTool
            if not hasattr(self, '_bounds_stream_focus') or self._bounds_stream_focus is None:
                return

            def attach():
                box_select = None
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            box_select = tool
                            break
                if not box_select or not hasattr(box_select, 'overlay') or box_select.overlay is None:
                    return
                stream_ref = self._bounds_stream_focus

                def push_bounds_to_stream():
                    o = box_select.overlay
                    left = getattr(o, 'left', None)
                    right = getattr(o, 'right', None)
                    top = getattr(o, 'top', None)
                    bottom = getattr(o, 'bottom', None)
                    if left is not None and right is not None and top is not None and bottom is not None:
                        x0, x1 = min(left, right), max(left, right)
                        y0, y1 = min(top, bottom), max(top, bottom)
                        try:
                            stream_ref.event(bounds=(x0, y0, x1, y1))
                        except Exception:
                            pass

                for prop in ('left', 'right', 'top', 'bottom'):
                    try:
                        box_select.overlay.on_change(prop, lambda a, o, n: push_bounds_to_stream())
                    except Exception:
                        pass

            attach()
            if hasattr(plot, 'state') and hasattr(plot.state, 'document') and plot.state.document:
                try:
                    plot.state.document.add_next_tick_callback(attach)
                except Exception:
                    pass
        
        # Hook to clear box selection when epoch changes - 3-CHANNEL PLOT
        def clear_box_selection_focus(plot, element):
            """Clear BoxSelectTool visual selection - 3-CHANNEL PLOT"""
            from bokeh.models import BoxSelectTool
            
            # Only clear if epoch_index actually changed (not just update_trigger)
            if self._should_clear_box_selection_focus:
                if hasattr(plot, 'state') and hasattr(plot.state, 'tools'):
                    for tool in plot.state.tools:
                        if isinstance(tool, BoxSelectTool):
                            # Clear the selection overlay (visual box)
                            if hasattr(tool, 'overlay'):
                                tool.overlay.left = None
                                tool.overlay.right = None
                                tool.overlay.top = None
                                tool.overlay.bottom = None
                            # Clear any selection on renderers
                            if hasattr(tool, 'renderers') and tool.renderers:
                                for renderer in tool.renderers:
                                    if hasattr(renderer, 'data_source'):
                                        # Clear selection indices
                                        if hasattr(renderer.data_source, 'selected'):
                                            renderer.data_source.selected.indices = []
                            break
                # Clear the flag after clearing
                self._should_clear_box_selection_focus = False
        
        # Build hooks list - ensure bounds stream update happens after range is set
        hooks_list = [set_y_range, lock_y_range, clear_box_selection_focus, ensure_box_select, optimize_selection, link_box_select_to_stream_focus, self._configure_hover_tool_hook]
        
        # X range: exact data range (0 to display_duration_s) with no extra padding
        display_duration_s = len(display_df) / self.sampling_rate
        
        # === CHANGE Y-AXIS SCALE (3-CHANNEL PLOT HEIGHT) ===
        focus_plot_height = int(self.plot_height * 1.5)  # Scale height by 1.5x
        
        # Build tools list
        focus_tools_list = ['tap', 'xwheel_zoom', 'xpan', 'box_select']
        
        plot_opts = opts.Overlay(
            height=focus_plot_height, shared_axes=True, show_legend=False,
            responsive=True,  # Make plot responsive to container width
            tools=focus_tools_list,
            active_tools=['tap', 'box_select'],
            hooks=hooks_list,
            ylim=(focus_y_min, focus_y_max),  # Explicitly set ylim to prevent auto-scaling - 3-CHANNEL PLOT ONLY
            xlim=(0, display_duration_s),  # No white space before 0 or after display end
            framewise=False,  # Critical: prevent auto-scaling on updates
        )
        
        t5 = time_module.time()
        total_time = (t5 - t0) * 1000
        if total_time > 50:
            print(f"⏱️ create_focus_channels_plot: get_epoch_data={((t1-t0)*1000):.1f}ms, curves={((t2-t1)*1000):.1f}ms, regions={((t3-t2)*1000):.1f}ms, overlay={((t4-t3)*1000):.1f}ms, opts={((t5-t4)*1000):.1f}ms, TOTAL={total_time:.1f}ms")
        
        return combined.opts(plot_opts)

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
        curve = hv.Curve((tt, dd)).opts(color=color, line_width=1)
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
            opts.Overlay(width=self.plot_width, height=150, shared_axes=True, xaxis=None, ylabel=f"Ch {channel_idx}", hooks=[set_y_range_focus])
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
        
        # Epoch jump input
        max_epoch = max(0, self.epoch_manager.get_epoch_count() - 1)
        epoch_jump_input = pn.widgets.IntInput(
            name='Epoch',
            value=self.epoch_index,
            start=0,
            end=max_epoch,
            width=80,
            step=1
        )
        # Sync input with epoch_index changes
        self.param.watch(lambda e: setattr(epoch_jump_input, 'value', e.new), 'epoch_index')
        # Jump to epoch when input changes
        def jump_to_epoch(event):
            try:
                new_idx = int(event.new)
                if 0 <= new_idx <= max_epoch:
                    self.epoch_index = new_idx
                else:
                    # Reset to current if out of bounds
                    epoch_jump_input.value = self.epoch_index
            except (ValueError, TypeError):
                # Reset to current if invalid
                epoch_jump_input.value = self.epoch_index
        epoch_jump_input.param.watch(jump_to_epoch, 'value')
        
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

        # Create bounds streams for box selection without source (HoloViews will auto-link to plots)
        if not hasattr(self, '_bounds_stream_focus') or self._bounds_stream_focus is None:
            self._bounds_stream_focus = streams.BoundsXY()
            self._bounds_stream_focus.add_subscriber(self._on_box_select)
        
        if not hasattr(self, '_bounds_stream') or self._bounds_stream is None:
            self._bounds_stream = streams.BoundsXY()
            self._bounds_stream.add_subscriber(self._on_box_select)
        
        # Create DynamicMaps with streams - HoloViews will automatically link BoundsXY to the plots
        focus_channels_dmap = hv.DynamicMap(self.create_focus_channels_plot, streams=[self.tap_stream, self._bounds_stream_focus])
        main_dmap = hv.DynamicMap(self.create_main_plot, streams=[self.tap_stream, self._bounds_stream])
        
        # Compact layout with controls in a single row
        controls_row = pn.Row(
            btn_prev, btn_next,
            pn.Spacer(width=10),
            epoch_jump_input,
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
        
        # Bottom row with topoplot panel
        bottom_row = pn.Row(
            self.topoplot_panel,
            sizing_mode='stretch_width'
        )
        
        return pn.Column(
            self.status_bar,  # Status bar at top
            controls_row,  # Compact controls row
            # IMPORTANT: disable cross-pane axis linking, otherwise the butterfly plot
            # can inherit the stacked plot's y-range (e.g. -230..1230).
            pn.pane.HoloViews(focus_channels_dmap, sizing_mode='stretch_width', linked_axes=False),  # NEW: stacked 3-channel plot
            pn.pane.HoloViews(main_dmap, sizing_mode='stretch_width', linked_axes=False),  # MOVED: butterfly plot
            bottom_row,  # Topoplot panel
            self.kb_listener,  # <--- INVISIBLE LISTENER COMPONENT
            style,
            sizing_mode='stretch_width',
            margin=(0, 10)
        )

def create_dashboard(epoch_manager, annotation_manager, focus_channels=None, main_plot_channels=None, chanlocs=None, channels_file=None, exclude_channels=None, plot_width: int = 1200, plot_height: int = 500):
    """Create dashboard instance."""
    db = EEGDashboard(
        epoch_manager,
        annotation_manager,
        focus_channels,
        main_plot_channels,
        chanlocs,
        channels_file,
        exclude_channels,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    return db.view()
