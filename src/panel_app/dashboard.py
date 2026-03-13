import panel as pn
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
from src.preprocessing import apply_bandpass_filter

from bokeh.plotting import figure as bk_figure
from bokeh.models import ColumnDataSource, BoxSelectTool, TapTool, HoverTool
from bokeh.models import Range1d, FixedTicker, CustomJSTickFormatter
from bokeh.events import SelectionGeometry, Tap

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available. Topoplots will be disabled.")

# Initialize extensions
pn.extension('tabulator', sizing_mode='stretch_width')

# --- CONFIGURATION ---
DEBUG = False


def debug_print(*args, **kwargs):
    """Debug logger for hot-path instrumentation."""
    if DEBUG:
        print(*args, **kwargs)


# === CHANGE ANNOTATION COLORS ===
# Modify these to change the color of annotated regions
ANNOTATION_COLORS = {
    'unannotated': '#95a5a6',  # Gray - color for unannotated regions
    'KC': '#27ae60',           # Green - color for KC-labeled regions
}

# === CHANGE CONTEXT DISPLAY SETTINGS ===
# Display settings: show extra context around each epoch
# Epoch progression/hop remains controlled by EpochManager.epoch_length_sec (e.g. 20s).
DISPLAY_CONTEXT_BEFORE_SEC = 2.0   # Seconds of context shown before epoch
DISPLAY_CONTEXT_AFTER_SEC = 2.0    # Seconds of context shown after epoch
CONTEXT_BACKGROUND_COLOR = '#cfe8ff'  # Light blue - background color for context regions

def downsample_minmax(data: np.ndarray, time: np.ndarray, max_points: int = 2000) -> tuple:
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


# --- DASHBOARD CLASS ---
class EEGDashboard(param.Parameterized):
    # Reactive Params
    epoch_index = param.Integer(default=0, bounds=(0, None))
    selected_region_id = param.Integer(default=-1)
    current_label = param.Selector(objects=['KC', 'unannotated'], default='unannotated')
    show_bandpass_overlay = param.Boolean(default=False)
    update_trigger = param.Integer(default=0)
    focus_plot_trigger = param.Integer(default=0)
    topoplot_popup = param.Parameter(default=None)
    topoplot_trigger = param.Integer(default=0)
    status_trigger = param.Integer(default=0)

    def __init__(self, epoch_manager, annotation_manager,
                 focus_channels: List[int] = None,
                 main_plot_channels: List[int] = None,
                 chanlocs: pd.DataFrame = None,
                 channels_file: Optional[str] = None,
                 exclude_channels: List[int] = None,
                 plot_width: int = 1200,
                 plot_height: int = 500,
                 **params):

        # --- Set attributes BEFORE super().__init__ ---
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
        # Cache for spectral power data (0.5-2 Hz) per epoch
        self._spectral_power_cache = {}
        self._spectral_power_in_progress = set()
        self._spectral_power_lock = threading.Lock()

        # Debounce mechanism
        self._update_timer = None
        self._pending_update = False
        self._topoplot_timer = None

        # Cache for epoch_data
        self._cached_epoch_data = None
        self._cached_epoch_data_index = -1
        self._cached_regions = None
        self._cached_epoch_start = None

        # Cache for focus bandpass
        self._focus_bandpass_cache = {}

        # Click handler debouncing
        self._last_click_time = 0
        self._last_click_region = None
        self._click_debounce_time = 0.3
        self._processing_click = False

        # Guard flag to prevent duplicate annotation writes
        self._skip_label_change_handler = False

        # Track epoch changes
        self._previous_epoch_index = -1

        # Bokeh figure references (created in _init_figures)
        self._main_fig = None
        self._focus_fig = None
        self._main_line_sources = []
        self._focus_line_sources = []
        self._focus_bandpass_sources = []
        self._main_rect_source = ColumnDataSource(data=self._empty_rect_data())
        self._focus_rect_source = ColumnDataSource(data=self._empty_rect_data())
        self._main_context_source = ColumnDataSource(data={'left': [], 'right': [], 'top': [], 'bottom': [], 'color': [], 'alpha': []})
        self._focus_context_source = ColumnDataSource(data={'left': [], 'right': [], 'top': [], 'bottom': [], 'color': [], 'alpha': []})
        self._main_ref_line_sources = []
        self._focus_ref_line_sources = []

        # Fixed topoplot HTML pane (avoids "Models must be owned by only a single document")
        self._topoplot_html_pane = pn.pane.HTML(
            "<div style='width:200px;height:200px;display:flex;align-items:center;justify-content:center;"
            "color:#aaa;font-size:11px;border:1px dashed #ddd;border-radius:4px;'>Select a region</div>",
            width=200, height=200
        )

        # Bokeh document reference for thread-safe callbacks (set in view())
        self._bokeh_doc = None

        # Now call super, which might trigger watchers immediately
        super().__init__(**params)

        self.param.epoch_index.bounds = (0, max(0, epoch_manager.get_epoch_count() - 1))
        self._previous_epoch_index = self.epoch_index

        # Initialize Keyboard Listener
        self.kb_listener = KeyboardListener()
        self.kb_listener.param.watch(self._handle_kb_event, 'key')

        # Watch selected_region_id to trigger delayed topoplot update
        self.param.watch(self._schedule_topoplot_update, 'selected_region_id')

        # Watch epoch_index and update_trigger to refresh plot data
        self.param.watch(self._on_epoch_or_update_change, ['epoch_index', 'update_trigger'])
        # Watch show_bandpass_overlay
        self.param.watch(self._on_bandpass_toggle, ['show_bandpass_overlay'])

        # Build Bokeh figures
        self._init_figures()
        # Initial data load
        self._full_update()

    @staticmethod
    def _empty_rect_data():
        return {
            'left': [], 'right': [], 'top': [], 'bottom': [],
            'fill_color': [], 'line_color': [], 'line_width': [], 'alpha': [],
            'region_id': [], 'status': [],
            'duration_html': [], 'max_diff_text': [], 'topo_html': [],
        }

    # === CHANGE DEFAULT CHANNEL COLOR ===
    def _get_channel_color(self, channel_idx: int) -> str:
        """Get color for a channel, using channel colors if available."""
        if channel_idx in self.channel_colors:
            r, g, b = self.channel_colors[channel_idx]
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#34495e'  # Default dark gray

    def _create_time_axis(self, n):
        return np.arange(n) / self.sampling_rate

    # ------------------------------------------------------------------ #
    #  Bokeh figure initialization (called ONCE)
    # ------------------------------------------------------------------ #
    def _init_figures(self):
        """Create Bokeh figures once. All subsequent updates go through ColumnDataSource."""
        self._init_main_figure()
        self._init_focus_figure()

    def _get_main_channel_indices(self, n_total):
        """Determine which channels to plot in the butterfly figure."""
        if self.main_plot_channels is not None:
            indices = [i for i in self.main_plot_channels if 0 <= i < n_total]
            if not indices:
                indices = list(range(min(30, n_total)))
        else:
            target = 30
            if n_total > target:
                step = max(1, n_total // target)
                indices = list(range(0, n_total, step))
            else:
                indices = list(range(n_total))
        return indices

    def _init_main_figure(self):
        """Create the butterfly (main) Bokeh figure with line renderers and rect glyphs."""
        # === CHANGE Y-AXIS RANGE (BUTTERFLY PLOT) ===
        y_min, y_max = -230, 230

        fig = bk_figure(
            height=self.plot_height,
            sizing_mode='stretch_width',
            tools='xwheel_zoom,xpan,reset',
            active_scroll='xwheel_zoom',
            y_range=Range1d(y_min, y_max, bounds=(y_min, y_max)),
            x_range=Range1d(0, 1),
            output_backend='webgl',
        )
        fig.toolbar.logo = None
        fig.xaxis.axis_label = 'Time (s)'
        fig.yaxis.axis_label = 'Channels (stacked)'

        # Context shading (pre/post epoch)
        fig.quad(source=self._main_context_source,
                 left='left', right='right', top='top', bottom='bottom',
                 fill_color='color', fill_alpha='alpha', line_width=0, level='underlay')

        # Channel lines
        n_total = len(self.epoch_manager.eeg_data.columns)
        indices = self._get_main_channel_indices(n_total)

        self._main_line_sources = []
        for ch_idx in indices:
            src = ColumnDataSource(data={'x': [], 'y': []})
            self._main_line_sources.append(src)
            color = self._get_channel_color(ch_idx)
            fig.line('x', 'y', source=src, color=color, line_width=0.6, alpha=0.7)

        self._main_channel_indices = indices

        # Reference lines: zero line + optional threshold
        zero_src = ColumnDataSource(data={'x': [0, 1], 'y': [0, 0]})
        fig.line('x', 'y', source=zero_src, color='gray', line_width=1, line_dash='dashed', alpha=0.5)
        self._main_ref_line_sources = [zero_src]

        threshold = getattr(config, 'AMPLITUDE_THRESHOLD', None)
        if threshold is not None:
            thr_src = ColumnDataSource(data={'x': [0, 1], 'y': [threshold, threshold]})
            fig.line('x', 'y', source=thr_src, color='red', line_width=1, line_dash='dashed', alpha=0.7)
            self._main_ref_line_sources.append(thr_src)

        # Region rectangles
        rect_renderer = fig.quad(source=self._main_rect_source,
                 left='left', right='right', top='top', bottom='bottom',
                 fill_color='fill_color', fill_alpha='alpha',
                 line_color='line_color', line_width='line_width',
                 level='overlay')

        # HoverTool for rectangles
        hover = HoverTool(
            tooltips="""
            <div style="font-size: 12px;">
                <strong>Region @region_id</strong><br>
                Status: @status<br>
                Time: @left{0.00}s - @right{0.00}s<br>
                Duration: @duration_html{safe}<br>
                Max diff: @max_diff_text<br>
                @topo_html{safe}
            </div>
            """,
            point_policy='follow_mouse',
            attachment='above',
            renderers=[rect_renderer],
        )
        fig.add_tools(hover)

        # BoxSelectTool: renderers=[] so hit-test doesn't clamp the visual overlay,
        # SelectionGeometry event still fires for our custom handler.
        box_select = BoxSelectTool(renderers=[])
        fig.add_tools(box_select)
        fig.toolbar.active_drag = box_select

        # TapTool
        tap_tool = TapTool()
        fig.add_tools(tap_tool)

        # Wire events
        fig.on_event(SelectionGeometry, self._on_bokeh_box_select)
        fig.on_event(Tap, self._on_bokeh_tap)

        self._main_fig = fig

    def _init_focus_figure(self):
        """Create the 3-channel stacked focus Bokeh figure."""
        n_channels = len(self.focus_channels)
        offset_per_channel = 500
        channel_range = 200
        y_min = -channel_range
        y_max = (n_channels - 1) * offset_per_channel + channel_range
        focus_height = int(self.plot_height * 1.5)

        fig = bk_figure(
            height=focus_height,
            sizing_mode='stretch_width',
            tools='xwheel_zoom,xpan,reset',
            active_scroll='xwheel_zoom',
            y_range=Range1d(y_min, y_max, bounds=(y_min, y_max)),
            x_range=Range1d(0, 1),
            output_backend='webgl',
        )
        fig.toolbar.logo = None
        fig.xaxis.axis_label = 'Time (s)'

        # Context shading
        fig.quad(source=self._focus_context_source,
                 left='left', right='right', top='top', bottom='bottom',
                 fill_color='color', fill_alpha='alpha', line_width=0, level='underlay')

        # Focus channel lines
        n_total = len(self.epoch_manager.eeg_data.columns)
        valid_channels = [ch for ch in self.focus_channels if 0 <= ch < n_total]
        self._focus_valid_channels = valid_channels

        self._focus_line_sources = []
        for ch_idx in valid_channels:
            src = ColumnDataSource(data={'x': [], 'y': []})
            self._focus_line_sources.append(src)
            color = self._get_channel_color(ch_idx)
            fig.line('x', 'y', source=src, color=color, line_width=0.5, alpha=1.0)

        # Bandpass overlay lines (toggled by show_bandpass_overlay)
        self._focus_bandpass_sources = []
        for _ in valid_channels:
            src = ColumnDataSource(data={'x': [], 'y': []})
            self._focus_bandpass_sources.append(src)
            fig.line('x', 'y', source=src, color='#2980b9', line_width=1.0, alpha=0.6)

        # Reference lines for each channel: zero line + threshold lines
        self._focus_ref_line_sources = []
        threshold = getattr(config, 'AMPLITUDE_THRESHOLD', None)
        for i_idx in range(len(valid_channels)):
            offset = (len(valid_channels) - 1 - i_idx) * offset_per_channel
            # Zero line
            zsrc = ColumnDataSource(data={'x': [0, 1], 'y': [offset, offset]})
            fig.line('x', 'y', source=zsrc, color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
            self._focus_ref_line_sources.append(zsrc)
            if threshold is not None:
                neg_src = ColumnDataSource(data={'x': [0, 1], 'y': [offset + threshold, offset + threshold]})
                fig.line('x', 'y', source=neg_src, color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
                self._focus_ref_line_sources.append(neg_src)
                pos_src = ColumnDataSource(data={'x': [0, 1], 'y': [offset - threshold, offset - threshold]})
                fig.line('x', 'y', source=pos_src, color='gray', line_width=0.7, line_dash='solid', alpha=0.5)
                self._focus_ref_line_sources.append(pos_src)

        # Region rectangles
        rect_renderer = fig.quad(source=self._focus_rect_source,
                 left='left', right='right', top='top', bottom='bottom',
                 fill_color='fill_color', fill_alpha='alpha',
                 line_color='line_color', line_width='line_width',
                 level='overlay')

        # HoverTool for rectangles
        hover = HoverTool(
            tooltips="""
            <div style="font-size: 12px;">
                <strong>Region @region_id</strong><br>
                Status: @status<br>
                Time: @left{0.00}s - @right{0.00}s<br>
                Duration: @duration_html{safe}<br>
                Max diff: @max_diff_text<br>
                @topo_html{safe}
            </div>
            """,
            point_policy='follow_mouse',
            attachment='above',
            renderers=[rect_renderer],
        )
        fig.add_tools(hover)

        # BoxSelectTool: renderers=[] so hit-test doesn't clamp the visual overlay,
        # SelectionGeometry event still fires for our custom handler.
        box_select = BoxSelectTool(renderers=[])
        fig.add_tools(box_select)
        fig.toolbar.active_drag = box_select

        # TapTool
        tap_tool = TapTool()
        fig.add_tools(tap_tool)

        # Wire events
        fig.on_event(SelectionGeometry, self._on_bokeh_box_select)
        fig.on_event(Tap, self._on_bokeh_tap)

        # Custom y-axis formatting for stacked channels
        ticks = []
        for i_stack in range(len(valid_channels)):
            base = i_stack * offset_per_channel
            for v in (-channel_range, 0, channel_range):
                ticks.append(base + v)
        fig.yaxis[0].ticker = FixedTicker(ticks=ticks)

        ch_names = getattr(config, 'CH_NAMES', None)
        if ch_names and len(ch_names) >= len(valid_channels):
            ch_names_reversed = list(reversed(ch_names[:len(valid_channels)]))
            ch_names_js = str(ch_names_reversed)
        else:
            ch_names_js = "null"

        fig.yaxis[0].formatter = CustomJSTickFormatter(code=f"""
            const offset = {offset_per_channel};
            const ch_names = {ch_names_js};
            const n_channels = {len(valid_channels)};
            let i_stack = Math.floor(tick / offset);
            let v = tick - i_stack * offset;
            if (v > offset/2) {{ v -= offset; i_stack += 1; }}
            if (Math.abs(v) < 1 && ch_names && i_stack >= 0 && i_stack < n_channels) {{
                return ch_names[i_stack] + ": 0";
            }}
            return v.toFixed(0);
        """)
        fig.yaxis[0].axis_label = "Amplitude (uV)"

        self._focus_fig = fig

    # ------------------------------------------------------------------ #
    #  Data helpers
    # ------------------------------------------------------------------ #
    def _run_on_doc_thread(self, fn):
        """
        Run fn() safely on the Bokeh document thread.
        If a document reference is stored, schedules via add_next_tick_callback.
        Otherwise calls fn() directly (safe during init before serving).
        """
        doc = self._bokeh_doc
        if doc is not None:
            doc.add_next_tick_callback(fn)
        else:
            fn()

    def _debounced_update(self, delay=0.01):
        """Debounced update trigger to prevent rapid-fire plot recreations."""
        if self._update_timer is not None:
            self._update_timer.cancel()
        self._pending_update = True

        def do_update():
            if self._pending_update:
                self._pending_update = False
                self._run_on_doc_thread(lambda: setattr(self, 'update_trigger', self.update_trigger + 1))
            self._update_timer = None

        self._update_timer = threading.Timer(delay, do_update)
        self._update_timer.daemon = True
        self._update_timer.start()

    def _schedule_topoplot_update(self, event=None):
        """Schedule a delayed update for the topoplot to avoid blocking UI."""
        if self._topoplot_timer is not None:
            self._topoplot_timer.cancel()

        def do_update():
            self._run_on_doc_thread(lambda: setattr(self, 'topoplot_trigger', self.topoplot_trigger + 1))

        self._topoplot_timer = threading.Timer(0.3, do_update)
        self._topoplot_timer.daemon = True
        self._topoplot_timer.start()

    def _get_epoch_data(self):
        """Get current epoch data and regions (uses cache to avoid redundant calls)."""
        epoch_data_cached = (self._cached_epoch_data_index == self.epoch_index and
                            self._cached_epoch_data is not None)

        if epoch_data_cached and self._cached_regions is not None:
            return self._cached_epoch_data, self._cached_regions, self._cached_epoch_start

        if not epoch_data_cached:
            self.epoch_manager.current_epoch_idx = self.epoch_index
            epoch_data = self.epoch_manager.get_current_epoch()
            epoch_info = self.epoch_manager.get_current_epoch_info()

            self._cached_epoch_data = epoch_data
            self._cached_epoch_data_index = self.epoch_index
            self._cached_epoch_start = int(epoch_info['start_idx'])
            self._cached_regions = None

            self._schedule_spectral_power_compute(self.epoch_index, epoch_data)
        else:
            epoch_info = self.epoch_manager.get_current_epoch_info()
            epoch_data = self._cached_epoch_data

        if self._cached_regions is None:
            epoch_start = int(epoch_info['start_idx'])
            epoch_end = int(epoch_info['end_idx'])
            current_regions = self.annotation_manager.get_regions_for_epoch(epoch_start, epoch_end)
            self._cached_regions = current_regions
        else:
            current_regions = self._cached_regions

        return epoch_data, current_regions, self._cached_epoch_start

    def _schedule_spectral_power_compute(self, epoch_index: int, epoch_data: pd.DataFrame):
        """Schedule per-epoch spectral power computation in the background."""
        if epoch_data is None or len(epoch_data) == 0:
            return

        with self._spectral_power_lock:
            if epoch_index in self._spectral_power_cache:
                return
            if epoch_index in self._spectral_power_in_progress:
                return
            self._spectral_power_in_progress.add(epoch_index)

        epoch_data_copy = epoch_data.copy()

        def worker():
            try:
                self._compute_epoch_spectral_power(epoch_data_copy, epoch_index=epoch_index)
            finally:
                with self._spectral_power_lock:
                    self._spectral_power_in_progress.discard(epoch_index)
                if self.epoch_index == epoch_index:
                    self._run_on_doc_thread(
                        lambda: setattr(self, 'topoplot_trigger', self.topoplot_trigger + 1)
                    )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _get_display_window_data(self) -> Tuple[pd.DataFrame, int, int, float, float]:
        """Get a display window around the current epoch with context."""
        if (hasattr(self, '_cached_display_epoch_index') and
            self._cached_display_epoch_index == self.epoch_index and
            getattr(self, '_cached_display_df', None) is not None):
            return (self._cached_display_df,
                    self._cached_display_pre_samples,
                    self._cached_display_post_samples,
                    self._cached_display_center_start_time_s,
                    self._cached_display_center_end_time_s)

        self.epoch_manager.current_epoch_idx = self.epoch_index
        epoch_info = self.epoch_manager.get_current_epoch_info()
        if not epoch_info:
            empty = pd.DataFrame()
            return empty, 0, 0, 0.0, 0.0

        epoch_start = int(epoch_info['start_idx'])
        epoch_end = int(epoch_info['end_idx'])

        sr = float(self.sampling_rate)
        pre_target = int(DISPLAY_CONTEXT_BEFORE_SEC * sr)
        post_target = int(DISPLAY_CONTEXT_AFTER_SEC * sr)

        n_samples_total = len(self.epoch_manager.eeg_data)
        win_start = max(0, epoch_start - pre_target)
        win_end = min(n_samples_total - 1, epoch_end + post_target)

        display_df = self.epoch_manager.eeg_data.iloc[win_start:win_end + 1].copy()

        pre_samples = int(epoch_start - win_start)
        post_samples = int(win_end - epoch_end)

        epoch_len_samples = int(epoch_end - epoch_start + 1)
        center_start_time_s = pre_samples / sr
        center_end_time_s = (pre_samples + epoch_len_samples) / sr

        self._cached_display_epoch_index = self.epoch_index
        self._cached_display_df = display_df
        self._cached_display_pre_samples = pre_samples
        self._cached_display_post_samples = post_samples
        self._cached_display_center_start_time_s = center_start_time_s
        self._cached_display_center_end_time_s = center_end_time_s

        return display_df, pre_samples, post_samples, center_start_time_s, center_end_time_s

    # ------------------------------------------------------------------ #
    #  Bokeh-native data update methods
    # ------------------------------------------------------------------ #
    def _on_epoch_or_update_change(self, *events):
        """Called when epoch_index or update_trigger changes — refreshes all plot data."""
        self._full_update()

    def _on_bandpass_toggle(self, *events):
        """Called when bandpass overlay is toggled."""
        self._update_focus_bandpass()

    def _full_update(self):
        """Refresh all ColumnDataSource data for both figures."""
        t0 = time.time()

        epoch_changed = (self._cached_epoch_data_index != self.epoch_index)
        if epoch_changed:
            self._cached_epoch_data = None
            self._cached_regions = None
            self._cached_epoch_start = None
            self._cached_display_epoch_index = -1
            self._cached_display_df = None

        epoch_data, current_regions, _ = self._get_epoch_data()
        display_df, pre_samples, post_samples, center_start_time_s, center_end_time_s = self._get_display_window_data()

        if epoch_data is None or len(epoch_data) == 0 or display_df is None or len(display_df) == 0:
            return

        t = self._create_time_axis(len(display_df))
        display_duration_s = len(display_df) / self.sampling_rate

        # Update x_range on both figures
        if self._main_fig is not None:
            self._main_fig.x_range.start = 0
            self._main_fig.x_range.end = display_duration_s
        if self._focus_fig is not None:
            self._focus_fig.x_range.start = 0
            self._focus_fig.x_range.end = display_duration_s

        # --- Main figure: channel lines ---
        for i_idx, ch_idx in enumerate(self._main_channel_indices):
            if i_idx >= len(self._main_line_sources):
                break
            col = display_df.columns[ch_idx] if ch_idx < len(display_df.columns) else display_df.columns[0]
            d = display_df[col].values
            d_normalized = (d - np.mean(d)) / (np.std(d) or 1) * 30
            tt, dd = downsample_minmax(d_normalized, t)
            self._main_line_sources[i_idx].data = {'x': tt, 'y': dd}

        # --- Main figure: reference lines ---
        for src in self._main_ref_line_sources:
            old_y = list(src.data['y'])
            src.data = {'x': [0, display_duration_s], 'y': old_y[:2] if len(old_y) >= 2 else [0, 0]}

        # --- Focus figure: channel lines ---
        n_valid = len(self._focus_valid_channels)
        offset_per_channel = 500
        for i_idx, ch_idx in enumerate(self._focus_valid_channels):
            if i_idx >= len(self._focus_line_sources):
                break
            col = display_df.columns[ch_idx] if ch_idx < len(display_df.columns) else display_df.columns[0]
            d = display_df[col].values
            offset = (n_valid - 1 - i_idx) * offset_per_channel
            d_offset = d + offset
            tt, dd = downsample_minmax(d_offset, t)
            self._focus_line_sources[i_idx].data = {'x': tt, 'y': dd}

        # --- Focus figure: reference lines ---
        threshold = getattr(config, 'AMPLITUDE_THRESHOLD', None)
        ref_idx = 0
        for i_idx in range(n_valid):
            offset = (n_valid - 1 - i_idx) * offset_per_channel
            if ref_idx < len(self._focus_ref_line_sources):
                self._focus_ref_line_sources[ref_idx].data = {'x': [0, display_duration_s], 'y': [offset, offset]}
                ref_idx += 1
            if threshold is not None:
                if ref_idx < len(self._focus_ref_line_sources):
                    self._focus_ref_line_sources[ref_idx].data = {'x': [0, display_duration_s], 'y': [offset + threshold, offset + threshold]}
                    ref_idx += 1
                if ref_idx < len(self._focus_ref_line_sources):
                    self._focus_ref_line_sources[ref_idx].data = {'x': [0, display_duration_s], 'y': [offset - threshold, offset - threshold]}
                    ref_idx += 1

        # --- Context shading ---
        y_min_main, y_max_main = -230, 230
        channel_range = 200
        y_min_focus = -channel_range
        y_max_focus = (n_valid - 1) * offset_per_channel + channel_range

        for ctx_src, y_lo, y_hi in [(self._main_context_source, y_min_main, y_max_main),
                                     (self._focus_context_source, y_min_focus, y_max_focus)]:
            left_list, right_list = [], []
            if pre_samples > 0 and center_start_time_s > 0:
                left_list.append(0.0)
                right_list.append(center_start_time_s)
            if post_samples > 0 and center_end_time_s < display_duration_s:
                left_list.append(center_end_time_s)
                right_list.append(display_duration_s)
            n_ctx = len(left_list)
            ctx_src.data = {
                'left': left_list, 'right': right_list,
                'top': [y_hi] * n_ctx, 'bottom': [y_lo] * n_ctx,
                'color': [CONTEXT_BACKGROUND_COLOR] * n_ctx, 'alpha': [0.25] * n_ctx,
            }

        # --- Region rectangles ---
        self._update_rect_data(current_regions, epoch_data, center_start_time_s)

        # --- Bandpass overlay ---
        self._update_focus_bandpass()

        t1 = time.time()
        debug_print(f"⏱️ _full_update: {(t1-t0)*1000:.1f}ms")

    def _update_rect_data(self, current_regions=None, epoch_data=None, time_offset_seconds=None):
        """Update rectangle ColumnDataSources for both main and focus figures."""
        if current_regions is None or epoch_data is None or time_offset_seconds is None:
            epoch_data, current_regions, _ = self._get_epoch_data()
            _, _, _, time_offset_seconds, _ = self._get_display_window_data()

        # Build rect data for main plot
        y_min_main, y_max_main = -230, 230
        main_data = self._build_rect_source_data(current_regions, (y_min_main, y_max_main), epoch_data, time_offset_seconds)
        self._main_rect_source.data = main_data

        # Build rect data for focus plot
        n_valid = len(self._focus_valid_channels)
        offset_per_channel = 500
        channel_range = 200
        y_min_focus = -channel_range
        y_max_focus = (n_valid - 1) * offset_per_channel + channel_range
        focus_data = self._build_rect_source_data(current_regions, (y_min_focus, y_max_focus), epoch_data, time_offset_seconds)
        self._focus_rect_source.data = focus_data

    def _build_rect_source_data(self, regions: dict, y_range: Tuple[float, float],
                                epoch_data: pd.DataFrame, time_offset_seconds: float) -> dict:
        """Build dict suitable for ColumnDataSource from region annotations."""
        result = self._empty_rect_data()
        if not regions:
            return result

        y_min, y_max = y_range

        # Initialize topoplot cache structure if needed
        if epoch_data is not None and MNE_AVAILABLE:
            if self._current_epoch_for_cache != self.epoch_index:
                self._topoplot_cache = {}
                self._current_epoch_for_cache = self.epoch_index
            if self.epoch_index not in self._topoplot_cache:
                self._topoplot_cache[self.epoch_index] = {}

        for region_id, region in regions.items():
            rel_start = region.get('relative_start', region['start_idx'])
            rel_stop = region.get('relative_stop', region['stop_idx'])

            status = self.annotation_manager.get_annotation_status(region_id)
            is_sel = (region_id == self.selected_region_id)

            if status == 'KC':
                fill = ANNOTATION_COLORS['KC']
                base_lc = ANNOTATION_COLORS['KC']
            else:
                fill = ANNOTATION_COLORS['unannotated']
                base_lc = ANNOTATION_COLORS['unannotated']

            if is_sel:
                lc = '#FFD700'
                lw = 4
                alpha = 0.25
            else:
                lc = base_lc
                lw = 2
                alpha = 0.15

            start_time = (rel_start / self.sampling_rate) + time_offset_seconds
            stop_time = (rel_stop / self.sampling_rate) + time_offset_seconds

            # Topoplot HTML from cache only
            topo_html = ''
            if epoch_data is not None and MNE_AVAILABLE:
                if (self.epoch_index in self._topoplot_cache and
                    region_id in self._topoplot_cache[self.epoch_index]):
                    topo_img = self._topoplot_cache[self.epoch_index][region_id]
                    topo_html = f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'

            duration_s = (rel_stop - rel_start) / self.sampling_rate
            duration_color = '#27ae60' if 0.5 <= duration_s <= 2.0 else '#e74c3c'
            duration_html = f"<span style='color:{duration_color};'>{duration_s:.3f}s</span>"

            max_diff_text = "n/a"
            if epoch_data is not None and rel_stop > rel_start:
                region_window = epoch_data.iloc[rel_start:rel_stop + 1]
                if not region_window.empty:
                    max_diff_uv = (region_window.max(axis=0) - region_window.min(axis=0)).max()
                    max_diff_text = f"{float(max_diff_uv):.1f} uV"

            result['left'].append(start_time)
            result['right'].append(stop_time)
            result['top'].append(y_max)
            result['bottom'].append(y_min)
            result['fill_color'].append(fill)
            result['line_color'].append(lc)
            result['line_width'].append(lw)
            result['alpha'].append(alpha)
            result['region_id'].append(region_id)
            result['status'].append(status)
            result['duration_html'].append(duration_html)
            result['max_diff_text'].append(max_diff_text)
            result['topo_html'].append(topo_html)

        return result

    def _fast_update_rect_colors(self):
        """
        Fast path: patch rectangle sources directly without rebuilding data.
        Uses source.patch() for instant browser updates.
        """
        sources = [self._main_rect_source, self._focus_rect_source]
        patched_any = False

        try:
            for source in sources:
                data = source.data
                if 'region_id' not in data or len(data['region_id']) == 0:
                    continue

                n = len(data['region_id'])
                fill_patches = []
                lc_patches = []
                lw_patches = []
                alpha_patches = []
                status_patches = []

                for i in range(n):
                    rid = int(data['region_id'][i])
                    status = self.annotation_manager.get_annotation_status(rid)
                    is_sel = (rid == self.selected_region_id)

                    if status == 'KC':
                        fill = ANNOTATION_COLORS['KC']
                        base_lc = ANNOTATION_COLORS['KC']
                    else:
                        fill = ANNOTATION_COLORS['unannotated']
                        base_lc = ANNOTATION_COLORS['unannotated']

                    if is_sel:
                        lc = '#FFD700'
                        lw = 4
                        alpha = 0.25
                    else:
                        lc = base_lc
                        lw = 2
                        alpha = 0.15

                    fill_patches.append((i, fill))
                    lc_patches.append((i, lc))
                    lw_patches.append((i, lw))
                    alpha_patches.append((i, alpha))
                    status_patches.append((i, status))

                source.patch({
                    'fill_color': fill_patches,
                    'line_color': lc_patches,
                    'line_width': lw_patches,
                    'alpha': alpha_patches,
                    'status': status_patches,
                })
                patched_any = True

            return patched_any
        except Exception as e:
            debug_print(f"Fast annotation update failed: {e}")
            return False

    def _update_focus_bandpass(self):
        """Update bandpass overlay data for focus plot."""
        if not self.show_bandpass_overlay:
            for src in self._focus_bandpass_sources:
                src.data = {'x': [], 'y': []}
            return

        display_df, _, _, _, _ = self._get_display_window_data()
        if display_df is None or len(display_df) == 0:
            return

        filtered_df = self._focus_bandpass_cache.get(self.epoch_index)
        if filtered_df is None:
            try:
                filtered_values = apply_bandpass_filter(
                    display_df, low_freq=0.5, high_freq=2.0,
                    sampling_rate=self.sampling_rate, order=4,
                ).values
                filtered_df = pd.DataFrame(filtered_values, columns=display_df.columns, index=display_df.index)
                self._focus_bandpass_cache[self.epoch_index] = filtered_df
            except Exception as e:
                print(f"[WARNING] Failed to compute 0.5-2 Hz overlay: {e}")
                return

        t = self._create_time_axis(len(display_df))
        n_valid = len(self._focus_valid_channels)
        offset_per_channel = 500

        for i_idx, ch_idx in enumerate(self._focus_valid_channels):
            if i_idx >= len(self._focus_bandpass_sources):
                break
            col = display_df.columns[ch_idx] if ch_idx < len(display_df.columns) else display_df.columns[0]
            offset = (n_valid - 1 - i_idx) * offset_per_channel
            overlay_vals = filtered_df[col].values + offset
            tt, dd = downsample_minmax(overlay_vals, t)
            self._focus_bandpass_sources[i_idx].data = {'x': tt, 'y': dd}

    # ------------------------------------------------------------------ #
    #  Bokeh event handlers (replace HoloViews streams)
    # ------------------------------------------------------------------ #
    def _on_bokeh_box_select(self, event):
        """Handle Bokeh SelectionGeometry event (box select)."""
        geom = event.geometry
        if geom is None:
            return

        x0 = geom.get('x0', None)
        x1 = geom.get('x1', None)
        if x0 is None or x1 is None:
            return

        start_time = min(x0, x1)
        end_time = max(x0, x1)

        # Time-guard
        now = time.time()
        if hasattr(self, '_last_box_select_time') and (now - self._last_box_select_time) < 0.25:
            return
        self._last_box_select_time = now

        epoch_data, _, epoch_start_idx = self._get_epoch_data()
        if epoch_data is None:
            return

        _, pre_samples, _, center_start_time_s, center_end_time_s = self._get_display_window_data()
        start_idx_display = int(start_time * self.sampling_rate)
        stop_idx_display = int(end_time * self.sampling_rate)

        start_idx = start_idx_display - pre_samples
        stop_idx = stop_idx_display - pre_samples

        start_idx = max(0, min(start_idx, len(epoch_data) - 1))
        stop_idx = max(0, min(stop_idx, len(epoch_data) - 1))

        if start_idx >= stop_idx:
            return

        # Dedup
        region_key = (epoch_start_idx, start_idx, stop_idx)
        if hasattr(self, '_last_created_region_key') and self._last_created_region_key == region_key:
            return
        self._last_created_region_key = region_key

        region_id = self.annotation_manager.add_region(start_idx, stop_idx, epoch_start_idx)
        self.selected_region_id = region_id

        self._cached_regions = None

        status = self.annotation_manager.get_annotation_status(region_id)
        self._skip_label_change_handler = True
        self.current_label = status
        self._skip_label_change_handler = False

        # Update rect data (new region added, need full rect rebuild)
        self._update_rect_data()
        self.status_trigger += 1

    def _on_bokeh_tap(self, event):
        """Handle Bokeh Tap event (click to select region)."""
        if event.x is None:
            return

        now = time.time()
        click_time = event.x
        click_sample_display = int(click_time * self.sampling_rate)
        _, pre_samples, _, _, _ = self._get_display_window_data()
        click_sample = click_sample_display - pre_samples

        # Debounce
        if (self._last_click_region is not None and
            abs(now - self._last_click_time) < self._click_debounce_time):
            return

        if self._processing_click:
            return
        self._processing_click = True

        try:
            epoch_data, current_regions, _ = self._get_epoch_data()
            if epoch_data is None or len(epoch_data) == 0:
                return
            if not current_regions:
                return

            epoch_len_samples = len(epoch_data)
            if click_sample < 0 or click_sample > epoch_len_samples:
                return

            for region_id, region in current_regions.items():
                rel_start = region.get('relative_start', region['start_idx'])
                rel_stop = region.get('relative_stop', region['stop_idx'])

                if rel_start <= click_sample <= rel_stop:
                    if self.selected_region_id == region_id:
                        return

                    self.selected_region_id = region_id
                    status = self.annotation_manager.get_annotation_status(region_id)
                    self._skip_label_change_handler = True
                    self.current_label = status
                    self._skip_label_change_handler = False

                    self._last_click_time = now
                    self._last_click_region = region_id

                    self._cached_regions = None
                    if not self._fast_update_rect_colors():
                        self._update_rect_data()
                    self.status_trigger += 1
                    return
        finally:
            self._processing_click = False

    # ------------------------------------------------------------------ #
    #  Keyboard & label handling
    # ------------------------------------------------------------------ #
    def _handle_kb_event(self, event):
        """Handle keyboard input from the ReactiveHTML component."""
        key = event.new
        if not key or self.selected_region_id == -1:
            return

        print(f"⌨️ Key Press: {key}")

        if key in ['k', 'c']:
            self.annotation_manager.set_annotation(self.selected_region_id, True)
            self._cached_regions = None
            self._skip_label_change_handler = True
            self.current_label = 'KC'
            self._skip_label_change_handler = False
            if not self._fast_update_rect_colors():
                self._update_rect_data()
            self.status_trigger += 1
        elif key in ['u', 'y']:
            self.annotation_manager.set_annotation(self.selected_region_id, False)
            self._cached_regions = None
            self._skip_label_change_handler = True
            self.current_label = 'unannotated'
            self._skip_label_change_handler = False
            if not self._fast_update_rect_colors():
                self._update_rect_data()
            self.status_trigger += 1
        elif key in ['d', 'delete']:
            if self.annotation_manager.delete_region(self.selected_region_id):
                print(f"🗑️ Deleted region {self.selected_region_id}")
                self.selected_region_id = -1
                self._skip_label_change_handler = True
                self.current_label = 'unannotated'
                self._skip_label_change_handler = False
                self._cached_regions = None
                self._update_rect_data()
                self.status_trigger += 1

        self.kb_listener.key = ""

    @param.depends('current_label', watch=True)
    def _on_label_change(self):
        """Update annotation when label changes (via radio button UI)."""
        if self._skip_label_change_handler:
            return
        if self.selected_region_id == -1:
            return

        if self.current_label == 'KC':
            self.annotation_manager.set_annotation(self.selected_region_id, True)
        else:
            self.annotation_manager.set_annotation(self.selected_region_id, False)

        self._cached_regions = None
        if not self._fast_update_rect_colors():
            self._update_rect_data()
        self.status_trigger += 1

    # ------------------------------------------------------------------ #
    #  Status bar & selection info
    # ------------------------------------------------------------------ #
    @param.depends('epoch_index', 'update_trigger', 'status_trigger')
    def status_bar(self):
        """Create reactive status bar with epoch info and annotation counts."""
        epoch_info = self.epoch_manager.get_current_epoch_info()
        sleep_stage = epoch_info.get('sleep_stage', '?')
        current_idx = epoch_info.get('current_idx', 0)
        total_filtered = epoch_info.get('total_filtered', 0)

        _, current_regions, _ = self._get_epoch_data()
        regions_in_epoch = len(current_regions)

        counts = self.annotation_manager.get_annotation_count()
        total_regions = len(self.annotation_manager.get_all_regions())

        status_text = (
            f"**Epoch**: {current_idx + 1}/{total_filtered} | "
            f"**Sleep Stage**: {sleep_stage} | "
            f"**Regions in Epoch**: {regions_in_epoch} | "
            f"**Total Regions**: {total_regions} | "
            f"**KC**: {counts['KC']} | **Unannotated**: {counts['unannotated']}"
        )

        return pn.pane.Markdown(
            status_text,
            styles={'font-size': '12px', 'padding': '5px 10px', 'background': '#ecf0f1', 'border-radius': '3px', 'margin': '0px'}
        )

    def _get_selection_info(self, region_id: int) -> Optional[Dict[str, float]]:
        """Return selection metrics for the currently selected region."""
        if region_id == -1:
            return None

        epoch_data, current_regions, _ = self._get_epoch_data()
        if epoch_data is None or region_id not in current_regions:
            return None

        region = current_regions[region_id]
        rel_start = int(region.get('relative_start', 0))
        rel_stop = int(region.get('relative_stop', 0))
        if rel_stop <= rel_start:
            return None

        duration_s = (rel_stop - rel_start) / self.sampling_rate
        window = epoch_data.iloc[rel_start:rel_stop + 1]
        if window.empty:
            return None

        peak_to_peak = (window.max(axis=0) - window.min(axis=0)).max()
        status = self.annotation_manager.get_annotation_status(region_id)
        return {
            'duration_s': float(duration_s),
            'max_diff_uv': float(peak_to_peak),
            'status': status,
        }

    def _selection_info_pane(self, rid: int):
        """Render selected region information with duration threshold highlighting."""
        if rid == -1:
            return pn.pane.HTML(
                "<div style='font-size:12px; margin:0px;'><b>No Region Selected</b></div>"
            )

        info = self._get_selection_info(rid)
        if info is None:
            return pn.pane.HTML(
                f"<div style='font-size:12px; margin:0px;'><b>Selected Region:</b> {rid}</div>"
            )

        duration_color = '#27ae60' if 0.5 <= info['duration_s'] <= 2.0 else '#e74c3c'
        html = (
            "<div style='font-size:12px; margin:0px; line-height:1.2;'>"
            f"<b>Region:</b> {rid} | "
            f"<b>Status:</b> {info['status']} | "
            f"<b>Duration:</b> <span style='color:{duration_color};'>{info['duration_s']:.3f}s</span> | "
            f"<b>Max diff:</b> {info['max_diff_uv']:.1f} uV"
            "</div>"
        )
        return pn.pane.HTML(html)

    # ------------------------------------------------------------------ #
    #  Topoplot generation
    # ------------------------------------------------------------------ #
    def _compute_epoch_spectral_power(self, epoch_data: pd.DataFrame, epoch_index: Optional[int] = None):
        """Compute spectral power (0.5-2 Hz) for the entire epoch using sliding window approach."""
        if epoch_data is None or len(epoch_data) == 0:
            return
        if epoch_index is None:
            epoch_index = self.epoch_index

        try:
            low_freq = 0.5
            high_freq = 2.0
            window_length_sec = 2.0
            overlap_sec = 0.5

            sr = self.sampling_rate
            window_length_samples = int(window_length_sec * sr)
            overlap_samples = int(overlap_sec * sr)
            hop_samples = window_length_samples - overlap_samples

            n_samples, n_channels = epoch_data.shape
            power_data = np.zeros((n_samples, n_channels))

            for ch_idx in range(n_channels):
                channel_data = epoch_data.iloc[:, ch_idx].values

                for start_idx in range(0, n_samples - window_length_samples + 1, hop_samples):
                    end_idx = start_idx + window_length_samples
                    window_data = channel_data[start_idx:end_idx]

                    freqs, psd = signal.welch(
                        window_data, fs=sr,
                        nperseg=min(window_length_samples, len(window_data)),
                        noverlap=overlap_samples // 2 if overlap_samples > 0 else None
                    )

                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):
                        if hasattr(np, 'trapezoid'):
                            band_power = np.trapezoid(psd[freq_mask], freqs[freq_mask])
                        else:
                            band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                    else:
                        band_power = 0.0

                    power_data[start_idx:end_idx, ch_idx] = band_power

                if n_samples > 0:
                    power_col = power_data[:, ch_idx]
                    nonzero_mask = power_col > 0
                    if np.any(nonzero_mask):
                        first_nonzero_idx = np.where(nonzero_mask)[0][0]
                        if first_nonzero_idx > 0:
                            power_data[:first_nonzero_idx, ch_idx] = power_data[first_nonzero_idx, ch_idx]
                        last_nonzero_idx = np.where(nonzero_mask)[0][-1]
                        if last_nonzero_idx < n_samples - 1:
                            power_data[last_nonzero_idx+1:, ch_idx] = power_data[last_nonzero_idx, ch_idx]

            power_df = pd.DataFrame(power_data, columns=epoch_data.columns, index=epoch_data.index)
            with self._spectral_power_lock:
                self._spectral_power_cache[epoch_index] = power_df

        except Exception as e:
            print(f"[ERROR SPECTRAL] Error computing spectral power for epoch {epoch_index}: {e}")
            import traceback
            traceback.print_exc()

    def _create_topoplot_image(self, region_id: int, epoch_data: pd.DataFrame,
                               start_idx: int, stop_idx: int) -> Optional[str]:
        """Create base64-encoded topoplot image for a selected region using spectral power."""
        if not MNE_AVAILABLE:
            return None

        try:
            if self.epoch_index not in self._spectral_power_cache:
                self._schedule_spectral_power_compute(self.epoch_index, epoch_data)
                return None

            power_df = self._spectral_power_cache.get(self.epoch_index)
            if power_df is None or len(power_df) == 0:
                return None

            if start_idx >= len(power_df) or stop_idx >= len(power_df):
                return None

            region_power = power_df.iloc[start_idx:stop_idx+1]
            mean_power_linear = region_power.mean(axis=0).values
            mean_power_db = 10.0 * np.log10(np.maximum(mean_power_linear, 1e-12))

            if not self.chanlocs.empty:
                if 'X' in self.chanlocs.columns and 'Y' in self.chanlocs.columns:
                    pos = np.array([-self.chanlocs['Y'].values, self.chanlocs['X'].values]).T
                elif 'x' in self.chanlocs.columns and 'y' in self.chanlocs.columns:
                    pos = np.array([-self.chanlocs['y'].values, self.chanlocs['x'].values]).T
                else:
                    n_chans = len(mean_power_db)
                    angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
                    pos = np.array([np.cos(angles), np.sin(angles)]).T
            else:
                n_chans = len(mean_power_db)
                angles = np.linspace(0, 2*np.pi, n_chans, endpoint=False)
                pos = np.array([np.cos(angles), np.sin(angles)]).T

            pos_original = pos.copy()
            max_radius = np.max(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))
            if max_radius > 0:
                pos = pos / max_radius * 0.9

            kept_indices = np.arange(len(mean_power_db))
            if self.exclude_channels:
                n_chans = len(mean_power_db)
                mask = np.ones(n_chans, dtype=bool)
                for idx in self.exclude_channels:
                    if 0 <= idx < n_chans:
                        mask[idx] = False
                mean_power_db = mean_power_db[mask]
                pos = pos[mask]
                pos_original = pos_original[mask]
                kept_indices = kept_indices[mask]

            finite_vals = mean_power_db[np.isfinite(mean_power_db)]
            if finite_vals.size >= 2:
                vmin, vmax = np.percentile(finite_vals, [5, 95])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                    vmin, vmax = np.percentile(finite_vals, [2, 98])
                if np.isclose(vmin, vmax):
                    pad = max(1e-3, 0.05 * (abs(vmin) + 1.0))
                    vmin -= pad
                    vmax += pad
                vlim = (float(vmin), float(vmax))
            else:
                vlim = (None, None)

            fig, ax = plt.subplots(figsize=(3, 3))
            mne.viz.plot_topomap(
                mean_power_db, pos, axes=ax, show=False,
                cmap='Reds', vlim=vlim
            )

            if self.focus_channels:
                for focus_ch in self.focus_channels:
                    match_idx = np.where(kept_indices == focus_ch)[0]
                    if len(match_idx) > 0:
                        idx = match_idx[0]
                        ax.plot(pos[idx, 0], pos[idx, 1], 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
                        ax.plot(pos[idx, 0], pos[idx, 1], 'k+', markersize=6, markeredgewidth=1.5)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            return f'data:image/png;base64,{img_str}'
        except Exception as e:
            print(f"[ERROR TOPO] Error creating topoplot for region {region_id}: {e}")
            return None

    def _get_topoplot_on_demand(self, region_id: int, epoch_data: pd.DataFrame,
                                rel_start: int, rel_stop: int) -> str:
        """Get topoplot image on-demand, checking cache first, generating if needed."""
        if not MNE_AVAILABLE or epoch_data is None:
            return ''

        if (self.epoch_index in self._topoplot_cache and
            region_id in self._topoplot_cache[self.epoch_index]):
            topo_img = self._topoplot_cache[self.epoch_index][region_id]
            return f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'

        topo_img = self._create_topoplot_image(region_id, epoch_data, rel_start, rel_stop)
        if topo_img:
            if self.epoch_index not in self._topoplot_cache:
                self._topoplot_cache[self.epoch_index] = {}
            self._topoplot_cache[self.epoch_index][region_id] = topo_img
            return f'<img src="{topo_img}" style="width: 200px; height: 200px; margin-top: 5px; display: block;">'

        return ''

    @param.depends('topoplot_trigger', 'selected_region_id', watch=True)
    def _refresh_topoplot_panel(self):
        """Update the fixed topoplot HTML pane in-place (avoids Bokeh document ownership errors)."""
        if self.selected_region_id == -1:
            self._topoplot_html_pane.object = ""
            return

        epoch_data, current_regions, _ = self._get_epoch_data()

        if not current_regions or self.selected_region_id not in current_regions:
            self._topoplot_html_pane.object = ""
            return

        region = current_regions[self.selected_region_id]
        rel_start = region.get('relative_start', region['start_idx'])
        rel_stop = region.get('relative_stop', region['stop_idx'])

        html = self._get_topoplot_on_demand(self.selected_region_id, epoch_data, rel_start, rel_stop)
        self._topoplot_html_pane.object = html if html else "<span style='font-size:12px;color:gray;'>Topoplot unavailable</span>"

        # Also refresh hover tooltip HTML in rect sources now that topo is cached
        if html:
            self._update_rect_data()

    # ------------------------------------------------------------------ #
    #  Dashboard layout (Bokeh-native: pn.pane.Bokeh instead of DynamicMap)
    # ------------------------------------------------------------------ #
    def view(self) -> pn.Column:
        """Create the dashboard view."""
        # Capture Bokeh document for thread-safe callbacks from background threads
        try:
            self._bokeh_doc = pn.state.curdoc()
        except Exception:
            pass

        btn_prev = pn.widgets.Button(name='◀ Prev', width=80)
        btn_next = pn.widgets.Button(name='Next ▶', width=80)
        btn_prev.on_click(lambda e: setattr(self, 'epoch_index', max(0, self.epoch_index - 1)))
        btn_next.on_click(lambda e: setattr(self, 'epoch_index', self.epoch_index + 1))

        max_epoch = max(0, self.epoch_manager.get_epoch_count() - 1)
        epoch_jump_input = pn.widgets.IntInput(
            name='Epoch', value=self.epoch_index, start=0, end=max_epoch, width=80, step=1
        )
        self.param.watch(lambda e: setattr(epoch_jump_input, 'value', e.new), 'epoch_index')
        def jump_to_epoch(event):
            try:
                new_idx = int(event.new)
                if 0 <= new_idx <= max_epoch:
                    self.epoch_index = new_idx
                else:
                    epoch_jump_input.value = self.epoch_index
            except (ValueError, TypeError):
                epoch_jump_input.value = self.epoch_index
        epoch_jump_input.param.watch(jump_to_epoch, 'value')

        radio_group = pn.widgets.RadioButtonGroup(
            name='Annotation', options=['KC', 'unannotated'],
            button_type='default', value=self.current_label
        )
        self.param.watch(lambda e: setattr(radio_group, 'value', e.new), 'current_label')
        radio_group.param.watch(lambda e: setattr(self, 'current_label', e.new), 'value')

        overlay_toggle = pn.widgets.Toggle(
            name='0.5-2 Hz', value=self.show_bandpass_overlay,
            button_type='primary', width=90,
        )
        self.param.watch(lambda e: setattr(overlay_toggle, 'value', e.new), 'show_bandpass_overlay')
        overlay_toggle.param.watch(lambda e: setattr(self, 'show_bandpass_overlay', bool(e.new)), 'value')

        info = pn.bind(
            lambda rid, trig, strig: self._selection_info_pane(rid),
            rid=self.param.selected_region_id,
            trig=self.param.update_trigger,
            strig=self.param.status_trigger,
        )

        style = pn.pane.HTML("""<style>
        .bk-btn-group .bk-btn:nth-child(1) { background-color: #27ae60 !important; color: white !important; }
        .bk-btn-group .bk-btn:nth-child(2) { background-color: #95a5a6 !important; color: white !important; }
        </style>""")

        controls_col = pn.Column(
            pn.Row(
                btn_prev, btn_next,
                pn.Spacer(width=10),
                epoch_jump_input,
                pn.Spacer(width=10),
                info,
                pn.Spacer(width=10),
                radio_group,
                pn.Spacer(width=10),
                overlay_toggle,
                align='center',
                sizing_mode='stretch_width',
            ),
            pn.pane.Markdown(
                "**Keys:** `K`/`C` = KC | `U`/`Y` = unannotated | `D` = delete | **Drag** to select",
                styles={'font-size': '11px', 'margin': '2px 0 0 0'}
            ),
            sizing_mode='stretch_width',
            margin=(5, 0),
        )

        # Topoplot in top-right, beside controls
        top_row = pn.Row(
            controls_col,
            pn.Spacer(width=10),
            self._topoplot_html_pane,
            align='start',
            sizing_mode='stretch_width',
            margin=(0, 0),
        )

        # Use pn.pane.Bokeh for direct Bokeh figures — NO DynamicMap rebuild overhead
        focus_pane = pn.pane.Bokeh(self._focus_fig, sizing_mode='stretch_width')
        main_pane = pn.pane.Bokeh(self._main_fig, sizing_mode='stretch_width')

        return pn.Column(
            self.status_bar,
            top_row,
            focus_pane,
            main_pane,
            self.kb_listener,
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
