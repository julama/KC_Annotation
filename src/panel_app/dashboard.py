import panel as pn
import holoviews as hv
from holoviews import opts, streams
import param
import numpy as np
import pandas as pd
from typing import Optional, List

# Initialize extensions
pn.extension('tabulator', sizing_mode='stretch_width')
hv.extension('bokeh')

# --- CONFIGURATION ---
ANNOTATION_COLORS = {
    'unannotated': '#95a5a6',  # Gray
    'KC': '#27ae60',           # Green  
    'non-KC': '#c0392b',       # Red
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
                // Sync specific keys to Python (KC: k/c, non-KC: n/x, unannotated: u/y)
                if (['k', 'c', 'n', 'x', 'u', 'y'].includes(k)) { 
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
    selected_event_id = param.Integer(default=-1)
    current_label = param.Selector(objects=['KC', 'non-KC', 'unannotated'], default='unannotated')
    update_trigger = param.Integer(default=0)
    
    def __init__(self, epoch_manager, sw_events: pd.DataFrame,
                 annotation_manager, focus_channels: List[int] = None, **params):
        
        # --- FIX: Set attributes BEFORE super().__init__ ---
        # This prevents "AttributeError" if watchers fire during init
        self.epoch_manager = epoch_manager
        self.sw_events = sw_events
        self.annotation_manager = annotation_manager
        self.focus_channels = focus_channels or [34, 55, 70]
        self.sampling_rate = epoch_manager.sampling_rate
        
        # Now call super, which might trigger watchers immediately
        super().__init__(**params)
        
        self.param.epoch_index.bounds = (0, max(0, epoch_manager.get_epoch_count() - 1))
        
        # Initialize Interaction Streams
        self.tap_stream = streams.Tap(transient=True)
        self.tap_stream.add_subscriber(self._on_plot_click)

        # Initialize Keyboard Listener
        self.kb_listener = KeyboardListener()
        self.kb_listener.param.watch(self._handle_kb_event, 'key')

    def _get_epoch_data(self):
        self.epoch_manager.current_epoch_idx = self.epoch_index
        epoch_data = self.epoch_manager.get_current_epoch()
        # Ensure this import path matches your project structure
        from src.sw_loader import get_sw_events_for_current_epoch
        return epoch_data, get_sw_events_for_current_epoch(self.sw_events, self.epoch_manager)
    
    def _create_time_axis(self, n): return np.arange(n) / self.sampling_rate

    # --- INTERACTION LOGIC ---
    def _on_plot_click(self, x, y):
        """Handle clicks on the plot (select event)."""
        if x is None: return
        _, current_sw = self._get_epoch_data()
        if current_sw is None or current_sw.empty: return

        starts = current_sw['relative_start'] / self.sampling_rate
        stops = current_sw['relative_stop'] / self.sampling_rate
        mask = (starts <= x) & (stops >= x)
        
        if mask.any():
            eid = current_sw[mask].iloc[0]['event_id']
            print(f"🎯 Clicked: {eid}")
            self.selected_event_id = int(eid)
            
            # Sync label
            status = self.annotation_manager.get_annotation_status(eid)
            self.current_label = status if status in ['KC', 'non-KC'] else 'unannotated'
            
            self.update_trigger += 1

    def _handle_kb_event(self, event):
        """Handle keyboard input from the ReactiveHTML component."""
        key = event.new
        if not key or self.selected_event_id == -1: return
        
        print(f"⌨️ Key Press: {key}")
        
        # Map keys to labels (JavaScript already converts to lowercase)
        if key in ['k', 'c']:
            self.current_label = 'KC'
        elif key in ['n', 'x']:
            self.current_label = 'non-KC'
        elif key in ['u', 'y']:
            self.current_label = 'unannotated'
        
        # Reset listener so repeated keys work
        self.kb_listener.key = ""

    @param.depends('current_label', watch=True)
    def _on_label_change(self):
        """Update annotation when label changes (via Key or UI)."""
        if self.selected_event_id == -1: return
        
        if self.current_label == 'KC':
            self.annotation_manager.set_annotation(self.selected_event_id, True)
        elif self.current_label == 'non-KC':
            self.annotation_manager.set_annotation(self.selected_event_id, False)
        else:
            if self.selected_event_id in self.annotation_manager.annotations:
                del self.annotation_manager.annotations[self.selected_event_id]
                self.annotation_manager._save_annotations()
        
        self.update_trigger += 1

    # --- PLOTTING ---
    def _create_sw_rects(self, sw_events, y_range):
        if sw_events.empty: return hv.Rectangles([], kdims=['x0', 'y0', 'x1', 'y1'])
        y_min, y_max = y_range
        rect_data = []
        for _, row in sw_events.iterrows():
            eid = row['event_id']
            status = self.annotation_manager.get_annotation_status(eid)
            is_sel = (eid == self.selected_event_id)
            
            fill = ANNOTATION_COLORS.get(status, ANNOTATION_COLORS['unannotated'])
            lc = '#f1c40f' if is_sel else fill
            lw = 4 if is_sel else 1
            alpha = 0.6 if is_sel else 0.3
            
            rect_data.append({'x0': row['relative_start']/self.sampling_rate, 'y0': y_min,
                              'x1': row['relative_stop']/self.sampling_rate, 'y1': y_max,
                              'event_id': eid, 'status': status, 'fill_color': fill, 
                              'line_color': lc, 'line_width': lw, 'alpha': alpha})
        
        return hv.Rectangles(pd.DataFrame(rect_data), kdims=['x0','y0','x1','y1'], 
                             vdims=['event_id','status','fill_color','line_color','line_width','alpha']).opts(
            color='fill_color', line_color='line_color', line_width='line_width', alpha='alpha', tools=['hover', 'tap'])

    @param.depends('epoch_index', 'update_trigger')
    def create_main_plot(self, **kwargs): # kwargs handles x,y from Tap stream
        epoch_data, current_sw = self._get_epoch_data()
        if epoch_data is None: return hv.Spacer()
        
        t = self._create_time_axis(len(epoch_data))
        curves = []
        y_off = 0
        for col in epoch_data.columns[:20]:
            d = epoch_data[col].values
            d = (d - np.mean(d)) / (np.std(d) or 1) * 30 + y_off
            tt, dd = downsample_minmax(d, t)
            curves.append(hv.Curve((tt, dd)).opts(color='black', line_width=1))
            y_off += 100
            
        combined = hv.Overlay(curves) * self._create_sw_rects(current_sw, (-50, y_off+50))
        return combined.opts(opts.Overlay(width=1200, height=500, shared_axes=True, show_legend=False, 
                         tools=['tap', 'hover', 'xwheel_zoom', 'xpan'], active_tools=['tap', 'xwheel_zoom']))

    @param.depends('epoch_index', 'update_trigger')
    def create_focus_plot(self, channel_idx):
        epoch_data, current_sw = self._get_epoch_data()
        if epoch_data is None: return hv.Spacer()
        d = epoch_data.iloc[:, channel_idx].values
        t = self._create_time_axis(len(d))
        tt, dd = downsample_minmax(d, t)
        curve = hv.Curve((tt, dd)).opts(color='#34495e', line_width=1.5)
        rects = self._create_sw_rects(current_sw, (np.min(dd)*1.1, np.max(dd)*1.1))
        return (curve * rects).opts(opts.Overlay(width=1200, height=150, shared_axes=True, xaxis=None, ylabel=f"Ch {channel_idx}"))

    def view(self) -> pn.Column:
        btn_prev = pn.widgets.Button(name='◀ Prev', width=80)
        btn_next = pn.widgets.Button(name='Next ▶', width=80)
        btn_prev.on_click(lambda e: setattr(self, 'epoch_index', max(0, self.epoch_index - 1)))
        btn_next.on_click(lambda e: setattr(self, 'epoch_index', self.epoch_index + 1))
        
        # Radio Buttons
        radio_group = pn.widgets.RadioButtonGroup(
            name='Annotation', options=['KC', 'non-KC', 'unannotated'], 
            button_type='default', value=self.current_label
        )
        self.param.watch(lambda e: setattr(radio_group, 'value', e.new), 'current_label')
        radio_group.param.watch(lambda e: setattr(self, 'current_label', e.new), 'value')

        info = pn.bind(lambda eid: pn.pane.Markdown(f"### 🎯 Selected Event: {eid}" if eid != -1 else "### No Event Selected"), eid=self.param.selected_event_id)

        style = pn.pane.HTML("""<style>
        .bk-btn-group .bk-btn:nth-child(1) { background-color: #27ae60 !important; color: white !important; }
        .bk-btn-group .bk-btn:nth-child(2) { background-color: #c0392b !important; color: white !important; }
        .bk-btn-group .bk-btn:nth-child(3) { background-color: #95a5a6 !important; color: white !important; }
        </style>""")

        main_dmap = hv.DynamicMap(self.create_main_plot, streams=[self.tap_stream])
        focus_col = pn.Column(*[hv.DynamicMap(lambda idx=ch: self.create_focus_plot(idx)) for ch in self.focus_channels])
        
        return pn.Column(
            pn.Row(btn_prev, btn_next, pn.Spacer(width=30), pn.Column(info, radio_group, pn.pane.Markdown("**Keys:** `K`/`C` = KC, `N`/`X` = non-KC, `U`/`Y` = unannotated")), align='center'),
            pn.pane.HoloViews(main_dmap),
            focus_col,
            self.kb_listener,  # <--- INVISIBLE LISTENER COMPONENT
            style,
            sizing_mode='stretch_width'
        )

def create_dashboard(epoch_manager, sw_events, annotation_manager, focus_channels=None):
    db = EEGDashboard(epoch_manager, sw_events, annotation_manager, focus_channels)
    return db.view()