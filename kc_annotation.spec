# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for KC Annotation Tool

Build commands:
    pyinstaller kc_annotation.spec              # One-folder mode (recommended)
    pyinstaller kc_annotation.spec --onefile    # One-file mode (slower startup)

Or use the build script:
    python build_exe.py
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get paths to package data
block_cipher = None

# Collect all submodules for packages that have dynamic imports
hiddenimports = [
    # Panel and its dependencies
    'panel',
    'panel.template',
    'panel.template.fast',
    'panel.template.base',
    'panel.widgets',
    'panel.pane',
    'panel.layout',
    'panel.io',
    'panel.io.server',
    'panel.io.state',
    'panel.reactive',
    'panel.viewable',
    'panel.param',
    
    # Bokeh
    'bokeh',
    'bokeh.core',
    'bokeh.core.properties',
    'bokeh.core.property',
    'bokeh.core.validation',
    'bokeh.models',
    'bokeh.models.widgets',
    'bokeh.models.plots',
    'bokeh.models.tools',
    'bokeh.models.layouts',
    'bokeh.models.formatters',
    'bokeh.models.tickers',
    'bokeh.models.ranges',
    'bokeh.models.sources',
    'bokeh.models.glyphs',
    'bokeh.models.annotations',
    'bokeh.models.callbacks',
    'bokeh.plotting',
    'bokeh.embed',
    'bokeh.io',
    'bokeh.resources',
    'bokeh.server',
    'bokeh.server.server',
    'bokeh.application',
    'bokeh.document',
    'bokeh.util',
    'bokeh.util.serialization',
    'bokeh.protocol',
    'bokeh.themes',
    
    # HoloViews
    'holoviews',
    'holoviews.core',
    'holoviews.core.data',
    'holoviews.core.options',
    'holoviews.element',
    'holoviews.plotting',
    'holoviews.plotting.bokeh',
    'holoviews.plotting.bokeh.element',
    'holoviews.plotting.bokeh.plot',
    'holoviews.plotting.bokeh.callbacks',
    'holoviews.operation',
    'holoviews.operation.datashader',
    'holoviews.streams',
    'holoviews.util',
    
    # Datashader
    'datashader',
    'datashader.core',
    'datashader.transfer_functions',
    'datashader.colors',
    'datashader.reductions',
    'datashader.glyphs',
    'datashader.bundling',
    
    # Param
    'param',
    'param.parameterized',
    
    # Colorcet
    'colorcet',
    
    # Scientific packages
    'scipy',
    'scipy.io',
    'scipy.io.matlab',
    'scipy.io.matlab._mio',
    'scipy.io.matlab._mio5',
    'scipy.io.matlab._mio5_params',
    'scipy.signal',
    'scipy.signal._signaltools',
    'scipy.fft',
    'scipy.fft._pocketfft',
    'scipy.linalg',
    'scipy.sparse',
    'scipy.ndimage',
    'scipy.interpolate',
    
    # NumPy
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'numpy.fft',
    'numpy.linalg',
    'numpy.random',
    
    # Pandas
    'pandas',
    'pandas.core',
    'pandas._libs',
    
    # MNE
    'mne',
    'mne.io',
    'mne.filter',
    'mne.channels',
    'mne.viz',
    
    # Tornado (used by Bokeh server)
    'tornado',
    'tornado.web',
    'tornado.websocket',
    'tornado.ioloop',
    'tornado.httpserver',
    'tornado.netutil',
    'tornado.platform',
    'tornado.platform.asyncio',
    
    # Other dependencies
    'pkg_resources',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
    'PIL',
    'PIL.Image',
    'markdown',
    'yaml',
    'jinja2',
    'markupsafe',
    'certifi',
    'charset_normalizer',
    'idna',
    'urllib3',
    'xyzservices',
    
    # Tkinter for file dialog
    'tkinter',
    'tkinter.filedialog',
    
    # Matplotlib (used for topoplot rendering)
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.figure',
    'matplotlib.backends',
    'matplotlib.backends.backend_agg',
    'matplotlib.cm',
    'matplotlib.colors',
]

# Collect data files from packages
datas = []

# Panel data files (templates, static files)
try:
    datas += collect_data_files('panel')
except Exception as e:
    print(f"Warning: Could not collect panel data files: {e}")

# Bokeh data files (JS, CSS)
try:
    datas += collect_data_files('bokeh')
except Exception as e:
    print(f"Warning: Could not collect bokeh data files: {e}")

# HoloViews data files
try:
    datas += collect_data_files('holoviews')
except Exception as e:
    print(f"Warning: Could not collect holoviews data files: {e}")

# Datashader data files
try:
    datas += collect_data_files('datashader')
except Exception as e:
    print(f"Warning: Could not collect datashader data files: {e}")

# Colorcet data files
try:
    datas += collect_data_files('colorcet')
except Exception as e:
    print(f"Warning: Could not collect colorcet data files: {e}")

# MNE data files
try:
    datas += collect_data_files('mne')
except Exception as e:
    print(f"Warning: Could not collect mne data files: {e}")

# xyzservices data files (for map tiles if used)
try:
    datas += collect_data_files('xyzservices')
except Exception as e:
    print(f"Warning: Could not collect xyzservices data files: {e}")

# Matplotlib data files (fonts, styles, etc.)
try:
    datas += collect_data_files('matplotlib')
except Exception as e:
    print(f"Warning: Could not collect matplotlib data files: {e}")

# Add local files
datas += [
    ('config.py', '.'),
    ('src', 'src'),
]

a = Analysis(
    ['main_panel.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks/runtime_hook.py'],
    excludes=[
        # Exclude packages we don't need to reduce size
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='kc_annotation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed mode (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='kc_annotation',
)
