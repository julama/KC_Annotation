# PyInstaller hook for holoviews
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all holoviews submodules
hiddenimports = collect_submodules('holoviews')

# Collect data files
datas = collect_data_files('holoviews')
