"""
Runtime hook for fixing package version issues in frozen apps.

This hook patches package version detection that fails when running as a PyInstaller executable.
"""
import sys

# Patch packaging.version to handle 'None' versions gracefully
def patch_packaging_version():
    """Patch packaging.version to handle invalid version strings."""
    try:
        import packaging.version
        original_init = packaging.version.Version.__init__
        
        def patched_init(self, version):
            if version is None or version == 'None' or version == '':
                version = '0.0.0'
            return original_init(self, version)
        
        packaging.version.Version.__init__ = patched_init
    except Exception:
        pass

# Patch importlib.metadata to provide package versions
def patch_importlib_metadata():
    """Provide fallback package versions for frozen apps."""
    try:
        import importlib.metadata
        
        # Known package versions - add versions from your requirements.txt
        KNOWN_VERSIONS = {
            'holoviews': '1.18.0',
            'panel': '1.3.0',
            'bokeh': '3.3.0',
            'datashader': '0.16.0',
            'param': '2.0.0',
            'colorcet': '3.0.0',
            'numpy': '1.23.0',
            'pandas': '1.6.0',
            'scipy': '1.9.0',
            'mne': '1.0.0',
        }
        
        original_version = importlib.metadata.version
        
        def patched_version(package_name):
            try:
                return original_version(package_name)
            except importlib.metadata.PackageNotFoundError:
                name_lower = package_name.lower().replace('-', '_')
                if name_lower in KNOWN_VERSIONS:
                    return KNOWN_VERSIONS[name_lower]
                raise
        
        importlib.metadata.version = patched_version
    except Exception:
        pass

# Apply patches
patch_packaging_version()
patch_importlib_metadata()
