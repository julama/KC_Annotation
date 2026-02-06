#!/usr/bin/env python3
"""
Build script for creating standalone EXE of KC Annotation Tool.

Usage:
    python build_exe.py                    # Build one-folder distribution
    python build_exe.py --onefile          # Build single-file EXE
    python build_exe.py --clean            # Clean build artifacts before building
    python build_exe.py --windowed         # Hide console window (GUI only)

Requirements:
    pip install pyinstaller
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_artifacts():
    """Remove previous build artifacts."""
    print("Cleaning build artifacts...")
    artifacts = ['build', 'dist', '__pycache__']
    spec_files = list(Path('.').glob('*.spec'))
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            print(f"  Removing {artifact}/")
            shutil.rmtree(artifact)
    
    # Clean __pycache__ in subdirectories
    for pycache in Path('.').rglob('__pycache__'):
        print(f"  Removing {pycache}/")
        shutil.rmtree(pycache)
    
    print("Clean complete.\n")


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required = ['pyinstaller', 'panel', 'holoviews', 'bokeh', 'scipy', 'mne']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nError: Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)
    
    print("All dependencies found.\n")


def build_exe(onefile=False, windowed=False, debug=False):
    """Run PyInstaller to build the executable."""
    print("Building executable...")
    print(f"  Mode: {'one-file' if onefile else 'one-folder'}")
    print(f"  Console: {'hidden' if windowed else 'visible'}")
    print()
    
    # Build command
    cmd = ['pyinstaller', 'kc_annotation.spec']
    
    if onefile:
        cmd.append('--onefile')
    
    if debug:
        cmd.append('--debug=all')
    
    # Note: windowed mode is controlled in the spec file
    # We'd need to modify the spec file dynamically for this
    if windowed:
        print("Note: To enable windowed mode, edit kc_annotation.spec and set console=False")
    
    print(f"Running: {' '.join(cmd)}\n")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    
    print("=" * 60)
    
    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print("\nBuild complete!")
    
    # Print output location
    if onefile:
        exe_path = Path('dist/kc_annotation')
        if sys.platform == 'win32':
            exe_path = exe_path.with_suffix('.exe')
    else:
        exe_path = Path('dist/kc_annotation')
        if sys.platform == 'win32':
            exe_path = exe_path / 'kc_annotation.exe'
        else:
            exe_path = exe_path / 'kc_annotation'
    
    print(f"\nOutput: {exe_path}")
    
    if exe_path.exists() or exe_path.parent.exists():
        print("\nTo run the application:")
        if sys.platform == 'win32':
            print(f"  {exe_path} --mat-file <path_to_file.mat>")
        else:
            print(f"  ./{exe_path} --mat-file <path_to_file.mat>")
        print("\nOr double-click the executable and use the file dialog.")


def main():
    parser = argparse.ArgumentParser(
        description='Build standalone EXE for KC Annotation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_exe.py                    # Standard one-folder build
    python build_exe.py --onefile          # Single EXE file (slower startup)
    python build_exe.py --clean            # Clean first, then build
    python build_exe.py --clean --onefile  # Clean and build single file
        """
    )
    
    parser.add_argument('--onefile', action='store_true',
                       help='Create a single executable file (slower startup)')
    parser.add_argument('--windowed', action='store_true',
                       help='Hide console window (GUI-only mode)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean build artifacts before building')
    parser.add_argument('--clean-only', action='store_true',
                       help='Only clean, do not build')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for troubleshooting')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("KC Annotation Tool - EXE Builder")
    print("=" * 60)
    print()
    
    # Clean if requested
    if args.clean or args.clean_only:
        clean_build_artifacts()
    
    if args.clean_only:
        print("Clean-only mode, skipping build.")
        return
    
    # Check dependencies
    if not args.skip_checks:
        check_dependencies()
    
    # Check spec file exists
    if not Path('kc_annotation.spec').exists():
        print("Error: kc_annotation.spec not found!")
        print("Make sure you're running this script from the project root.")
        sys.exit(1)
    
    # Build
    build_exe(
        onefile=args.onefile,
        windowed=args.windowed,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
