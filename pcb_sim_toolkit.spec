# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for PCB Simulation Toolkit macOS .app bundle."""

import os
from pathlib import Path

block_cipher = None
project_root = os.path.abspath(".")

a = Analysis(
    ["run_gui.py"],
    pathex=[project_root],
    binaries=[],
    datas=[
        ("examples", "examples"),
        ("configs", "configs"),
    ],
    hiddenimports=[
        # src submodules
        "src",
        "src.core",
        "src.core.cli",
        "src.core.config",
        "src.core.models",
        "src.core.simulator",
        "src.analysis",
        "src.analysis.magnetics",
        "src.analysis.signal_integrity",
        "src.analysis.spice",
        "src.analysis.thermal",
        "src.exporters",
        "src.exporters.magnetics_plots",
        "src.exporters.report",
        "src.gui",
        "src.gui.app",
        "src.parsers",
        "src.parsers.kicad_loader",
        "src.parsers.yaml_loader",
        "src.resource_path",
        # scipy submodules commonly needed
        "scipy.special",
        "scipy.special._ufuncs",
        "scipy.special._ufuncs_cxx",
        "scipy.linalg",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        "scipy.sparse",
        "scipy.sparse.csgraph",
        "scipy.sparse.linalg",
        "scipy.ndimage",
        "scipy.interpolate",
        "scipy.integrate",
        "scipy.optimize",
        # tkinter
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        # matplotlib backends
        "matplotlib.backends.backend_tkagg",
        # other
        "yaml",
        "numpy",
        "webbrowser",
        "json",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
        "sphinx",
        "gi",
        "gi.repository",
        "gi.repository.GLib",
        "gi.repository.Gtk",
        "gi.repository.Gdk",
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
    name="PCB Simulation Toolkit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PCB Simulation Toolkit",
)

icon_path = os.path.join(project_root, "assets", "icon.icns")

app = BUNDLE(
    coll,
    name="PCB Simulation Toolkit.app",
    icon=icon_path if os.path.exists(icon_path) else None,
    bundle_identifier="com.pcbsim.toolkit",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleName": "PCB Simulation Toolkit",
    },
)
