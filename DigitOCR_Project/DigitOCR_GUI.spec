# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "config"), "config"),
    (str(project_root / "data" / "input"), "data/input"),
    (str(project_root / "docs"), "docs"),
    (str(project_root / "handcount" / "assets"), "handcount/assets"),
    (str(project_root / "README.md"), "."),
]
binaries = []
hiddenimports = []

for package_name in ("paddleocr", "paddle", "paddlex", "numpy", "PIL", "cv2", "mediapipe"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

hiddenimports = sorted(set(hiddenimports + ["tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"]))

a = Analysis(
    ["gui_app.pyw"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DigitOCR_GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DigitOCR_GUI",
)
