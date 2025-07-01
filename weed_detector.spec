from kivy_deps import sdl2, glew
import os
import kivymd

block_cipher = None

# Get KivyMD directory for icons
kivymd_dir = os.path.dirname(kivymd.__file__)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/weed_detector.tflite', 'assets'),
        ('assets/icon.ico', 'assets'),
        ('assets/splash.png', 'assets'),
        ('utils/tflite_predictor.py', 'utils'),
        # Add KivyMD icons and fonts
        (os.path.join(kivymd_dir, "icon_definitions.py"), "kivymd"),
        (os.path.join(kivymd_dir, "fonts"), "kivymd/fonts"),
        (os.path.join(kivymd_dir, "images"), "kivymd/images"),
    ],
    hiddenimports=[
        'kivymd.uix.fitimage',
        'cv2',
        'numpy',
        'tflite_runtime',
        'asyncgui',
        'asynckivy',
        'certifi',
        'charset-normalizer',
        'docutils',
        'exceptiongroup',
        'ffmpeg',
        'ffpyplayer',
        'filetype',
        'flatbuffers',
        'kivy',
        'kivy_deps.angle',
        'kivy_deps.glew',
        'kivy_deps.sdl2',
        'kivy.garden',
        'kivymd',
        'kivymd.icon_definitions',  # Add this line
        'materialyoucolor',
        'pillow',
        'pygments',
        'win32api',
        'win32gui',
        'win32con',
        'pythoncom',
        'win32com.shell',
        'win32com.shell.shell',
        'win32com.shell.shellcon',
        'win32com.client',
        'win32timezone',
        'pywintypes',
        'requests',
        'tflite',
        'urllib3'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash(
    'assets/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),  # Position of the text (x, y)
    text_size=14,       # Font size
    text_color='black', # Text color
    fullscreen=True,   # Set to True for fullscreen splash
    minify=False       # Don't minify the splash image
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    splash,
    splash.binaries,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    name='WeedDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
)