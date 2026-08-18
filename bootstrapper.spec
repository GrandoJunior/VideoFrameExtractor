# bootstrapper.spec — PyInstaller spec para VideoFrameExtractor bootstrapper
# Gera um .exe único e compacto contendo apenas o bootstrapper (stdlib apenas).
# O bootstrapper gerencia a instalação de todas as dependências pesadas no venv.

block_cipher = None

a = Analysis(
    ['bootstrapper.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Inclui apenas os módulos de instalação e config (sem as deps pesadas)
        ('install/*.py', 'install'),
        ('config.py', '.'),
        ('requirements.txt', '.'),
    ],
    hiddenimports=[
        'install.env_detector',
        'install.dep_installer',
        'install.setup_manager',
        'importlib.metadata',
        'ctypes',
        'subprocess',
        'pathlib',
        'codecs',
        'base64',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclui tudo que é pesado — instalado no venv pelo bootstrapper
        'cv2', 'numpy', 'torch', 'torchvision', 'PIL', 'Pillow',
        'pyiqa', 'realesrgan', 'basicsr', 'groq', 'decord',
        'tqdm', 'rich', 'matplotlib',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VideoFrameExtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # Terminal visível (aplicativo CLI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Ícone opcional (descomente se tiver um .ico)
    # icon='assets/icon.ico',
)
