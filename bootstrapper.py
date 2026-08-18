# -*- coding: utf-8 -*-
"""
bootstrapper.py — Executável mínimo auto-instalável do VideoFrameExtractor.

Comportamento:
  - Copia/sincroniza todos os arquivos do projeto da origem (pode ser UNC)
    para a máquina local (%LOCALAPPDATA%\VideoFrameExtractor\app).
  - Executa a configuração inicial e a instalação de dependências localmente.
  - Cria um atalho na Área de Trabalho apontando para o executável da rede.
  - Ativa o venv local e lança o main.py local.
"""

from __future__ import annotations

import os
import sys
import subprocess
import ctypes
import shutil
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────

APP_NAME = 'VideoFrameExtractor'

# Diretório de dados locais (por máquina, não UNC)
LOCAL_DATA_DIR = Path(os.getenv('LOCALAPPDATA', Path.home())) / APP_NAME
LOCAL_APP_DIR = LOCAL_DATA_DIR / 'app'
VENV_DIR = LOCAL_DATA_DIR / 'venv'
SETUP_FLAG = LOCAL_DATA_DIR / 'setup_done.flag'
LOG_FILE = LOCAL_DATA_DIR / 'bootstrapper.log'

# Diretório onde o .exe está localizado (pode ser UNC de origem)
if getattr(sys, 'frozen', False):
    # Rodando como .exe (PyInstaller)
    EXE_DIR = Path(sys.executable).resolve().parent
else:
    # Rodando como .py (modo desenvolvimento)
    EXE_DIR = Path(__file__).resolve().parent

REQUIREMENTS_PATH = EXE_DIR / 'requirements.txt'


# ── Utilitários ───────────────────────────────────────────────────────────────

def _log(message: str) -> None:
    """Log simples para arquivo e stdout durante o bootstrap."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        from datetime import datetime
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)


def _get_venv_python() -> Path:
    """Retorna o caminho do Python no venv (Windows ou Unix), ou fallback global."""
    win_python = VENV_DIR / 'Scripts' / 'python.exe'
    unix_python = VENV_DIR / 'bin' / 'python'
    if win_python.exists():
        return win_python
    if unix_python.exists():
        return unix_python
        
    try:
        from install.env_detector import _find_system_python
        sys_python = _find_system_python()
        if sys_python:
            return Path(sys_python)
    except Exception:
        pass
        
    return win_python


def _check_python_version() -> bool:
    """Verifica versão mínima do Python."""
    return sys.version_info >= (3, 10)


# ── Sincronização Local ───────────────────────────────────────────────────────

def _copy_item_recursive(src: Path, dst: Path) -> None:
    """Copia arquivos/pastas de forma recursiva e otimizada (apenas se alterados)."""
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name in ('__pycache__', '.git', 'build', 'dist', 'teste_output') or item.suffix in ('.pyc', '.pyo'):
                continue
            _copy_item_recursive(item, dst / item.name)
    else:
        should_copy = True
        if dst.exists():
            try:
                src_stat = src.stat()
                dst_stat = dst.stat()
                if src_stat.st_size == dst_stat.st_size and abs(src_stat.st_mtime - dst_stat.st_mtime) < 2:
                    should_copy = False
            except Exception:
                pass
        if should_copy:
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                _log(f"Aviso: Falha ao copiar {src.name} -> {dst.name}: {exc}")


def _sync_project_to_local() -> bool:
    """Sincroniza os arquivos de execução do projeto UNC para a máquina local."""
    _log(f"Sincronizando arquivos do projeto para execucao local...")
    _log(f"Origem: {EXE_DIR}")
    _log(f"Destino: {LOCAL_APP_DIR}")
    
    items_to_copy = [
        'main.py',
        'config.py',
        'requirements.txt',
        'diagnostico_brisque.py',
        'diagnostico_frameloop.py',
        'diagnostico_isolamento.py',
        'run_e2e_test.py',
        'core',
        'install',
    ]
    
    try:
        LOCAL_APP_DIR.mkdir(parents=True, exist_ok=True)
        for item in items_to_copy:
            src_path = EXE_DIR / item
            dst_path = LOCAL_APP_DIR / item
            if src_path.exists():
                _copy_item_recursive(src_path, dst_path)
        return True
    except Exception as exc:
        _log(f"ERRO durante a sincronizacao local: {exc}")
        return False


def _create_desktop_shortcut() -> None:
    """Cria um atalho na área de trabalho apontando para o executável ou script na rede."""
    try:
        desktop = Path(os.environ['USERPROFILE']) / 'Desktop'
        if not desktop.exists():
            desktop = Path(os.path.expanduser('~')) / 'Desktop'
            
        shortcut_path = desktop / 'VideoFrameExtractor.lnk'
        
        # O alvo do atalho é o executável na rede
        target_path = EXE_DIR / 'VideoFrameExtractor.exe'
        if not target_path.exists():
            target_path = EXE_DIR / 'executar.bat'
            
        _log(f"Criando atalho na area de trabalho apontando para: {target_path}")
        
        # Escape seguro de aspas simples para PowerShell (' -> '')
        safe_shortcut = str(shortcut_path).replace("'", "''")
        safe_target = str(target_path).replace("'", "''")
        safe_wdir = str(EXE_DIR).replace("'", "''")
        
        # Script powershell para criar atalho .lnk
        ps_script = (
            f"$WshShell = New-Object -ComObject WScript.Shell; "
            f"$Shortcut = $WshShell.CreateShortcut('{safe_shortcut}'); "
            f"$Shortcut.TargetPath = '{safe_target}'; "
            f"$Shortcut.WorkingDirectory = '{safe_wdir}'; "
            f"$Shortcut.Description = 'VideoFrameExtractor - Extrator Inteligente'; "
            f"$Shortcut.Save()"
        )
        
        subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=0x08000000
        )
        _log("Atalho na area de trabalho criado com sucesso!")
    except Exception as exc:
        _log(f"Aviso: Nao foi possivel criar o atalho na area de trabalho: {exc}")


# ── Primeiro uso: setup completo ──────────────────────────────────────────────

def _run_first_time_setup() -> bool:
    """
    Executa o setup de primeiro uso adicionando o diretório do projeto local ao path
    e chamando o setup_manager.
    """
    # Garante que o projeto local está no PYTHONPATH para importar install.*
    project_dir = str(LOCAL_APP_DIR)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        from install.setup_manager import run_first_time_setup, is_already_configured
    except ImportError as exc:
        _log(f"ERRO: Nao foi possivel importar setup_manager local: {exc}")
        return False

    if is_already_configured(SETUP_FLAG):
        return True

    # Passa EXE_DIR como source_dir para o setup_manager buscar wheels na pasta de origem (UNC)
    return run_first_time_setup(
        source_dir=EXE_DIR,
        requirements_path=LOCAL_APP_DIR / 'requirements.txt',
        venv_dir=VENV_DIR,
        setup_flag=SETUP_FLAG,
    )


# ── Lançamento do aplicativo principal ───────────────────────────────────────

def _launch_main_in_venv() -> int:
    """Lança o main.py local usando o Python do venv instalado."""
    venv_python = _get_venv_python()

    if not venv_python.exists():
        _log(f"ERRO: Python do venv nao encontrado em: {venv_python}")
        _log("Execute o VideoFrameExtractor novamente para reconfigurar.")
        SETUP_FLAG.unlink(missing_ok=True)
        return 1

    local_main = LOCAL_APP_DIR / 'main.py'
    if not local_main.exists():
        _log(f"ERRO: main.py nao encontrado localmente em: {local_main}")
        return 1

    cmd = [str(venv_python), str(local_main)]
    _log(f"Lancando localmente: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=str(LOCAL_APP_DIR))
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        _log(f"ERRO ao lancar main.py localmente: {exc}")
        return 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Evita crashes de encoding em consoles Windows cp1252/cp850
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(errors='replace')
        except Exception:
            pass

    # Configura o logger do dep_installer para exibir progresso no console
    import logging
    dep_logger = logging.getLogger('install.dep_installer')
    dep_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    dep_logger.addHandler(handler)
    dep_logger.propagate = False

    print()
    print(f"  VideoFrameExtractor - Bootstrapper v1.0")
    print(f"  Origem de rede: {EXE_DIR}")
    print()

    # 1. Sincroniza arquivos para máquina local
    if not _sync_project_to_local():
        print("  ERRO: Falha ao copiar arquivos do projeto para a maquina local.")
        input("  Pressione ENTER para sair...")
        sys.exit(1)

    # Garante que o projeto local está no PYTHONPATH para permitir imports do install.*
    project_dir = str(LOCAL_APP_DIR)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    LOCAL_REQ_PATH = LOCAL_APP_DIR / 'requirements.txt'

    # Verificação de Python
    if not _check_python_version():
        print(f"  ERRO: Python 3.10+ e requerido. Versao atual: {sys.version}")
        input("  Pressione ENTER para sair...")
        sys.exit(1)

    # 2. Verifica se o requirements.txt foi atualizado desde o último setup
    if SETUP_FLAG.exists() and LOCAL_REQ_PATH.exists():
        if LOCAL_REQ_PATH.stat().st_mtime > SETUP_FLAG.stat().st_mtime:
            _log("Nova versao do requirements.txt detectada. Atualizando dependencias locais...")
            SETUP_FLAG.unlink(missing_ok=True)

    # 3. Setup de primeiro uso ou atualizações
    if not SETUP_FLAG.exists():
        _log("Iniciando configuracao/instalacao de dependencias locais...")
        ok = _run_first_time_setup()
        if not ok:
            print("\n  ERRO: Configuracao local falhou.")
            print("  Verifique o log em:", LOG_FILE)
            input("  Pressione ENTER para sair...")
            sys.exit(1)
        # Cria o atalho na área de trabalho após o setup bem-sucedido
        _create_desktop_shortcut()

    # 4. Lança o aplicativo principal localmente no venv
    exit_code = _launch_main_in_venv()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
