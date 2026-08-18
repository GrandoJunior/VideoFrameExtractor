# -*- coding: utf-8 -*-
"""
install/env_detector.py — Detecção do ambiente local.

Responsabilidades:
- Detectar versão do Python (>= 3.10 requerido).
- Detectar presença de CUDA e versão do driver NVIDIA.
- Listar pacotes já instalados no ambiente atual.
- Detectar wheels disponíveis no diretório UNC de origem.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_system_python() -> Optional[str]:
    """
    Procura por um interpretador Python real e valido no sistema (nao o stub da MS Store).
    Retorna o caminho absoluto do python.exe ou None.
    """
    import shutil
    import subprocess
    
    # 1. Verifica 'py.exe' (Python Launcher)
    py_exe = shutil.which('py')
    if not py_exe and os.path.exists(r'C:\Windows\py.exe'):
        py_exe = r'C:\Windows\py.exe'
    if py_exe:
        try:
            res = subprocess.run(
                [py_exe, '-c', 'import sys; print(sys.executable)'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if res.returncode == 0:
                path = res.stdout.strip()
                if path and os.path.exists(path) and 'WindowsApps' not in path:
                    return path
        except Exception:
            pass

    # 2. Verifica 'python.exe' no PATH (filtrando o stub da MS Store)
    python_in_path = shutil.which('python')
    if python_in_path and 'WindowsApps' not in python_in_path:
        try:
            res = subprocess.run([python_in_path, '-c', 'import sys'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if res.returncode == 0:
                return python_in_path
        except Exception:
            pass

    # 3. Busca no Registro do Windows (HKCU e HKLM)
    try:
        import winreg
        for hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hkey, r"Software\Python\PythonCore") as core_key:
                    i = 0
                    while True:
                        try:
                            ver_name = winreg.EnumKey(core_key, i)
                            with winreg.OpenKey(core_key, f"{ver_name}\\InstallPath") as path_key:
                                path_val, _ = winreg.QueryValueEx(path_key, "ExecutablePath")
                                if path_val and os.path.exists(path_val) and 'WindowsApps' not in path_val:
                                    return path_val
                        except OSError:
                            break
                        i += 1
            except Exception:
                pass
    except Exception:
        pass

    # 4. Busca em caminhos padroes do Windows
    user_local = os.getenv('LOCALAPPDATA', '')
    program_files = os.getenv('ProgramFiles', 'C:\\Program Files')
    program_files_x86 = os.getenv('ProgramFiles(x86)', 'C:\\Program Files (x86)')
    
    search_dirs = []
    if user_local:
        search_dirs.append(Path(user_local) / 'Programs' / 'Python')
    search_dirs.append(Path(program_files) / 'Python')
    search_dirs.append(Path(program_files_x86) / 'Python')
    
    for base_dir in search_dirs:
        if base_dir.exists():
            for p in base_dir.glob('Python*/python.exe'):
                if p.exists() and 'WindowsApps' not in str(p):
                    return str(p)

    # 5. Busca em caminhos customizados de desenvolvimento/portateis nos drives locais
    drives = ['C', 'D', 'G', 'H', 'K']
    custom_paths = [
        'Meu Drive/dev/programs/Python311/python.exe',
        'dev/programs/Python311/python.exe',
        'programs/Python311/python.exe',
        'Python311/python.exe',
    ]
    for d in drives:
        for s in custom_paths:
            p = Path(f"{d}:/{s}")
            if p.exists() and 'WindowsApps' not in str(p):
                return str(p)

    return None


def get_python_version() -> tuple[int, int]:
    """Retorna (major, minor) da versao Python real do sistema (ou do venv atual)."""
    if getattr(sys, 'frozen', False):
        python_exe = _find_system_python()
        if python_exe:
            try:
                res = subprocess.run(
                    [python_exe, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if res.returncode == 0:
                    parts = res.stdout.strip().split('.')
                    return int(parts[0]), int(parts[1])
            except Exception:
                pass
        return 0, 0
    return sys.version_info.major, sys.version_info.minor


def check_python_version(required_major: int = 3, required_minor: int = 10) -> bool:
    """Verifica se a versao Python atende ao minimo requerido."""
    major, minor = get_python_version()
    ok = (major, minor) >= (required_major, required_minor)
    if not ok:
        logger.error(
            "Python %d.%d detectado. Minimo requerido: %d.%d.",
            major, minor, required_major, required_minor,
        )
    return ok


def detect_cuda() -> dict:
    """
    Detecta presenca de CUDA e retorna informacoes do driver.

    Returns:
        {
            'available': bool,
            'device_name': str | None,
            'cuda_version': str | None,
            'vram_gb': float | None,
        }
    """
    result = {
        'available': False,
        'device_name': None,
        'cuda_version': None,
        'vram_gb': None,
    }
    
    # 1. Tenta usar torch primeiro (se ja estiver instalado no venv)
    try:
        import torch
        if torch.cuda.is_available():
            result['available'] = True
            result['device_name'] = torch.cuda.get_device_name(0)
            result['cuda_version'] = torch.version.cuda
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            result['vram_gb'] = round(vram_bytes / (1024 ** 3), 1)
            logger.info(
                "CUDA detectado via torch: %s | CUDA %s | VRAM %.1f GB",
                result['device_name'],
                result['cuda_version'],
                result['vram_gb'],
            )
            return result
        else:
            logger.info("CUDA nao disponivel via torch.")
    except ImportError:
        logger.debug("torch nao instalado; tentando via nvidia-smi.")
    except Exception as exc:
        logger.warning("Erro ao detectar CUDA via torch: %s", exc)

    # 2. Se torch nao estiver disponivel ou nao detectar CUDA, tenta via nvidia-smi
    import subprocess
    import re
    
    cmds = ['nvidia-smi', r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe']
    for cmd in cmds:
        try:
            creationflags = 0x08000000 if os.name == 'nt' else 0
            res = subprocess.run(
                [cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
                timeout=5
            )
            if res.returncode == 0:
                output = res.stdout
                result['available'] = True
                
                # Versao maxima do CUDA suportada pelo driver
                cuda_match = re.search(r'CUDA Version:\s*([\d\.]+)', output)
                if cuda_match:
                    result['cuda_version'] = cuda_match.group(1)
                
                # Nome do dispositivo
                device_match = re.search(r'\|\s*\d+\s+([^\|]+?)\s+(?:WDDM|TCC)\s*\|', output)
                if device_match:
                    result['device_name'] = device_match.group(1).strip()
                
                # VRAM total
                vram_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', output)
                if vram_match:
                    total_mib = int(vram_match.group(2))
                    result['vram_gb'] = round(total_mib / 1024, 1)
                
                logger.info(
                    "CUDA detectado via nvidia-smi: %s | CUDA %s | VRAM %.1f GB",
                    result['device_name'],
                    result['cuda_version'],
                    result['vram_gb'],
                )
                break
        except Exception as exc:
            logger.debug("Falha ao rodar %s: %s", cmd, exc)
            
    return result


def get_installed_packages() -> dict[str, str]:
    """
    Retorna dicionário {nome_pacote_lower: versão} de todos os pacotes instalados.
    Usa importlib.metadata para compatibilidade com venvs e instalações customizadas.
    """
    packages: dict[str, str] = {}
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get('Name', '').lower()
            version = dist.metadata.get('Version', '')
            if name:
                packages[name] = version
    except Exception as exc:
        logger.warning("Erro ao listar pacotes instalados: %s", exc)
    return packages


def find_local_wheels(source_dir: Path) -> list[Path]:
    """
    Procura wheels (.whl) no subdiretório 'wheels' do diretório de origem.
    Suporta UNC sem remapeamento de drive.

    Args:
        source_dir: Diretório UNC ou local de onde o bootstrapper foi iniciado.

    Returns:
        Lista de caminhos de wheels encontrados.
    """
    wheels_dir = source_dir / 'wheels'
    if not wheels_dir.exists():
        logger.debug("Diretório de wheels não encontrado: %s", wheels_dir)
        return []

    wheels = list(wheels_dir.glob('*.whl'))
    logger.info(
        "Wheels locais encontradas em %s: %d arquivo(s).",
        wheels_dir,
        len(wheels),
    )
    return wheels


def get_torch_pip_index_url() -> str:
    """
    Retorna o índice PyPI correto para instalação do PyTorch com CUDA.
    Usa versão estável mais recente: cu124 para CUDA 12.x.
    """
    cuda_info = detect_cuda()
    if cuda_info['available'] and cuda_info['cuda_version']:
        cuda_ver = cuda_info['cuda_version'].replace('.', '')[:3]  # ex: '12.4' → '124'
        # Mapeamento para índices PyTorch oficiais
        cuda_index_map = {
            '124': 'https://download.pytorch.org/whl/cu124',
            '121': 'https://download.pytorch.org/whl/cu121',
            '118': 'https://download.pytorch.org/whl/cu118',
        }
        for cuda_key, url in cuda_index_map.items():
            if cuda_ver >= cuda_key:
                logger.info("Índice PyTorch CUDA selecionado: %s", url)
                return url
    # Fallback para CPU
    return 'https://download.pytorch.org/whl/cpu'


def build_environment_report() -> dict:
    """Gera relatório completo do ambiente para exibição no diagnóstico."""
    return {
        'python_version': '.'.join(str(v) for v in get_python_version()),
        'python_ok': check_python_version(),
        'cuda': detect_cuda(),
        'installed_packages_count': len(get_installed_packages()),
        'executable': sys.executable,
        'platform': sys.platform,
    }
