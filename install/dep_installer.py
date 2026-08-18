# -*- coding: utf-8 -*-
"""
install/dep_installer.py — Instalação silenciosa de dependências.

Estratégia de instalação (em ordem de prioridade):
  1. Reutiliza pacote já instalado no ambiente (sem fazer nada).
  2. Instala wheel do diretório 'wheels/' na UNC de origem.
  3. Faz download silencioso do PyPI (com índice PyTorch correto para GPU).

Nunca instala no Python global — sempre no venv isolado criado em
%LOCALAPPDATA%\VideoFrameExtractor\venv.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _run_pip(
    args: list[str],
    venv_python: Optional[Path] = None,
) -> bool:
    """
    Executa o pip com os argumentos fornecidos.
    Usa o Python do venv se fornecido; caso contrário, usa o atual.
    Retorna True se bem-sucedido.
    """
    python_exe = str(venv_python) if venv_python else sys.executable
    cmd = [python_exe, '-m', 'pip', 'install', '--quiet'] + args

    logger.debug("Executando: %s", ' '.join(cmd))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5 minutos max por instalação
        )
        if result.returncode != 0:
            logger.error(
                "pip falhou (código %d):\n%s",
                result.returncode,
                result.stderr.strip(),
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Timeout ao instalar via pip.")
        return False
    except Exception as exc:
        logger.error("Erro ao executar pip: %s", exc)
        return False


def get_installed_packages_in_venv(venv_python: Optional[Path]) -> dict[str, str]:
    """
    Lista os pacotes instalados no venv usando 'pip list' via subprocess.
    Garante verificação no contexto correto do venv (não no Python global).
    Retorna dicionário {nome_lower: versão}.
    """
    python_exe = str(venv_python) if venv_python else sys.executable
    try:
        result = subprocess.run(
            [python_exe, '-m', 'pip', 'list', '--format=freeze'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        packages: dict[str, str] = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if '==' in line:
                    name, _, version = line.partition('==')
                    # Indexa com hífen e underscore para ambas as formas de nome
                    packages[name.lower()] = version
                    packages[name.lower().replace('-', '_')] = version
        logger.debug("Pacotes encontrados no venv: %d", len(packages) // 2)
        return packages
    except Exception as exc:
        logger.warning("Erro ao listar pacotes do venv: %s", exc)
        return {}


def create_venv(venv_dir: Path) -> Optional[Path]:
    """
    Cria um venv isolado no caminho indicado.
    Retorna o caminho do executável Python do venv, ou None se falhar.
    """
    logger.info("Criando venv em: %s", venv_dir)
    
    # Determina o executável Python real a utilizar
    if getattr(sys, 'frozen', False):
        from install.env_detector import _find_system_python
        python_exe = _find_system_python()
        if not python_exe:
            logger.error("Nao foi possivel encontrar um interpretador Python instalado no sistema para criar o venv.")
            return None
    else:
        python_exe = sys.executable

    try:
        result = subprocess.run(
            [python_exe, '-m', 'venv', str(venv_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120, # Maior tempo limite para segurança
        )
        if result.returncode == 0:
            # Caminho do Python dentro do venv (Windows)
            venv_python = venv_dir / 'Scripts' / 'python.exe'
            if not venv_python.exists():
                # Unix fallback
                venv_python = venv_dir / 'bin' / 'python'

            if venv_python.exists():
                logger.info("Venv criado com sucesso: %s", venv_python)
                return venv_python
        
        # Registra a falha detalhada e retorna None
        stderr_msg = result.stderr or ""
        logger.error("Falha ao criar venv (retorno %d): %s", result.returncode, stderr_msg)

    except Exception as exc:
        logger.error("Erro ao criar venv: %s", exc)

    return None


def _find_wheel_for_package(
    package_name: str,
    wheels: list[Path],
) -> Optional[Path]:
    """
    Procura uma wheel compatível para o pacote indicado na lista de wheels locais.
    Comparação case-insensitive e ignora hífens/underscores.
    """
    normalized = package_name.lower().replace('-', '_').replace('.', '_')
    for wheel in wheels:
        wheel_name = wheel.stem.lower().replace('-', '_').split('-')[0]
        if wheel_name == normalized:
            return wheel
    return None


def install_package(
    package_spec: str,
    installed_packages: dict[str, str],
    local_wheels: list[Path],
    venv_python: Optional[Path],
    torch_index_url: Optional[str] = None,
) -> bool:
    """
    Instala um pacote usando a estratégia de 3 camadas.

    Args:
        package_spec: Especificação do pacote (ex: 'opencv-python>=4.9,<5').
        installed_packages: Pacotes já instalados (nome_lower → versão).
        local_wheels: Wheels disponíveis na UNC de origem.
        venv_python: Executável Python do venv destino.
        torch_index_url: URL do índice PyTorch para GPU (apenas para torch/torchvision).

    Returns:
        True se instalação bem-sucedida (ou desnecessária).
    """
    # Extrai nome base do pacote (antes de >= < etc.)
    import re
    base_name = re.split(r'[>=<!;\[,]', package_spec)[0].strip().lower()
    normalized_name = base_name.replace('-', '_')

    # 1. Reutiliza se já instalado
    if normalized_name in installed_packages or base_name in installed_packages:
        existing_ver = installed_packages.get(normalized_name) or installed_packages.get(base_name)
        logger.info(
            "  [OK] %-35s ja instalado (v%s) - reutilizando.",
            base_name,
            existing_ver,
        )
        return True

    # 2. Tenta wheel local (UNC)
    wheel = _find_wheel_for_package(base_name, local_wheels)
    if wheel:
        logger.info(
            "  [INSTALL] %-35s instalando via wheel local: %s",
            base_name,
            wheel.name,
        )
        if _run_pip([str(wheel), '--no-deps'], venv_python):
            logger.info("  [OK] %-35s instalado via wheel local.", base_name)
            return True
        logger.warning("  [FAIL] Wheel local falhou, tentando PyPI.")

    # 3. Download do PyPI
    pip_args = [package_spec]

    # Adiciona índice PyTorch para torch/torchvision
    if torch_index_url and base_name in ('torch', 'torchvision', 'torchaudio'):
        pip_args += ['--index-url', torch_index_url]

    logger.info("  [INSTALL] %-35s instalando via PyPI...", base_name)
    if _run_pip(pip_args, venv_python):
        logger.info("  [OK] %-35s instalado via PyPI.", base_name)
        return True

    logger.error("  [FAIL] FALHA ao instalar: %s", package_spec)
    return False


def install_all_requirements(
    requirements_path: Path,
    installed_packages: dict[str, str],
    local_wheels: list[Path],
    venv_python: Optional[Path],
    torch_index_url: Optional[str] = None,
) -> tuple[int, int]:
    """
    Instala todas as dependências do requirements.txt.

    Returns:
        Tupla (sucessos, falhas).
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt não encontrado: {requirements_path}")

    lines = requirements_path.read_text(encoding='utf-8').splitlines()
    packages = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith('#')
    ]

    sucessos = 0
    falhas = 0

    logger.info("Instalando %d dependências...", len(packages))
    for pkg in packages:
        ok = install_package(
            package_spec=pkg,
            installed_packages=installed_packages,
            local_wheels=local_wheels,
            venv_python=venv_python,
            torch_index_url=torch_index_url,
        )
        if ok:
            sucessos += 1
        else:
            falhas += 1

    return sucessos, falhas
