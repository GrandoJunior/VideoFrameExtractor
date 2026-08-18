# -*- coding: utf-8 -*-
"""
install/setup_manager.py — Orquestrador do primeiro uso.

Coordena a detecção de ambiente e instalação de dependências,
exibindo progresso amigável em PT-BR via terminal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _print_banner() -> None:
    print()
    print("============================================================")
    print("        VideoFrameExtractor - Configuracao Inicial          ")
    print("============================================================")
    print()


def _print_step(step: int, total: int, description: str) -> None:
    print(f"  [{step}/{total}] {description}")


def run_first_time_setup(
    source_dir: Path,
    requirements_path: Path,
    venv_dir: Path,
    setup_flag: Path,
) -> bool:
    """
    Executa a configuração completa de primeiro uso.

    Args:
        source_dir: Diretório onde o bootstrapper está localizado (UNC ou local).
        requirements_path: Caminho para o requirements.txt do projeto.
        venv_dir: Diretório onde o venv será criado (%LOCALAPPDATA%/VideoFrameExtractor/venv).
        setup_flag: Arquivo de flag que indica instalação completa.

    Returns:
        True se a configuração foi concluída com sucesso.
    """
    from install.env_detector import (
        check_python_version,
        detect_cuda,
        find_local_wheels,
        get_torch_pip_index_url,
    )
    from install.dep_installer import (
        create_venv,
        get_installed_packages_in_venv,
        install_all_requirements,
    )

    _print_banner()

    total_steps = 5

    # Passo 1: Verificar Python
    _print_step(1, total_steps, "Verificando versao do Python...")
    if not check_python_version():
        print("\n  [ERRO] Python 3.10 ou superior e necessario.")
        print(f"    Versao atual: {sys.version}")
        return False
    print(f"    Python {sys.version.split()[0]} - OK")

    # Passo 2: Detectar hardware
    _print_step(2, total_steps, "Detectando hardware (GPU/CPU)...")
    cuda_info = detect_cuda()
    if cuda_info['available']:
        print(f"    GPU: {cuda_info['device_name']} | CUDA {cuda_info['cuda_version']} | {cuda_info['vram_gb']} GB VRAM")
    else:
        print("    GPU com CUDA nao detectada. Processamento via CPU.")

    # Passo 3: Criar venv
    _print_step(3, total_steps, f"Criando ambiente virtual em: {venv_dir}")
    venv_python = None
    if venv_dir.exists():
        win_py = venv_dir / 'Scripts' / 'python.exe'
        unix_py = venv_dir / 'bin' / 'python'
        if win_py.exists():
            venv_python = win_py
        elif unix_py.exists():
            venv_python = unix_py

    if venv_python is not None:
        print("    Ambiente virtual ja existe - reutilizando.")
    else:
        venv_python = create_venv(venv_dir)
        if venv_python is None:
            print("\n  [ERRO] Nao foi possivel criar o ambiente virtual.")
            return False
            
    # Se o venv_python retornado for o Python global do sistema (fallback)
    is_real_venv = False
    try:
        is_real_venv = venv_python.resolve() == (venv_dir / 'Scripts' / 'python.exe').resolve() or venv_python.resolve() == (venv_dir / 'bin' / 'python').resolve()
    except Exception:
        pass
        
    if not is_real_venv:
        print(f"    Usando interpretador global como fallback: {venv_python}")
    else:
        print(f"    Python do venv: {venv_python}")

    # Passo 4: Verificar dependências já instaladas no venv e wheels locais
    _print_step(4, total_steps, "Verificando dependencias disponiveis...")
    # Consulta pacotes do venv (não do Python global do bootstrapper)
    installed = get_installed_packages_in_venv(venv_python)
    local_wheels = find_local_wheels(source_dir)
    torch_index = get_torch_pip_index_url()
    pkg_count = len({k for k in installed if '_' not in k or '-' not in k}) // 1
    print(f"    {len(installed) // 2} pacotes ja instalados no venv, {len(local_wheels)} wheel(s) local(is).")

    # Passo 5: Instalar dependências
    _print_step(5, total_steps, "Instalando dependencias...")
    print()
    sucessos, falhas = install_all_requirements(
        requirements_path=requirements_path,
        installed_packages=installed,
        local_wheels=local_wheels,
        venv_python=venv_python,
        torch_index_url=torch_index,
    )
    print()

    if falhas > 0:
        print(f"  [AVISO] {falhas} dependencia(s) falharam na instalacao.")
        print("    Verifique a conectividade de rede e tente novamente.")
        print("    O aplicativo pode funcionar com funcionalidades reduzidas.")

    # Grava flag de instalação completa (mesmo com falhas parciais)
    setup_flag.parent.mkdir(parents=True, exist_ok=True)
    setup_flag.write_text(
        f"Instalacao concluida. Sucessos: {sucessos} | Falhas: {falhas}\n",
        encoding='utf-8',
    )

    print()
    print("  [OK] Configuracao inicial concluida!")
    print(f"    Instalados: {sucessos} | Ignorados/Falha: {falhas}")
    print()

    return True


def is_already_configured(setup_flag: Path) -> bool:
    """Verifica se o setup de primeiro uso já foi executado."""
    return setup_flag.exists()
