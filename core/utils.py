# -*- coding: utf-8 -*-
"""
core/utils.py — Funções utilitárias e de manipulação de caminhos.

Isolamento de lógica em conformidade com o princípio de Responsabilidade Única (SOLID).
"""

from __future__ import annotations

from pathlib import Path
from config import FRAME_COUNT_SUGGESTIONS


def suggest_frame_count(duration_seconds: float) -> int:
    """Retorna a sugestão automática de frames baseada na duração do vídeo."""
    for max_duration, suggested_count in FRAME_COUNT_SUGGESTIONS:
        if duration_seconds < max_duration:
            return suggested_count
    return FRAME_COUNT_SUGGESTIONS[-1][1]


def normalize_unc_path(raw_path: str) -> Path:
    """
    Normaliza um caminho UNC ou local para um objeto Path válido.
    Remove aspas, converte barras e resolve caracteres especiais PT-BR.
    """
    cleaned = raw_path.strip().strip('"').strip("'")
    # Normaliza separadores de diretório sem quebrar o prefixo UNC \\
    if cleaned.startswith('//'):
        cleaned = '\\\\' + cleaned[2:].replace('/', '\\')
    elif cleaned.startswith('\\\\'):
        pass  # Já é UNC correto
    else:
        cleaned = cleaned.replace('/', '\\')
    return Path(cleaned)
