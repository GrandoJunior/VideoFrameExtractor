# -*- coding: utf-8 -*-
"""
config.py — Configurações centralizadas do VideoFrameExtractor.

Princípio de responsabilidade única: todas as constantes e caminhos
configuráveis residem aqui. Os demais módulos importam deste arquivo.
"""

import os
import sys
import codecs
import base64
from pathlib import Path

# ── Detecção dinâmica do diretório do projeto ─────────────────────────────────
# Funciona tanto ao rodar como .py quanto como .exe (PyInstaller frozen)

if getattr(sys, 'frozen', False):
    PROJECT_DIR = Path(sys.executable).parent
else:
    PROJECT_DIR = Path(__file__).resolve().parent

# ── Diretório de dados locais (instalação por máquina) ───────────────────────
# Nunca usar UNC para dados locais — performance e permissões
LOCAL_DATA_DIR = Path(os.getenv('LOCALAPPDATA', Path.home())) / 'VideoFrameExtractor'
LOCAL_VENV_DIR = LOCAL_DATA_DIR / 'venv'
LOCAL_MODELS_DIR = LOCAL_DATA_DIR / 'models'
LOCAL_CACHE_DIR = LOCAL_DATA_DIR / 'cache'
SETUP_FLAG = LOCAL_DATA_DIR / 'setup_done.flag'

# ── Extensões de vídeo suportadas ────────────────────────────────────────────
VIDEO_EXTENSIONS = frozenset({
    '.mp4', '.mov', '.avi', '.mkv', '.wmv',
    '.3gp', '.mts', '.m4v', '.flv', '.webm',
    '.ts', '.mpg', '.mpeg', '.m2ts',
})

# ── Configuração de extração de frames ───────────────────────────────────────
# Pasta de saída padrão (subpasta relativa à pasta dos vídeos)
OUTPUT_SUBDIR = 'frames_extraidos'

# Qualidade JPEG de saída [1–95]
JPEG_QUALITY = 93

# Distância mínima em segundos entre frames selecionados (evita duplicatas)
MIN_FRAME_DISTANCE_SEC = 2.0

# Sugestão automática de quantidade de frames por duração do vídeo (em segundos)
FRAME_COUNT_SUGGESTIONS: list[tuple[float, int]] = [
    (60,   5),    # < 1 min  → 5 frames
    (300,  10),   # < 5 min  → 10 frames
    (1800, 20),   # < 30 min → 20 frames
    (float('inf'), 30),  # >= 30 min → 30 frames
]

# ── Pesos do score de qualidade ───────────────────────────────────────────────
# Soma deve ser 1.0
QUALITY_WEIGHTS = {
    'laplacian':   0.35,   # Nitidez via variância do Laplaciano
    'brisque':     0.30,   # Qualidade perceptual sem referência (pyiqa)
    'illumination': 0.20,  # Qualidade de iluminação (histograma)
    'tenengrad':   0.15,   # Ausência de motion blur (gradiente Sobel)
}

# ── Aprimoramento de imagem ───────────────────────────────────────────────────
# Por padrão: apenas ajustes leves (CLAHE, denoising, sharpening) sem upscaling.
# Ativar upscaling Real-ESRGAN apenas se o usuário configurar explicitamente.
ENHANCEMENT_APPLY_UPSCALING = False   # Real-ESRGAN 4x (requer GPU ou muito tempo CPU)
ENHANCEMENT_APPLY_CLAHE = True        # Equalização adaptativa de histograma
ENHANCEMENT_APPLY_DENOISE = True      # Denoising NLMD (Non-Local Means)
ENHANCEMENT_APPLY_SHARPEN = True      # Unsharp Mask suave
ENHANCEMENT_APPLY_WHITE_BALANCE = True  # Correção automática de balanço de branco

# Modelo Real-ESRGAN a usar para upscaling
REALESRGAN_MODEL = 'RealESRGAN_x4plus'
REALESRGAN_TILE = 256  # Tamanho do tile para gerenciamento de VRAM

# ── Groq Cloud API (fallback para análise de qualidade) ──────────────────────
# Chave ofuscada em base64+rot13 — mesma chave do transcritorwhats
# Groq é usado APENAS para análise/scoring quando pyiqa não está disponível.
# NÃO faz processamento de pixel (upscaling).
GROQ_MODEL_VISION = 'meta-llama/llama-4-scout-17b-16e-instruct'
GROQ_TIMEOUT_SEC = 30

# ── Logging ───────────────────────────────────────────────────────────────────
# Alterado para LOGS localizados no AppData do usuário
LOG_DIR = LOCAL_DATA_DIR / 'logs'
LOG_LEVEL = 'INFO'  # DEBUG | INFO | WARNING | ERROR


def resolve_groq_api_key() -> str:
    """Retorna a chave Groq a partir da variável de ambiente GROQ_API_KEY. Retorna '' se não configurada."""
    return os.getenv('GROQ_API_KEY', '').strip()

