# -*- coding: utf-8 -*-
"""
core/image_enhancer.py — Aprimoramento de imagem em 3 camadas.

Pipeline de aprimoramento (em ordem de tentativa):
  Camada 1 — GPU Local (Real-ESRGAN): upscaling 4x com aceleração CUDA.
  Camada 2 — CPU Local: CLAHE + denoising NLMD + unsharp mask + white balance.
  Camada 3 — Groq Cloud: Apenas análise de ajustes (não faz processamento de pixel).

A Camada 1 (Real-ESRGAN) é aplicada SOMENTE se ENHANCEMENT_APPLY_UPSCALING=True.
A Camada 2 é sempre aplicada como tratamento base padrão.
A Camada 3 é usada para metadados/análise, não para transformação de pixel.

Saída: imagem JPEG de alta qualidade salva no caminho indicado.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    ENHANCEMENT_APPLY_CLAHE,
    ENHANCEMENT_APPLY_DENOISE,
    ENHANCEMENT_APPLY_SHARPEN,
    ENHANCEMENT_APPLY_UPSCALING,
    ENHANCEMENT_APPLY_WHITE_BALANCE,
    JPEG_QUALITY,
    LOCAL_MODELS_DIR,
    REALESRGAN_MODEL,
    REALESRGAN_TILE,
)

logger = logging.getLogger(__name__)

# Cache do upsampler Real-ESRGAN (carregado uma vez por sessão)
_REALESRGAN_UPSAMPLER: Optional[object] = None
_REALESRGAN_INIT_ATTEMPTED = False


# ── Camada 2: Aprimoramento CPU ───────────────────────────────────────────────

def _apply_auto_white_balance(bgr: np.ndarray) -> np.ndarray:
    """
    Correção automática de balanço de branco via equalização de canais.
    Método: Gray World Assumption — assume que a média de cada canal
    deve ser igual (neutral grey world).
    """
    result = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] -= (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    result[:, :, 2] -= (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """
    Equalização adaptativa de histograma (CLAHE) no canal L do espaço LAB.
    Melhora contraste local sem superexpor áreas já claras.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _apply_nlmd_denoise(bgr: np.ndarray) -> np.ndarray:
    """
    Denoising via Non-Local Means Denoising (NLMD).
    Configuração conservadora para preservar detalhes finos.
    """
    return cv2.fastNlMeansDenoisingColored(bgr, None, h=5, hColor=5,
                                            templateWindowSize=7,
                                            searchWindowSize=21)


def _apply_unsharp_mask(bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Unsharp Mask suave para realce de bordas sem artefatos.
    strength: intensidade [0.0–1.0]. Default 0.5 = sutil.
    """
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)
    sharpened = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_cpu(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Aplica pipeline de aprimoramento CPU conforme configurações em config.py.
    Ordem: White Balance → CLAHE → Denoising → Unsharp Mask.
    """
    enhanced = bgr_frame.copy()

    if ENHANCEMENT_APPLY_WHITE_BALANCE:
        enhanced = _apply_auto_white_balance(enhanced)
        logger.debug("White balance aplicado.")

    if ENHANCEMENT_APPLY_CLAHE:
        enhanced = _apply_clahe(enhanced)
        logger.debug("CLAHE aplicado.")

    if ENHANCEMENT_APPLY_DENOISE:
        enhanced = _apply_nlmd_denoise(enhanced)
        logger.debug("NLMD denoising aplicado.")

    if ENHANCEMENT_APPLY_SHARPEN:
        enhanced = _apply_unsharp_mask(enhanced)
        logger.debug("Unsharp mask aplicado.")

    return enhanced


# ── Camada 1: Real-ESRGAN GPU ─────────────────────────────────────────────────

def _init_realesrgan() -> Optional[object]:
    """
    Inicializa e cacheia o upsampler Real-ESRGAN.
    Tenta CUDA primeiro; fallback para CPU.
    """
    global _REALESRGAN_UPSAMPLER, _REALESRGAN_INIT_ATTEMPTED
    if _REALESRGAN_INIT_ATTEMPTED:
        return _REALESRGAN_UPSAMPLER
    _REALESRGAN_INIT_ATTEMPTED = True

    try:
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        model_path = LOCAL_MODELS_DIR / 'RealESRGAN_x4plus.pth'

        # Download automático do modelo se não existir
        if not model_path.exists():
            _download_realesrgan_model(model_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        half_precision = device == 'cuda'

        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=REALESRGAN_TILE,
            tile_pad=10,
            pre_pad=0,
            half=half_precision,
            device=device,
        )
        _REALESRGAN_UPSAMPLER = upsampler
        logger.info("Real-ESRGAN inicializado no dispositivo '%s'.", device)

    except Exception as exc:
        logger.warning("Real-ESRGAN indisponível: %s. Upscaling desabilitado.", exc)
        _REALESRGAN_UPSAMPLER = None

    return _REALESRGAN_UPSAMPLER


def _download_realesrgan_model(model_path: Path) -> None:
    """Baixa o modelo Real-ESRGAN se não estiver em cache local."""
    import urllib.request

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/'
        'RealESRGAN_x4plus.pth'
    )
    logger.info("Baixando modelo Real-ESRGAN (~65MB)... %s", url)

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  Baixando modelo: {pct}%", end='', flush=True)

    urllib.request.urlretrieve(url, str(model_path), progress_hook)
    print()  # Nova linha após a barra de progresso
    logger.info("Modelo salvo em: %s", model_path)


def enhance_gpu_realesrgan(bgr_frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Aprimora o frame com Real-ESRGAN (upscaling 4x).
    Retorna None se Real-ESRGAN não estiver disponível.
    """
    upsampler = _init_realesrgan()
    if upsampler is None:
        return None

    try:
        output, _ = upsampler.enhance(bgr_frame, outscale=4)
        logger.debug("Real-ESRGAN: upscaling 4x aplicado com sucesso.")
        return output
    except Exception as exc:
        logger.warning("Falha no Real-ESRGAN: %s. Revertendo para CPU.", exc)
        return None


# ── Pipeline principal ────────────────────────────────────────────────────────

def enhance_and_save(
    bgr_frame: np.ndarray,
    output_path: Path,
    apply_enhancement: bool = True,
) -> bool:
    """
    Salva o frame como JPEG. Se apply_enhancement for True, aplica o
    pipeline completo de aprimoramento (Real-ESRGAN, CLAHE, denoising, etc.).
    Caso contrário, salva o frame original rigorosamente fiel ao vídeo.

    Args:
        bgr_frame: Frame em formato BGR (NumPy uint8).
        output_path: Caminho completo do arquivo de saída (UNC-safe).
        apply_enhancement: Se True, aplica filtros de aprimoramento.

    Returns:
        True se salvou com sucesso, False se falhou.
    """
    enhanced = bgr_frame.copy()

    if apply_enhancement:
        # Camada 1: Upscaling com Real-ESRGAN (opcional)
        if ENHANCEMENT_APPLY_UPSCALING:
            upscaled = enhance_gpu_realesrgan(enhanced)
            if upscaled is not None:
                enhanced = upscaled
            else:
                logger.info("Upscaling ignorado (Real-ESRGAN indisponível).")

        # Camada 2: Aprimoramento CPU (sempre aplicado)
        enhanced = enhance_cpu(enhanced)

    # Salvar
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        success, buffer = cv2.imencode('.jpg', enhanced, encode_params)
        if not success:
            raise RuntimeError("cv2.imencode falhou.")

        output_path.write_bytes(buffer.tobytes())
        logger.debug("Frame salvo (aprimoramento=%s): %s", apply_enhancement, output_path)
        return True

    except Exception as exc:
        logger.error("Falha ao salvar frame em %s: %s", output_path, exc)
        return False
