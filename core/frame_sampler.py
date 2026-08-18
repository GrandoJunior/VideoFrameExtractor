# -*- coding: utf-8 -*-
"""
core/frame_sampler.py — Amostragem uniforme de frames de vídeo.

Responsabilidades:
- Amostrar N frames distribuídos uniformemente ao longo do vídeo.
- Suportar backends OpenCV (padrão) e decord (mais rápido, se disponível).
- Retornar frames como arrays NumPy (BGR para OpenCV, RGB para decord).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """Um frame amostrado de um vídeo com sua posição temporal."""

    index: int            # Índice do frame candidato (0-based)
    timestamp_sec: float  # Posição temporal no vídeo
    frame_number: int     # Número real do frame no vídeo
    image_bgr: np.ndarray = field(compare=False, repr=False)  # Imagem em BGR (formato OpenCV)


def _sample_with_opencv(
    video_path: Path,
    frame_numbers: list[int],
) -> list[SampledFrame]:
    """
    Amostrador via OpenCV VideoCapture.
    Funciona com todos os codecs suportados pelo OpenCV + FFmpeg.
    """
    import cv2

    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir: {path_str}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    results: list[SampledFrame] = []

    try:
        for idx, frame_num in enumerate(frame_numbers):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(
                    "Não foi possível ler o frame %d de %s",
                    frame_num,
                    video_path.name,
                )
                continue
            results.append(SampledFrame(
                index=idx,
                timestamp_sec=frame_num / fps,
                frame_number=frame_num,
                image_bgr=frame,
            ))
    finally:
        cap.release()

    return results


def _sample_with_decord(
    video_path: Path,
    frame_numbers: list[int],
) -> list[SampledFrame]:
    """
    Amostrador via decord — mais eficiente para random access em vídeos grandes.
    Requer o pacote 'decord'. Retorna frames em BGR (convertido de RGB).
    """
    import cv2
    from decord import VideoReader, cpu  # type: ignore[import]

    # decord não suporta UNC nativo; usa str com barras corretas
    path_str = str(video_path)
    vr = VideoReader(path_str, ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 1.0

    # Clipa índices para não ultrapassar o tamanho do vídeo
    valid_nums = [min(n, len(vr) - 1) for n in frame_numbers]
    frames_rgb = vr.get_batch(valid_nums).asnumpy()  # (N, H, W, 3) RGB

    results: list[SampledFrame] = []
    for idx, (frame_num, frame_rgb) in enumerate(zip(valid_nums, frames_rgb)):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results.append(SampledFrame(
            index=idx,
            timestamp_sec=frame_num / fps,
            frame_number=frame_num,
            image_bgr=frame_bgr,
        ))
    return results


def _compute_sample_frame_numbers(
    total_frames: int,
    fps: float,
    count: int,
    min_distance_sec: float = 2.0,
) -> list[int]:
    """
    Calcula os números de frames a amostrar, distribuídos uniformemente.
    Garante distância mínima entre amostras para evitar frames redundantes.
    """
    if count <= 0 or total_frames <= 0:
        return []

    min_distance_frames = max(1, int(min_distance_sec * fps))
    # Distribuição uniforme ao longo do vídeo (evita primeiro e último frame)
    step = total_frames / (count + 1)
    candidates = [int(step * (i + 1)) for i in range(count)]

    # Garante distância mínima entre candidatos
    filtered: list[int] = []
    last_selected = -min_distance_frames
    for frame_num in candidates:
        if frame_num - last_selected >= min_distance_frames:
            filtered.append(min(frame_num, total_frames - 1))
            last_selected = frame_num

    logger.debug(
        "Frames a amostrar (%d de %d candidatos): %s",
        len(filtered),
        len(candidates),
        filtered,
    )
    return filtered


def sample_frames(
    video_path: Path,
    count: int,
    min_distance_sec: float = 2.0,
    prefer_decord: bool = False,
) -> list[SampledFrame]:
    """
    Amostra `count` frames distribuidos uniformemente no video.

    Args:
        video_path: Caminho do video (local ou UNC).
        count: Numero de frames a amostrar (candidatos para selecao).
        min_distance_sec: Distancia minima em segundos entre amostras.
        prefer_decord: Tenta usar decord antes do OpenCV.
            IMPORTANTE: decord inicializa contexto CUDA proprio que conflita
            com PyTorch/pyiqa, causando crash nativo irrecuperavel.
            Manter False (padrao) quando BRISQUE/pyiqa estiver ativo.

    Returns:
        Lista de SampledFrame com as imagens amostradas.
    """
    import cv2

    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path.name}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    finally:
        cap.release()

    # Amostrar no máximo total_frames
    effective_count = min(count, total_frames)
    frame_numbers = _compute_sample_frame_numbers(
        total_frames=total_frames,
        fps=fps,
        count=effective_count,
        min_distance_sec=min_distance_sec,
    )

    if not frame_numbers:
        logger.warning("Nenhum frame a amostrar em %s", video_path.name)
        return []

    # Tenta decord primeiro se solicitado e disponível
    if prefer_decord:
        try:
            sampled = _sample_with_decord(video_path, frame_numbers)
            logger.debug(
                "Backend decord usado para %s (%d frames)",
                video_path.name,
                len(sampled),
            )
            return sampled
        except Exception as exc:
            logger.debug("decord indisponível (%s), usando OpenCV.", exc)

    # Fallback para OpenCV
    sampled = _sample_with_opencv(video_path, frame_numbers)
    logger.debug(
        "Backend OpenCV usado para %s (%d frames)",
        video_path.name,
        len(sampled),
    )
    return sampled
