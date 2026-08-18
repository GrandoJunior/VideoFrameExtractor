# -*- coding: utf-8 -*-
"""
core/frame_selector.py — Seleção dos melhores frames por score composto.

Responsabilidades:
- Recebe lista de SampledFrame e FrameQualityScore correspondentes.
- Seleciona os N melhores frames usando score composto.
- Garante diversidade temporal: penaliza frames muito próximos entre si.
- Usa algoritmo guloso com penalidade de proximidade para maximizar
  tanto qualidade quanto cobertura temporal do vídeo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.frame_sampler import SampledFrame
from core.quality_scorer import FrameQualityScore

logger = logging.getLogger(__name__)


@dataclass
class SelectedFrame:
    """Frame selecionado com seu score e metadados para uso posterior."""

    rank: int                          # Posição no ranking (1 = melhor)
    sampled: SampledFrame              # Dados originais do frame amostrado
    score: FrameQualityScore           # Score de qualidade calculado
    output_filename: str               # Nome sugerido para o arquivo de saída


def build_output_filename(
    video_stem: str,
    rank: int,
    timestamp_sec: float,
    composite_score: float,
) -> str:
    """
    Gera nome de arquivo de saída padronizado para o frame selecionado.
    Formato: <nome_video>_frame<rank>_<mm>m<ss>s_q<score>.jpg
    """
    total = int(timestamp_sec)
    minutes, seconds = divmod(total, 60)
    score_int = int(round(composite_score))
    return f"{video_stem}_frame{rank:02d}_{minutes:02d}m{seconds:02d}s_q{score_int:03d}.jpg"


def select_best_frames(
    sampled_frames: list[SampledFrame],
    scores: list[FrameQualityScore],
    target_count: int,
    min_distance_sec: float,
    video_stem: str,
) -> list[SelectedFrame]:
    """
    Seleciona os N melhores frames com diversidade temporal garantida.

    Algoritmo guloso:
    1. Ordena candidatos por score composto (decrescente).
    2. Itera pelos candidatos em ordem de qualidade.
    3. Aceita o candidato se sua distância temporal para todos os
       já selecionados for >= min_distance_sec.
    4. Para quando atingir target_count selecionados.

    Args:
        sampled_frames: Lista de SampledFrame (mesma ordem que scores).
        scores: Lista de FrameQualityScore correspondentes.
        target_count: Número de frames a selecionar.
        min_distance_sec: Distância mínima em segundos entre selecionados.
        video_stem: Nome do vídeo (sem extensão) para nomear os arquivos.

    Returns:
        Lista de SelectedFrame, ordenada por timestamp (ordem cronológica).
    """
    if len(sampled_frames) != len(scores):
        raise ValueError(
            f"Número de frames ({len(sampled_frames)}) diferente do número "
            f"de scores ({len(scores)})."
        )

    if not sampled_frames:
        return []

    # Combina frames e scores em pares para ordenação conjunta
    paired = list(zip(sampled_frames, scores))

    # Ordena por score composto decrescente
    ranked = sorted(paired, key=lambda p: p[1].composite, reverse=True)

    selected: list[tuple[SampledFrame, FrameQualityScore]] = []
    selected_timestamps: list[float] = []

    for sf, qs in ranked:
        if len(selected) >= target_count:
            break

        ts = sf.timestamp_sec

        # Verifica distância mínima para todos os já selecionados
        too_close = any(
            abs(ts - existing_ts) < min_distance_sec
            for existing_ts in selected_timestamps
        )

        if not too_close:
            selected.append((sf, qs))
            selected_timestamps.append(ts)

    # Se não conseguiu target_count por restrição de distância,
    # pega os melhores restantes sem restrição (fallback).
    # Usa id() para comparar SampledFrame: evita ValueError ao comparar np.ndarray.
    if len(selected) < target_count and len(selected) < len(ranked):
        selected_ids = {id(sf) for sf, _ in selected}
        for sf, qs in ranked:
            if len(selected) >= target_count:
                break
            if id(sf) not in selected_ids:
                selected.append((sf, qs))
                selected_ids.add(id(sf))

    # Reordena os selecionados por timestamp (ordem cronológica do vídeo)
    selected_chronological = sorted(selected, key=lambda p: p[0].timestamp_sec)

    result: list[SelectedFrame] = []
    for rank, (sf, qs) in enumerate(selected_chronological, start=1):
        filename = build_output_filename(
            video_stem=video_stem,
            rank=rank,
            timestamp_sec=sf.timestamp_sec,
            composite_score=qs.composite,
        )
        result.append(SelectedFrame(
            rank=rank,
            sampled=sf,
            score=qs,
            output_filename=filename,
        ))
        logger.debug(
            "Selecionado rank #%d: timestamp=%.1fs | score=%.1f | arquivo=%s",
            rank,
            sf.timestamp_sec,
            qs.composite,
            filename,
        )

    logger.info(
        "Seleção concluída: %d frames selecionados de %d candidatos para '%s'.",
        len(result),
        len(sampled_frames),
        video_stem,
    )
    return result
