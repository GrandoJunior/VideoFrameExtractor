# -*- coding: utf-8 -*-
"""
core/video_scanner.py — Varredura de pasta por arquivos de vídeo.

Responsabilidades:
- Aceita caminhos locais e UNC (\\servidor\share) sem remapeamento de drive.
- Extrai metadados de cada vídeo (duração, FPS, resolução, tamanho).
- Retorna lista de VideoFile ordenada por nome.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFile:
    """Metadados de um arquivo de vídeo encontrado na varredura."""

    path: Path
    size_bytes: int
    duration_sec: float
    fps: float
    width: int
    height: int

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def duration_formatted(self) -> str:
        """Duração formatada como HH:MM:SS."""
        total = int(self.duration_sec)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def name(self) -> str:
        return self.path.name


def _extract_video_metadata(video_path: Path) -> dict:
    """
    Extrai metadados de um vídeo usando OpenCV.
    Retorna dicionário com duration_sec, fps, width, height.
    Lança RuntimeError se o arquivo não puder ser aberto.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python não está instalado.") from exc

    # Converter para string com barras corretas para OpenCV no Windows
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)

    if not cap.isOpened():
        raise RuntimeError(f"OpenCV não conseguiu abrir: {path_str}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = (frame_count / fps) if fps > 0 else 0.0
        return {
            'duration_sec': duration_sec,
            'fps': fps,
            'width': width,
            'height': height,
        }
    finally:
        cap.release()


def _iter_video_files(
    folder: Path,
    extensions: frozenset[str],
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Itera arquivos de vídeo na pasta.
    Suporte a UNC via pathlib.Path (sem remapeamento de drive).
    """
    try:
        iterator = folder.rglob('*') if recursive else folder.iterdir()
    except PermissionError as exc:
        logger.warning("Sem permissão para acessar %s: %s", folder, exc)
        return
    except OSError as exc:
        logger.error("Erro de SO ao acessar %s: %s", folder, exc)
        return

    for item in iterator:
        if item.is_file() and item.suffix.lower() in extensions:
            yield item


def scan_folder(
    folder: Path,
    extensions: frozenset[str],
    recursive: bool = False,
) -> list[VideoFile]:
    """
    Varre a pasta indicada e retorna lista de VideoFile com metadados completos.

    Args:
        folder: Caminho da pasta (local ou UNC).
        extensions: Conjunto de extensões válidas (ex: {'.mp4', '.mov'}).
        recursive: Se True, varre subpastas recursivamente.

    Returns:
        Lista de VideoFile ordenada por nome de arquivo.

    Raises:
        FileNotFoundError: Se a pasta não existir.
        NotADirectoryError: Se o caminho não for um diretório.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"O caminho não é um diretório: {folder}")

    logger.info("Varrendo vídeos em: %s (recursivo=%s)", folder, recursive)

    results: list[VideoFile] = []
    skipped = 0

    for video_path in sorted(_iter_video_files(folder, extensions, recursive)):
        try:
            size_bytes = video_path.stat().st_size
            meta = _extract_video_metadata(video_path)
            video_file = VideoFile(
                path=video_path,
                size_bytes=size_bytes,
                duration_sec=meta['duration_sec'],
                fps=meta['fps'],
                width=meta['width'],
                height=meta['height'],
            )
            results.append(video_file)
            logger.debug(
                "Encontrado: %s | %.1fMB | %s | %.1f fps",
                video_file.name,
                video_file.size_mb,
                video_file.duration_formatted,
                video_file.fps,
            )
        except RuntimeError as exc:
            logger.warning("Ignorando %s: %s", video_path.name, exc)
            skipped += 1
        except OSError as exc:
            logger.warning("Erro ao ler %s: %s", video_path.name, exc)
            skipped += 1

    logger.info(
        "Varredura concluída: %d vídeos encontrados, %d ignorados.",
        len(results),
        skipped,
    )
    return results
