# -*- coding: utf-8 -*-
"""
run_e2e_test.py — Teste end-to-end com o video da pasta teste/.
Uso: python run_e2e_test.py
Requer: opencv-python, numpy (do venv ou ambiente ativo)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Garante que o projeto esta no PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('e2e_test')


def run_test() -> None:
    test_folder = PROJECT_ROOT / 'teste'
    output_folder = PROJECT_ROOT / 'teste_output'

    print()
    print("=" * 60)
    print("  VideoFrameExtractor — Teste End-to-End")
    print(f"  Pasta de entrada : {test_folder}")
    print(f"  Pasta de saida   : {output_folder}")
    print("=" * 60)
    print()

    # ── Importacoes do projeto ────────────────────────────────────────────────
    print("[1/5] Importando modulos do projeto...")
    try:
        from config import VIDEO_EXTENSIONS, MIN_FRAME_DISTANCE_SEC
        from core.video_scanner import scan_folder
        from core.frame_sampler import sample_frames
        from core.quality_scorer import score_frames_batch
        from core.frame_selector import select_best_frames
        from core.image_enhancer import enhance_and_save
        print("      OK\n")
    except ImportError as exc:
        print(f"      FALHA ao importar modulos: {exc}")
        print("      Verifique se as dependencias estao instaladas no venv.")
        sys.exit(1)

    # ── Varredura de videos ───────────────────────────────────────────────────
    print(f"[2/5] Varrendo videos em: {test_folder}")
    if not test_folder.exists():
        print(f"      FALHA: pasta nao encontrada: {test_folder}")
        sys.exit(1)

    videos = scan_folder(test_folder, VIDEO_EXTENSIONS)
    if not videos:
        print("      FALHA: nenhum video encontrado na pasta teste/")
        sys.exit(1)

    for v in videos:
        print(f"      Encontrado: {v.name} | {v.size_mb:.1f}MB | {v.duration_formatted} | {v.resolution_label}")
    print()

    # ── Processamento de cada video ───────────────────────────────────────────
    FRAMES_POR_VIDEO = 3  # Quantia reduzida para teste rapido

    total_salvos = 0
    total_falhas = 0

    for video in videos:
        print(f"[3/5] Amostrando frames de: {video.name}")
        candidate_count = min(FRAMES_POR_VIDEO * 3, 30)
        sampled = sample_frames(video.path, count=candidate_count)
        print(f"      {len(sampled)} frames amostrados")

        if not sampled:
            print("      AVISO: nenhum frame amostrado. Video pode estar corrompido.")
            total_falhas += 1
            continue

        print(f"[4/5] Calculando scores de qualidade...")
        scores = score_frames_batch(sampled)
        for i, (sf, sc) in enumerate(zip(sampled, scores)):
            print(f"      Frame {i}: ts={sf.timestamp_sec:.1f}s | "
                  f"lap={sc.laplacian:.1f} | ten={sc.tenengrad:.1f} | "
                  f"ill={sc.illumination:.1f} | brisque={sc.brisque} | "
                  f"composite={sc.composite:.1f} | src={sc.brisque_source}")

        print(f"[5/5] Selecionando os {FRAMES_POR_VIDEO} melhores frames...")
        selected = select_best_frames(
            sampled_frames=sampled,
            scores=scores,
            target_count=FRAMES_POR_VIDEO,
            min_distance_sec=MIN_FRAME_DISTANCE_SEC,
            video_stem=video.path.stem,
        )
        print(f"      {len(selected)} frames selecionados")

        output_folder.mkdir(parents=True, exist_ok=True)
        for sel in selected:
            out_path = output_folder / sel.output_filename
            ok = enhance_and_save(sel.sampled.image_bgr, out_path)
            if ok:
                total_salvos += 1
                print(f"      Salvo: {sel.output_filename} (score={sel.score.composite:.1f})")
            else:
                total_falhas += 1
                print(f"      FALHA ao salvar frame #{sel.rank}")

    print()
    print("=" * 60)
    print(f"  RESULTADO: {total_salvos} frames salvos | {total_falhas} falhas")
    print(f"  Saida em : {output_folder}")
    print("=" * 60)
    print()

    if total_falhas > 0:
        sys.exit(1)


if __name__ == '__main__':
    run_test()
