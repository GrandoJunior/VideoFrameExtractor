# -*- coding: utf-8 -*-
"""
Replica exatamente o fluxo do run_e2e_test.py, mas com captura de BaseException
para identificar o crash.
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

from config import VIDEO_EXTENSIONS, MIN_FRAME_DISTANCE_SEC
from core.video_scanner import scan_folder
from core.frame_sampler import sample_frames
from core.quality_scorer import score_frames_batch, score_frame, _get_pyiqa_metric

test_folder = PROJECT_ROOT / 'teste'
videos = scan_folder(test_folder, VIDEO_EXTENSIONS)
print(f"Videos: {len(videos)}")

video = videos[0]
print(f"Amostrando {video.name}...")
sampled = sample_frames(video.path, count=9)
print(f"Amostrados: {len(sampled)} frames")

# Inicializa BRISQUE antes do loop
print("Inicializando BRISQUE...")
metric = _get_pyiqa_metric()
print(f"Metrica: {metric is not None}")

# Testa frame a frame com captura de BaseException
print("Pontuando frames um a um...")
for i, sf in enumerate(sampled):
    print(f"  Frame {i} (shape={sf.image_bgr.shape})...", end=' ', flush=True)
    try:
        sc = score_frame(sf.image_bgr, sf.index)
        print(f"OK composite={sc.composite:.1f}")
    except SystemExit as e:
        print(f"\nCRASH SystemExit({e.code}) no frame {i}")
        sys.exit(1)
    except BaseException as e:
        print(f"\nCRASH {type(e).__name__}: {e} no frame {i}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("SUCESSO: todos os frames pontuados!")
