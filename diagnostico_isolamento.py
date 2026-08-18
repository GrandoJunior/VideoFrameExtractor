# -*- coding: utf-8 -*-
"""
Teste de isolamento: verifica se o crash no BRISQUE e causado pelo decord.
Testa BRISQUE com frame lido via OpenCV (sem decord).
"""
import sys
import os
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configura logging identico ao main.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('isolamento')

video_path = PROJECT_ROOT / "teste" / "VID-20260609-WA0041.mp4"

print("\n=== TESTE A: BRISQUE com frame OpenCV (sem decord) ===")
import cv2
import numpy as np

cap = cv2.VideoCapture(str(video_path))
frames_cv = []
for fn in [111, 223, 334]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if ret:
        frames_cv.append(frame)
cap.release()
print(f"  Frames lidos via OpenCV: {len(frames_cv)}, shape: {frames_cv[0].shape}")

from core.quality_scorer import score_frame
print("  Pontuando frame 0 com BRISQUE...")
try:
    sc = score_frame(frames_cv[0], 0)
    print(f"  SUCESSO: composite={sc.composite:.1f} brisque_source={sc.brisque_source}")
except Exception as e:
    print(f"  FALHA (Exception): {e}")
except BaseException as e:
    print(f"  FALHA (BaseException/{type(e).__name__}): {e}")

print("\n=== TESTE B: BRISQUE com frame decord ===")
try:
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frames_dec = vr.get_batch([111, 223, 334]).asnumpy()
    frame_bgr = cv2.cvtColor(frames_dec[0], cv2.COLOR_RGB2BGR)
    print(f"  Frame lido via decord: shape={frame_bgr.shape}")

    print("  Pontuando frame decord com BRISQUE...")
    try:
        sc2 = score_frame(frame_bgr, 0)
        print(f"  SUCESSO: composite={sc2.composite:.1f} brisque_source={sc2.brisque_source}")
    except Exception as e:
        print(f"  FALHA (Exception): {e}")
    except BaseException as e:
        print(f"  FALHA (BaseException/{type(e).__name__}): {e}")
except Exception as e:
    print(f"  FALHA ao carregar decord: {e}")

print("\n=== TESTE C: score_frames_batch com frames OpenCV ===")
from core.frame_sampler import SampledFrame
from core.quality_scorer import score_frames_batch

sampled = [
    SampledFrame(index=i, timestamp_sec=i*3.7, frame_number=fn, image_bgr=f)
    for i, (fn, f) in enumerate(zip([111, 223, 334], frames_cv))
]
print(f"  {len(sampled)} SampledFrames criados")
try:
    scores = score_frames_batch(sampled)
    print(f"  SUCESSO: {len(scores)} scores calculados")
    for s in scores:
        print(f"    frame#{s.frame_index}: composite={s.composite:.1f} src={s.brisque_source}")
except Exception as e:
    print(f"  FALHA (Exception): {e}")
except BaseException as e:
    print(f"  FALHA (BaseException/{type(e).__name__}): {e}")

print("\n=== TODOS OS TESTES CONCLUIDOS ===")
