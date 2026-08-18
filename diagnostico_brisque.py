# -*- coding: utf-8 -*-
"""
Diagnóstico isolado: testa BRISQUE com frame real do vídeo de teste.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Python:", sys.version)
print("Executável:", sys.executable)

import cv2
print("OpenCV OK:", cv2.__version__)

import numpy as np
print("NumPy OK:", np.__version__)

# Abre o vídeo de teste e extrai um frame real
video_path = PROJECT_ROOT / "teste" / "VID-20260609-WA0041.mp4"
print(f"\nAbrindo vídeo: {video_path}")
cap = cv2.VideoCapture(str(video_path))
print("Aberto:", cap.isOpened())
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()
print(f"Frame lido: {ret}, shape: {frame.shape if ret else 'N/A'}")

if not ret:
    print("FALHA: não foi possível ler frame")
    sys.exit(1)

print("\nTestando PyTorch...")
import torch
print("Torch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM total:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
    print("VRAM livre:", round(torch.cuda.memory_reserved(0) / 1e9, 3), "GB reservada")

print("\nConvertendo frame para tensor...")
from torchvision import transforms
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
tensor = transforms.ToTensor()(rgb).unsqueeze(0)
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, min: {tensor.min():.3f}, max: {tensor.max():.3f}")

print("\nCarregando BRISQUE...")
import pyiqa

# Redireciona stderr para /dev/null durante criação
devnull = open(os.devnull, 'w')
old_stderr = os.dup(2)
os.dup2(devnull.fileno(), 2)
metric = pyiqa.create_metric('brisque', device='cuda')
os.dup2(old_stderr, 2)
os.close(old_stderr)
devnull.close()
print("BRISQUE carregado OK")

print("\nMovendo tensor para CUDA...")
tensor_cuda = tensor.cuda()
print(f"Tensor na CUDA: {tensor_cuda.device}")

print("\nExecutando inferência BRISQUE...")
try:
    with torch.no_grad():
        result = metric(tensor_cuda)
        print(f"Score BRISQUE: {result.item():.4f}")
    print("\nSUCESSO: Pipeline completa funcionando no venv!")
except Exception as e:
    import traceback
    print(f"\nFALHA na inferência: {e}")
    traceback.print_exc()
    sys.exit(1)
