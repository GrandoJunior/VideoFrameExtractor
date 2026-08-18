# -*- coding: utf-8 -*-
"""
core/quality_scorer.py — Scoring multicriterio de qualidade de frames.

Pipeline de avaliação (em ordem de execução):
  1. Nitidez via variância do Laplaciano (OpenCV) — peso 35%
  2. Qualidade perceptual BRISQUE via pyiqa (GPU→CPU fallback) — peso 30%
  3. Qualidade de iluminação via histograma (OpenCV/NumPy) — peso 20%
  4. Ausência de motion blur via Tenengrad/Sobel (OpenCV) — peso 15%

Se pyiqa não estiver disponível, o peso do BRISQUE é redistribuído
proporcionalmente entre os demais critérios.

Fallback para análise via Groq Cloud se pyiqa estiver totalmente indisponível
e for necessário um score mais preciso em modo de qualidade alta.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import QUALITY_WEIGHTS, resolve_groq_api_key

logger = logging.getLogger(__name__)

# Cache de dispositivo PyTorch para não redetectar a cada frame
_TORCH_DEVICE: Optional[str] = None
_PYIQA_METRIC: Optional[object] = None
_PYIQA_AVAILABLE: Optional[bool] = None


@contextlib.contextmanager
def _suppress_native_stderr():
    """
    Suprime saídas nativas em stderr (fd=2) de bibliotecas C/nativas como basicsr.
    Usa dup2 no file descriptor 2 diretamente, independente do objeto sys.stderr.
    Abordagem segura mesmo com logging ativo: preserva sys.stderr do Python intacto.
    """
    devnull_fd = None
    saved_fd = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd = os.dup(2)          # Salva stderr nativo (fd 2)
        os.dup2(devnull_fd, 2)        # Redireciona fd 2 para /dev/null
    except OSError:
        # Se não for possível manipular fds (e.g. em subprocessos especiais), ignora
        if devnull_fd is not None:
            os.close(devnull_fd)
        devnull_fd = None
        saved_fd = None

    try:
        yield
    finally:
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 2)  # Restaura stderr nativo
                os.close(saved_fd)
            except OSError:
                pass
        if devnull_fd is not None:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


def _get_torch_device() -> str:
    """Detecta e cacheia o dispositivo PyTorch disponível (cuda > cpu)."""
    global _TORCH_DEVICE
    if _TORCH_DEVICE is not None:
        return _TORCH_DEVICE
    try:
        import torch
        _TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("PyTorch device: %s", _TORCH_DEVICE)
    except ImportError:
        _TORCH_DEVICE = 'cpu'
        logger.debug("torch não disponível, usando cpu.")
    return _TORCH_DEVICE


def _get_pyiqa_metric() -> Optional[object]:
    """
    Inicializa e cacheia a métrica BRISQUE do pyiqa (GPU se disponível).
    Suprime o stderr nativo de basicsr/pyiqa durante a inicialização para
    evitar NativeCommandError no PowerShell.
    """
    global _PYIQA_METRIC, _PYIQA_AVAILABLE
    if _PYIQA_AVAILABLE is not None:
        return _PYIQA_METRIC

    # Silencia loggers do Python que basicsr/pyiqa usam para imprimir
    for noisy in ('basicsr', 'basicsr.utils.logger', 'pyiqa', 'pyiqa.metrics'):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)

    # Desativa tqdm temporariamente para evitar WinError 1 no console redirecionado no Windows
    old_tqdm_disable = os.environ.get('TQDM_DISABLE')
    os.environ['TQDM_DISABLE'] = '1'

    try:
        import pyiqa
        device = _get_torch_device()
        try:
            # Tenta criar com supressão de stderr nativo
            with _suppress_native_stderr():
                _PYIQA_METRIC = pyiqa.create_metric('brisque', device=device)
        except SystemExit as sys_exc:
            logger.debug(
                "basicsr disparou sys.exit() ao carregar pyiqa com supressao (%s). Tentando sem supressao.", sys_exc
            )
            _PYIQA_METRIC = pyiqa.create_metric('brisque', device=device)
        except Exception as inner_exc:
            logger.debug(
                "Falha ao carregar pyiqa com supressao de stderr (%s: %s). Tentando sem supressao.",
                type(inner_exc).__name__, inner_exc
            )
            # Fallback: carrega sem suprimir o stderr nativo
            _PYIQA_METRIC = pyiqa.create_metric('brisque', device=device)
            
        _PYIQA_AVAILABLE = True
        logger.info("pyiqa BRISQUE metric carregada no dispositivo '%s'.", device)
    except SystemExit as exc:
        _PYIQA_AVAILABLE = False
        _PYIQA_METRIC = None
        logger.warning("basicsr causou SystemExit ao carregar pyiqa: %s. BRISQUE desabilitado.", exc)
    except Exception as exc:
        _PYIQA_AVAILABLE = False
        _PYIQA_METRIC = None
        logger.warning("pyiqa indisponivel (%s: %s). BRISQUE desabilitado.",
                       type(exc).__name__, exc)
    finally:
        # Restaura a variável de ambiente TQDM_DISABLE
        if old_tqdm_disable is not None:
            os.environ['TQDM_DISABLE'] = old_tqdm_disable
        else:
            os.environ.pop('TQDM_DISABLE', None)

    return _PYIQA_METRIC


# ── Funções de pontuação individuais ─────────────────────────────────────────

def _score_laplacian(gray: np.ndarray) -> float:
    """
    Nitidez via variância do Laplaciano.
    Escala: quanto maior, mais nítido. Normalizado para [0, 100].
    Referência: > 500 = muito nítido, < 50 = borrado.
    """
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalização: log-scale para comprimir a cauda longa
    # log(500+1) ≈ 6.215 → mapeado para 100
    normalized = min(100.0, math.log1p(variance) / math.log1p(500) * 100.0)
    return normalized


def _score_tenengrad(gray: np.ndarray) -> float:
    """
    Ausência de motion blur via gradiente Sobel (Tenengrad).
    Escala: quanto maior, mais nítido.
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sobel_x ** 2 + sobel_y ** 2)
    # Normalização similar ao Laplaciano
    normalized = min(100.0, math.log1p(tenengrad) / math.log1p(10000) * 100.0)
    return normalized


def _score_illumination(gray: np.ndarray) -> float:
    """
    Qualidade de iluminação via análise de histograma.
    Penaliza frames muito escuros, muito claros ou com baixo contraste.
    Score = combinação de: brilho médio centralizado + desvio padrão.
    """
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    # Brilho ideal: próximo de 128 (centro). Penalidade quadrática pela distância.
    brightness_score = 100.0 - ((mean_brightness - 128.0) / 128.0) ** 2 * 100.0

    # Contraste: desvio padrão ideal entre 40 e 80. Abaixo de 20 é sem contraste.
    if std_brightness < 20:
        contrast_score = (std_brightness / 20.0) * 60.0
    elif std_brightness > 100:
        contrast_score = max(0.0, 100.0 - (std_brightness - 100.0))
    else:
        contrast_score = 60.0 + min(40.0, (std_brightness - 20.0) / 80.0 * 40.0)

    return (brightness_score * 0.5 + contrast_score * 0.5)


def _score_brisque_pyiqa(bgr_frame: np.ndarray) -> Optional[float]:
    """
    Score BRISQUE via pyiqa.
    BRISQUE original: menor = melhor (0-100, onde 0 = perfeito).
    Invertemos para [0-100] onde 100 = melhor.
    Toda a operacao e envolta em _suppress_native_stderr() pois bibliotecas
    nativas (basicsr/torch) podem escrever para stderr durante a inferencia.
    """
    metric = _get_pyiqa_metric()
    if metric is None:
        return None

    try:
        import torch
        from torchvision import transforms

        # Converte BGR -> RGB -> tensor float32 normalizado [0,1]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        tensor = transforms.ToTensor()(rgb).unsqueeze(0)  # (1, C, H, W)

        device = _get_torch_device()
        tensor = tensor.to(device)

        with _suppress_native_stderr():
            with torch.no_grad():
                raw_score = float(metric(tensor).item())

        # BRISQUE: 0 = perfeito, ~100 = pessimo. Inverte para nosso padrao.
        inverted = max(0.0, min(100.0, 100.0 - raw_score))
        return inverted
    except SystemExit as exc:
        logger.warning("basicsr/pyiqa disparou SystemExit no frame: %s", exc)
        return None
    except Exception as exc:
        logger.debug("Falha no score BRISQUE (%s: %s)", type(exc).__name__, exc)
        return None


def _score_brisque_groq(bgr_frame: np.ndarray) -> Optional[float]:
    """
    Fallback Groq Cloud: análise de qualidade de imagem via LLM Vision.
    Retorna estimativa de score [0–100] ou None se falhar.
    Usado APENAS quando pyiqa está completamente indisponível.
    """
    groq_key = resolve_groq_api_key()
    if not groq_key:
        return None

    try:
        import base64
        import json
        from groq import Groq

        _, buffer = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64_image = base64.b64encode(buffer).decode('utf-8')

        client = Groq(api_key=groq_key)

        from config import GROQ_MODEL_VISION, GROQ_TIMEOUT_SEC

        response = client.chat.completions.create(
            model=GROQ_MODEL_VISION,
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            'Avalie a qualidade técnica desta imagem em uma escala de 0 a 100. '
                            'Considere: nitidez, ausência de borrões, iluminação equilibrada, '
                            'contraste adequado e ausência de ruído. '
                            'Responda APENAS com um JSON: {"score": <número inteiro 0-100>}'
                        ),
                    },
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64_image}'},
                    },
                ],
            }],
            max_tokens=50,
            timeout=GROQ_TIMEOUT_SEC,
        )

        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        score = float(data.get('score', 50))
        return max(0.0, min(100.0, score))

    except Exception as exc:
        logger.debug("Fallback Groq BRISQUE falhou: %s", exc)
        return None


# ── Score composto ────────────────────────────────────────────────────────────

@dataclass
class FrameQualityScore:
    """Resultado completo do scoring de um frame."""

    frame_index: int
    laplacian: float       # [0–100]
    tenengrad: float       # [0–100]
    illumination: float    # [0–100]
    brisque: Optional[float]  # [0–100] ou None se indisponível
    composite: float       # [0–100] score final ponderado
    brisque_source: str = field(default='none')  # 'pyiqa' | 'groq' | 'none'


def score_frame(
    bgr_frame: np.ndarray,
    frame_index: int,
    allow_groq: bool = False,
) -> FrameQualityScore:
    """
    Calcula o score de qualidade composto de um frame.

    Args:
        bgr_frame: Frame em formato BGR (NumPy array uint8).
        frame_index: Índice do frame na sequência amostrada.
        allow_groq: Se permite o uso de fallback para Groq Cloud se pyiqa estiver indisponível.

    Returns:
        FrameQualityScore com todos os scores individuais e composto.
    """
    # Converte para cinza uma vez para reutilização
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    lap = _score_laplacian(gray)
    ten = _score_tenengrad(gray)
    ill = _score_illumination(gray)

    # Tenta BRISQUE: pyiqa (GPU/CPU) → Groq Cloud → None
    brisque_val: Optional[float] = None
    brisque_source = 'none'

    if _PYIQA_AVAILABLE is not False:  # Ainda não tentou ou disponível
        brisque_val = _score_brisque_pyiqa(bgr_frame)
        if brisque_val is not None:
            brisque_source = 'pyiqa'

    if brisque_val is None and allow_groq:
        logger.warning(
            "Acionando fallback em nuvem (Groq Cloud API). Enviando frame #%d para analise externa.",
            frame_index
        )
        brisque_val = _score_brisque_groq(bgr_frame)
        if brisque_val is not None:
            brisque_source = 'groq'

    # Cálculo do score composto com redistribuição de peso se BRISQUE indisponível
    weights = dict(QUALITY_WEIGHTS)
    if brisque_val is None:
        # Redistribui o peso do BRISQUE proporcionalmente
        brisque_weight = weights.pop('brisque')
        total_remaining = sum(weights.values())
        for key in weights:
            weights[key] += (weights[key] / total_remaining) * brisque_weight

    scores = {
        'laplacian':    lap,
        'tenengrad':    ten,
        'illumination': ill,
    }
    if brisque_val is not None:
        scores['brisque'] = brisque_val

    composite = sum(scores.get(k, 0.0) * w for k, w in weights.items())
    composite = max(0.0, min(100.0, composite))

    return FrameQualityScore(
        frame_index=frame_index,
        laplacian=lap,
        tenengrad=ten,
        illumination=ill,
        brisque=brisque_val,
        composite=composite,
        brisque_source=brisque_source,
    )


def score_frames_batch(
    frames: list,  # list[SampledFrame]
    allow_groq: bool = False,
) -> list[FrameQualityScore]:
    """
    Pontua uma lista de frames e retorna os scores em ordem.
    Utiliza processamento em lote (batching) no PyTorch se pyiqa estiver disponível.

    Args:
        frames: Lista de SampledFrame (de frame_sampler.py).
        allow_groq: Se permite o uso de fallback para Groq Cloud se pyiqa estiver indisponível.

    Returns:
        Lista de FrameQualityScore na mesma ordem dos frames de entrada.
    """
    global _PYIQA_AVAILABLE
    if not frames:
        return []

    # Verifica se pyiqa está disponível
    metric_avail = _PYIQA_AVAILABLE if _PYIQA_AVAILABLE is not None else (_get_pyiqa_metric() is not None)
    metric = _get_pyiqa_metric()

    effective_allow_groq = allow_groq
    if not metric_avail and len(frames) > 5 and allow_groq:
        effective_allow_groq = False
        logger.warning(
            "pyiqa indisponivel e lote de frames grande (%d frames). "
            "Groq fallback desativado para evitar lentidao extrema e estouro de limites de requisicao (HTTP 429). "
            "Usando redistribuicao proporcional de pesos local.",
            len(frames)
        )

    # Inicializa scores BRISQUE com None
    brisque_scores: list[Optional[float]] = [None] * len(frames)
    brisque_sources: list[str] = ['none'] * len(frames)

    # Se a métrica local pyiqa/BRISQUE estiver disponível, calcula em lote (batch)
    if metric_avail and metric is not None:
        try:
            import torch
            from torchvision import transforms

            tensors = []
            for sf in frames:
                # Converte BGR -> RGB
                rgb = cv2.cvtColor(sf.image_bgr, cv2.COLOR_BGR2RGB)
                # Converte para Tensor e adiciona à lista
                tensors.append(transforms.ToTensor()(rgb))
            
            # Stack de tensores para criar um lote (N, C, H, W)
            batch_tensor = torch.stack(tensors)
            device = _get_torch_device()
            batch_tensor = batch_tensor.to(device)

            with _suppress_native_stderr():
                with torch.no_grad():
                    raw_scores = metric(batch_tensor)
                    if raw_scores.ndim == 0:
                        raw_scores_list = [raw_scores.item()]
                    else:
                        raw_scores_list = raw_scores.cpu().view(-1).tolist()

            # BRISQUE: 0 = perfeito, ~100 = péssimo. Inverte para padrão do projeto
            for idx, raw_val in enumerate(raw_scores_list):
                inverted = max(0.0, min(100.0, 100.0 - raw_val))
                brisque_scores[idx] = inverted
                brisque_sources[idx] = 'pyiqa'
            logger.info("Scoring BRISQUE concluido em lote (GPU/CPU batch) para %d frames.", len(frames))

        except SystemExit as sys_exc:
            logger.warning("basicsr/pyiqa disparou SystemExit no lote: %s. Desabilitando pyiqa.", sys_exc)
            _PYIQA_AVAILABLE = False
        except Exception as exc:
            logger.warning("Falha ao processar lote no BRISQUE (%s: %s). Revertendo para processamento individual.",
                           type(exc).__name__, exc)

    # Processa os demais critérios e o fallback individual para Groq se necessário
    results: list[FrameQualityScore] = []
    for idx, sf in enumerate(frames):
        # Converte para cinza uma vez para reutilização
        gray = cv2.cvtColor(sf.image_bgr, cv2.COLOR_BGR2GRAY)

        lap = _score_laplacian(gray)
        ten = _score_tenengrad(gray)
        ill = _score_illumination(gray)

        # Se o BRISQUE em lote falhou ou não foi executado para este frame, tenta fallback individual
        b_val = brisque_scores[idx]
        b_src = brisque_sources[idx]

        if b_val is None:
            # 1. Tenta pyiqa individual (caso o erro no lote tenha sido pontual)
            if metric_avail:
                b_val = _score_brisque_pyiqa(sf.image_bgr)
                if b_val is not None:
                    b_src = 'pyiqa'

            # 2. Tenta Groq se permitido
            if b_val is None and effective_allow_groq:
                logger.warning(
                    "Acionando fallback em nuvem (Groq Cloud API). Enviando frame #%d para analise externa.",
                    sf.index
                )
                b_val = _score_brisque_groq(sf.image_bgr)
                if b_val is not None:
                    b_src = 'groq'

        # Cálculo do score composto com redistribuição de peso se BRISQUE indisponível
        weights = dict(QUALITY_WEIGHTS)
        if b_val is None:
            # Redistribui o peso do BRISQUE proporcionalmente
            brisque_weight = weights.pop('brisque')
            total_remaining = sum(weights.values())
            for key in weights:
                weights[key] += (weights[key] / total_remaining) * brisque_weight

        scores = {
            'laplacian':    lap,
            'tenengrad':    ten,
            'illumination': ill,
        }
        if b_val is not None:
            scores['brisque'] = b_val

        composite = sum(scores.get(k, 0.0) * w for k, w in weights.items())
        composite = max(0.0, min(100.0, composite))

        score_obj = FrameQualityScore(
            frame_index=sf.index,
            laplacian=lap,
            tenengrad=ten,
            illumination=ill,
            brisque=b_val,
            composite=composite,
            brisque_source=b_src,
        )

        logger.debug(
            "Frame #%d | lap=%.1f ten=%.1f ill=%.1f brisque=%s composite=%.1f | src=%s",
            sf.index,
            score_obj.laplacian,
            score_obj.tenengrad,
            score_obj.illumination,
            f"{score_obj.brisque:.1f}" if score_obj.brisque is not None else "N/A",
            score_obj.composite,
            score_obj.brisque_source,
        )
        results.append(score_obj)

    return results
