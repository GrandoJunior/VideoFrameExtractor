# -*- coding: utf-8 -*-
"""
tests/test_quality_scorer.py — Testes do módulo de scoring de qualidade.

Verifica que imagens nítidas têm score maior que imagens borradas.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _create_sharp_frame(width=640, height=480) -> np.ndarray:
    """Cria frame nítido com padrão de tabuleiro de xadrez (bordas definidas)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    tile = 20
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if (x // tile + y // tile) % 2 == 0:
                img[y:y+tile, x:x+tile] = [255, 255, 255]
    return img


def _create_blurry_frame(sharp_frame: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Aplica forte blur gaussiano para simular frame borrado."""
    return cv2.GaussianBlur(sharp_frame, (0, 0), sigma)


def _create_dark_frame(width=640, height=480) -> np.ndarray:
    """Cria frame muito escuro (iluminação ruim)."""
    return np.ones((height, width, 3), dtype=np.uint8) * 10


def _create_overexposed_frame(width=640, height=480) -> np.ndarray:
    """Cria frame superexposto (iluminação ruim)."""
    return np.ones((height, width, 3), dtype=np.uint8) * 245


class TestLaplacianScore(unittest.TestCase):
    def test_sharp_scores_higher_than_blurry(self):
        from core.quality_scorer import _score_laplacian
        sharp = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        blurry = cv2.cvtColor(_create_blurry_frame(_create_sharp_frame()), cv2.COLOR_BGR2GRAY)
        self.assertGreater(_score_laplacian(sharp), _score_laplacian(blurry))

    def test_score_in_valid_range(self):
        from core.quality_scorer import _score_laplacian
        frame = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        score = _score_laplacian(frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


class TestTenengraduScore(unittest.TestCase):
    def test_sharp_scores_higher_than_blurry(self):
        from core.quality_scorer import _score_tenengrad
        sharp = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        blurry = cv2.cvtColor(_create_blurry_frame(_create_sharp_frame()), cv2.COLOR_BGR2GRAY)
        self.assertGreater(_score_tenengrad(sharp), _score_tenengrad(blurry))

    def test_score_in_valid_range(self):
        from core.quality_scorer import _score_tenengrad
        frame = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        score = _score_tenengrad(frame)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


class TestIlluminationScore(unittest.TestCase):
    def test_dark_frame_scores_lower_than_normal(self):
        from core.quality_scorer import _score_illumination
        normal = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        dark = cv2.cvtColor(_create_dark_frame(), cv2.COLOR_BGR2GRAY)
        self.assertGreater(_score_illumination(normal), _score_illumination(dark))

    def test_overexposed_frame_scores_lower_than_normal(self):
        from core.quality_scorer import _score_illumination
        normal = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        bright = cv2.cvtColor(_create_overexposed_frame(), cv2.COLOR_BGR2GRAY)
        self.assertGreater(_score_illumination(normal), _score_illumination(bright))

    def test_score_in_valid_range(self):
        from core.quality_scorer import _score_illumination
        gray = cv2.cvtColor(_create_sharp_frame(), cv2.COLOR_BGR2GRAY)
        score = _score_illumination(gray)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)


class TestCompositeScore(unittest.TestCase):
    def test_sharp_frame_composite_higher_than_blurry(self):
        from core.quality_scorer import score_frame
        sharp = _create_sharp_frame()
        blurry = _create_blurry_frame(sharp)
        score_sharp = score_frame(sharp, 0)
        score_blurry = score_frame(blurry, 1)
        self.assertGreater(score_sharp.composite, score_blurry.composite)

    def test_composite_in_valid_range(self):
        from core.quality_scorer import score_frame
        frame = _create_sharp_frame()
        result = score_frame(frame, 0)
        self.assertGreaterEqual(result.composite, 0.0)
        self.assertLessEqual(result.composite, 100.0)

    def test_composite_uses_available_metrics(self):
        from core.quality_scorer import score_frame
        frame = _create_sharp_frame()
        result = score_frame(frame, 0)
        # Mesmo sem BRISQUE disponível, deve retornar um score válido
        self.assertIsInstance(result.composite, float)
        self.assertGreater(result.composite, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
