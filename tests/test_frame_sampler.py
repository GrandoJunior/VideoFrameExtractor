# -*- coding: utf-8 -*-
"""
tests/test_frame_sampler.py — Testes unitários do módulo frame_sampler.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestComputeSampleFrameNumbers(unittest.TestCase):
    """Testa a função de cálculo de números de frames a amostrar."""

    def _call(self, total, fps, count, min_dist=2.0):
        from core.frame_sampler import _compute_sample_frame_numbers
        return _compute_sample_frame_numbers(total, fps, count, min_dist)

    def test_zero_frames_returns_empty(self):
        self.assertEqual(self._call(0, 30.0, 5), [])

    def test_negative_count_returns_empty(self):
        self.assertEqual(self._call(300, 30.0, 0), [])

    def test_basic_distribution(self):
        result = self._call(300, 30.0, 5, min_dist=0.0)
        self.assertEqual(len(result), 5)
        # Todos dentro do intervalo válido
        for num in result:
            self.assertGreaterEqual(num, 0)
            self.assertLess(num, 300)

    def test_min_distance_enforced(self):
        """Com distância mínima de 10s e FPS=30, frames devem estar > 300 frames apart."""
        result = self._call(3000, 30.0, 10, min_dist=10.0)
        if len(result) >= 2:
            for i in range(1, len(result)):
                diff = result[i] - result[i - 1]
                # 10s * 30fps = 300 frames mínimo
                self.assertGreaterEqual(diff, 300)

    def test_count_exceeds_total_frames_clamped(self):
        result = self._call(10, 30.0, 100, min_dist=0.0)
        # Não pode gerar mais frames do que o total
        self.assertLessEqual(len(result), 10)

    def test_frame_numbers_sorted(self):
        result = self._call(600, 30.0, 8, min_dist=1.0)
        self.assertEqual(result, sorted(result))


class TestSampleFrames(unittest.TestCase):
    """Testa a função principal sample_frames com mock do OpenCV."""

    def _make_mock_cap(self, total_frames=90, fps=30.0):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: fps,            # CAP_PROP_FPS
            7: total_frames,   # CAP_PROP_FRAME_COUNT
        }.get(prop, 0.0)
        # Simula frame lido com sucesso
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        return mock_cap

    @patch('cv2.VideoCapture')
    def test_samples_correct_count(self, mock_cap_cls):
        mock_cap_cls.return_value = self._make_mock_cap(total_frames=300, fps=30.0)
        from core.frame_sampler import sample_frames
        result = sample_frames(Path('fake.mp4'), count=5, prefer_decord=False)
        self.assertEqual(len(result), 5)

    @patch('cv2.VideoCapture')
    def test_sampled_frame_has_image(self, mock_cap_cls):
        mock_cap_cls.return_value = self._make_mock_cap()
        from core.frame_sampler import sample_frames
        result = sample_frames(Path('fake.mp4'), count=3, prefer_decord=False)
        for sf in result:
            self.assertIsInstance(sf.image_bgr, np.ndarray)
            self.assertEqual(sf.image_bgr.shape[2], 3)  # 3 canais BGR

    @patch('cv2.VideoCapture')
    def test_timestamps_positive(self, mock_cap_cls):
        mock_cap_cls.return_value = self._make_mock_cap(total_frames=300, fps=30.0)
        from core.frame_sampler import sample_frames
        result = sample_frames(Path('fake.mp4'), count=5, prefer_decord=False)
        for sf in result:
            self.assertGreaterEqual(sf.timestamp_sec, 0.0)

    @patch('cv2.VideoCapture')
    def test_raises_if_cannot_open(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap
        from core.frame_sampler import sample_frames
        with self.assertRaises(RuntimeError):
            sample_frames(Path('nonexistent.mp4'), count=5, prefer_decord=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
