# -*- coding: utf-8 -*-
"""
tests/test_image_enhancer.py — Testes do módulo de aprimoramento de imagem.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _create_test_frame(width=320, height=240) -> np.ndarray:
    """Cria frame de teste com gradiente de cor."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img[i, :] = [int(i * 255 / height)] * 3
    return img


class TestCpuEnhancement(unittest.TestCase):
    """Testa pipeline de aprimoramento CPU."""

    def test_clahe_returns_same_shape(self):
        from core.image_enhancer import _apply_clahe
        frame = _create_test_frame()
        result = _apply_clahe(frame)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_white_balance_returns_same_shape(self):
        from core.image_enhancer import _apply_auto_white_balance
        frame = _create_test_frame()
        result = _apply_auto_white_balance(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_unsharp_mask_returns_same_shape(self):
        from core.image_enhancer import _apply_unsharp_mask
        frame = _create_test_frame()
        result = _apply_unsharp_mask(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_nlmd_denoise_returns_same_shape(self):
        from core.image_enhancer import _apply_nlmd_denoise
        frame = _create_test_frame()
        result = _apply_nlmd_denoise(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_enhance_cpu_full_pipeline(self):
        """Pipeline completo CPU deve retornar imagem do mesmo tamanho."""
        from core.image_enhancer import enhance_cpu
        frame = _create_test_frame()
        result = enhance_cpu(frame)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_enhance_cpu_values_clamped(self):
        """Valores de pixel devem estar no intervalo [0, 255]."""
        from core.image_enhancer import enhance_cpu
        frame = _create_test_frame()
        result = enhance_cpu(frame)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)


class TestEnhanceAndSave(unittest.TestCase):
    """Testa salvamento de frames aprimorados."""

    def test_save_creates_jpeg_file(self):
        """Deve criar um arquivo JPEG válido no caminho indicado."""
        import tempfile, os
        from core.image_enhancer import enhance_and_save

        frame = _create_test_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'test_frame.jpg'
            ok = enhance_and_save(frame, output)
            self.assertTrue(ok)
            self.assertTrue(output.exists())
            # Verifica que é um JPEG válido
            loaded = cv2.imread(str(output))
            self.assertIsNotNone(loaded)

    def test_save_creates_parent_directory(self):
        """Deve criar o diretório pai se não existir."""
        import tempfile
        from core.image_enhancer import enhance_and_save

        frame = _create_test_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'subdir' / 'deep' / 'frame.jpg'
            ok = enhance_and_save(frame, output)
            self.assertTrue(ok)
            self.assertTrue(output.exists())

    @patch('core.image_enhancer.enhance_gpu_realesrgan')
    def test_falls_back_to_cpu_if_realesrgan_fails(self, mock_gpu):
        """Se Real-ESRGAN falhar, deve continuar com pipeline CPU."""
        import tempfile
        from core.image_enhancer import enhance_and_save
        import core.image_enhancer as enhancer

        mock_gpu.return_value = None  # Simula falha no GPU
        original_upscaling = enhancer.ENHANCEMENT_APPLY_UPSCALING
        enhancer.ENHANCEMENT_APPLY_UPSCALING = True

        frame = _create_test_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'frame.jpg'
            try:
                ok = enhance_and_save(frame, output)
                self.assertTrue(ok)
            finally:
                enhancer.ENHANCEMENT_APPLY_UPSCALING = original_upscaling


class TestRealesrganUnavailable(unittest.TestCase):
    """Testa comportamento quando Real-ESRGAN não está instalado."""

    def test_enhance_gpu_returns_none_without_realesrgan(self):
        """Sem Real-ESRGAN instalado, deve retornar None graciosamente."""
        from core.image_enhancer import enhance_gpu_realesrgan
        import core.image_enhancer as enhancer

        # Força re-tentativa de inicialização
        enhancer._REALESRGAN_INIT_ATTEMPTED = False
        enhancer._REALESRGAN_UPSAMPLER = None

        with patch.dict('sys.modules', {'realesrgan': None, 'basicsr': None}):
            frame = _create_test_frame()
            result = enhance_gpu_realesrgan(frame)
            # Deve retornar None sem lançar exceção
            self.assertIsNone(result)


class TestImageEnhancementBypass(unittest.TestCase):
    """Testa o bypass de aprimoramento de imagem."""

    def test_bypass_enhancement_saves_original_image(self):
        """Quando apply_enhancement=False, o frame gravado deve ser idêntico ao original."""
        import tempfile
        from core.image_enhancer import enhance_and_save

        frame = _create_test_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'original_frame.jpg'
            ok = enhance_and_save(frame, output, apply_enhancement=False)
            self.assertTrue(ok)
            self.assertTrue(output.exists())

            # Carrega e compara os frames (tolerando compressão JPEG normal)
            loaded = cv2.imread(str(output))
            self.assertEqual(loaded.shape, frame.shape)
            
            # Devido à compressão JPEG, os pixels podem variar ligeiramente,
            # mas devem ser extremamente próximos (MAE < 2.0). Se CLAHE fosse aplicado,
            # o erro médio seria consideravelmente maior.
            mae = np.mean(np.abs(loaded.astype(int) - frame.astype(int)))
            self.assertLess(mae, 2.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
