# -*- coding: utf-8 -*-
"""
tests/test_video_scanner.py — Testes unitários do módulo video_scanner.

Usa mocks para simular OpenCV e PathLib, permitindo testes sem vídeos reais.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Garante acesso ao projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestNormalizeUncPath(unittest.TestCase):
    """Testa a normalização de caminhos UNC e locais em config.py."""

    def test_local_path_unchanged(self):
        from core.utils import normalize_unc_path
        result = normalize_unc_path('C:\\Users\\test\\videos')
        self.assertEqual(str(result), 'C:\\Users\\test\\videos')

    def test_unc_path_double_backslash(self):
        from core.utils import normalize_unc_path
        result = normalize_unc_path('\\\\servidor\\share\\pasta')
        self.assertTrue(str(result).startswith('\\\\'))

    def test_removes_quotes(self):
        from core.utils import normalize_unc_path
        result = normalize_unc_path('"\\\\servidor\\share\\pasta"')
        self.assertFalse(str(result).startswith('"'))

    def test_forward_slash_unc(self):
        from core.utils import normalize_unc_path
        result = normalize_unc_path('//servidor/share/pasta')
        self.assertTrue(str(result).startswith('\\\\'))

    def test_strips_whitespace(self):
        from core.utils import normalize_unc_path
        result = normalize_unc_path('  C:\\videos  ')
        self.assertEqual(str(result), 'C:\\videos')


class TestSuggestFrameCount(unittest.TestCase):
    """Testa sugestão automática de quantidade de frames."""

    def setUp(self):
        from core.utils import suggest_frame_count
        self.suggest = suggest_frame_count

    def test_short_video_under_1min(self):
        self.assertEqual(self.suggest(30), 5)

    def test_medium_video_under_5min(self):
        self.assertEqual(self.suggest(240), 10)

    def test_long_video_under_30min(self):
        self.assertEqual(self.suggest(600), 20)

    def test_very_long_video(self):
        self.assertEqual(self.suggest(3600), 30)

    def test_boundary_exactly_60sec(self):
        # Exatamente 60s → já não é "< 60" → categoria 1-5min
        self.assertEqual(self.suggest(60), 10)


class TestScanFolder(unittest.TestCase):
    """Testa a varredura de pasta via mocks."""

    def test_raises_if_folder_not_exists(self):
        from core.video_scanner import scan_folder
        fake_path = Path('Z:\\caminho\\inexistente')
        with self.assertRaises(FileNotFoundError):
            scan_folder(fake_path, frozenset({'.mp4'}))

    def test_raises_if_not_directory(self):
        from core.video_scanner import scan_folder
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=False):
            with self.assertRaises(NotADirectoryError):
                scan_folder(Path('C:\\arquivo.mp4'), frozenset({'.mp4'}))

    def test_empty_folder_returns_empty_list(self):
        from core.video_scanner import scan_folder
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(Path, 'iterdir', return_value=iter([])):
            result = scan_folder(Path('C:\\videos'), frozenset({'.mp4'}))
        self.assertEqual(result, [])

    def test_scans_mp4_files(self):
        """Testa que arquivos .mp4 são detectados e metadados extraídos via mock de _extract_video_metadata."""
        from core.video_scanner import scan_folder
        import core.video_scanner as scanner_mod

        # Simula pasta com um arquivo .mp4
        fake_mp4 = MagicMock(spec=Path)
        fake_mp4.name = 'video_teste.mp4'
        fake_mp4.suffix = '.mp4'
        fake_mp4.is_file.return_value = True
        fake_mp4.stat.return_value = MagicMock(st_size=10 * 1024 * 1024)
        fake_mp4.__lt__ = lambda self, other: self.name < other.name

        # Mocka _extract_video_metadata diretamente para não depender de constantes OpenCV
        def fake_extract(path):
            return {'duration_sec': 10.0, 'fps': 30.0, 'width': 1920, 'height': 1080}

        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_dir', return_value=True), \
             patch.object(Path, 'iterdir', return_value=iter([fake_mp4])), \
             patch.object(scanner_mod, '_extract_video_metadata', side_effect=fake_extract):
            result = scan_folder(Path('C:\\videos'), frozenset({'.mp4'}))

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].fps, 30.0)
        self.assertEqual(result[0].width, 1920)
        self.assertEqual(result[0].height, 1080)
        self.assertAlmostEqual(result[0].size_mb, 10.0)

    def test_video_file_duration_formatted_minutes(self):
        """Testa formatação de duração."""
        from core.video_scanner import VideoFile
        vf = VideoFile(
            path=Path('video.mp4'), size_bytes=1024,
            duration_sec=125.0, fps=30.0, width=1920, height=1080,
        )
        self.assertEqual(vf.duration_formatted, '02:05')

    def test_video_file_duration_formatted_hours(self):
        from core.video_scanner import VideoFile
        vf = VideoFile(
            path=Path('video.mp4'), size_bytes=1024,
            duration_sec=3725.0, fps=30.0, width=1920, height=1080,
        )
        self.assertEqual(vf.duration_formatted, '01:02:05')


if __name__ == '__main__':
    unittest.main(verbosity=2)
