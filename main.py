# -*- coding: utf-8 -*-
"""
main.py — CLI interativa PT-BR do VideoFrameExtractor.

Menu principal → seleção de pasta → configuração de frames → processamento.
Suporte completo a caminhos UNC (\\servidor\share\pasta).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Garante que o diretório do projeto está no PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    MIN_FRAME_DISTANCE_SEC,
    OUTPUT_SUBDIR,
    VIDEO_EXTENSIONS,
)
from core.utils import (
    normalize_unc_path,
    suggest_frame_count,
)

logger = logging.getLogger(__name__)


# ── Configuração de logging ───────────────────────────────────────────────────

def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silencia logs muito verbosos de bibliotecas externas
    for lib in ('PIL', 'basicsr', 'torch', 'torchvision', 'urllib3'):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── Utilitários de terminal ───────────────────────────────────────────────────

def _clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def _print_header() -> None:
    print("============================================================")
    print("         VideoFrameExtractor - Extrator Inteligente         ")
    print("              Selecao por IA | GPU/CPU | UNC                ")
    print("============================================================")
    print()


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Retorna barra de progresso ASCII."""
    filled = int(width * current / max(total, 1))
    bar = '#' * filled + '-' * (width - filled)
    pct = int(100 * current / max(total, 1))
    return f"[{bar}] {pct}% ({current}/{total})"


def _ask_input(prompt: str, default: str = '') -> str:
    """Solicita entrada ao usuário com valor padrão."""
    if default:
        full_prompt = f"  {prompt} [{default}]: "
    else:
        full_prompt = f"  {prompt}: "
    try:
        value = input(full_prompt).strip()
        return value if value else default
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Operação cancelada pelo usuário.")
        sys.exit(0)


def _ask_int(prompt: str, default: int, min_val: int = 1, max_val: int = 999) -> int:
    """Solicita um número inteiro ao usuário."""
    while True:
        raw = _ask_input(prompt, str(default))
        try:
            value = int(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  [ERRO] Valor fora do intervalo [{min_val}-{max_val}]. Tente novamente.")
        except ValueError:
            print("  [ERRO] Digite um numero inteiro valido.")


# ── Menu principal ────────────────────────────────────────────────────────────

def _menu_principal() -> str:
    _print_header()
    print("  [1] Processar vídeos de uma pasta")
    print("  [2] Diagnóstico de hardware (GPU/CPU/CUDA)")
    print("  [3] Sair")
    print()
    return _ask_input("Escolha uma opção", "1")


# ── Fluxo de processamento ────────────────────────────────────────────────────

def _solicitar_pasta() -> Path:
    """Solicita e valida o caminho da pasta de vídeos."""
    while True:
        raw = _ask_input(
            "Caminho da pasta com os vídeos\n  (local ou UNC, ex: \\\\servidor\\share\\videos)"
        )
        if not raw:
            print("  [ERRO] Caminho nao pode ser vazio.")
            continue
        folder = normalize_unc_path(raw)
        if not folder.exists():
            print(f"  [ERRO] Pasta nao encontrada: {folder}")
            continue
        if not folder.is_dir():
            print(f"  [ERRO] O caminho nao e um diretorio: {folder}")
            continue
        return folder


def _exibir_lista_videos(videos: list) -> None:
    """Exibe tabela formatada com os vídeos encontrados."""
    print()
    print(f"  {'#':<4} {'Arquivo':<45} {'Tamanho':>9} {'Duracao':>10} {'Resolucao':>12}")
    print("  " + "-" * 85)
    for i, v in enumerate(videos, 1):
        nome = v.name[:44] + '...' if len(v.name) > 44 else v.name
        print(
            f"  {i:<4} {nome:<45} {v.size_mb:>8.1f}MB"
            f" {v.duration_formatted:>10} {v.resolution_label:>12}"
        )
    print()


def _processar_video(
    video,
    frame_count: int,
    output_dir: Path,
    apply_enhancement: bool = True,
    use_scoring: bool = True,
    allow_groq: bool = False,
) -> tuple[int, int]:
    """
    Executa o pipeline completo para um único vídeo.
    Returns: (frames_salvos, frames_falha).
    """
    from core.frame_sampler import sample_frames
    from core.frame_selector import select_best_frames, SelectedFrame, build_output_filename
    from core.image_enhancer import enhance_and_save
    from core.quality_scorer import score_frames_batch, FrameQualityScore

    print(f"\n  [>] Processando: {video.name}")

    if use_scoring:
        # 1. Amostragem: pega frame_count * 3 candidatos para ter margem de seleção
        candidate_count = min(frame_count * 3, 90)
        print(f"    Amostrando {candidate_count} frames candidatos...", end=' ', flush=True)
        sampled = sample_frames(video.path, count=candidate_count)
        print(f"{len(sampled)} frames amostrados.")

        if not sampled:
            print("    [ERRO] Nenhum frame amostrado. Video pode estar corrompido.")
            return 0, 1

        # 2. Scoring de qualidade
        print(f"    Calculando scores de qualidade...", end=' ', flush=True)
        scores = score_frames_batch(sampled, allow_groq=allow_groq)
        print("concluído.")

        # 3. Seleção dos melhores
        print(f"    Selecionando os {frame_count} melhores frames...", end=' ', flush=True)
        selected = select_best_frames(
            sampled_frames=sampled,
            scores=scores,
            target_count=frame_count,
            min_distance_sec=MIN_FRAME_DISTANCE_SEC,
            video_stem=video.path.stem,
        )
        print(f"{len(selected)} selecionados.")
    else:
        # Divisão uniforme direta sem scoring
        print(f"    Amostrando {frame_count} frames uniformes (sem scoring)...", end=' ', flush=True)
        sampled = sample_frames(video.path, count=frame_count)
        print(f"{len(sampled)} frames amostrados.")

        if not sampled:
            print("    [ERRO] Nenhum frame amostrado. Video pode estar corrompido.")
            return 0, 1

        selected = []
        for rank, sf in enumerate(sampled, start=1):
            filename = build_output_filename(
                video_stem=video.path.stem,
                rank=rank,
                timestamp_sec=sf.timestamp_sec,
                composite_score=0.0
            )
            dummy_score = FrameQualityScore(
                frame_index=sf.index,
                laplacian=0.0,
                tenengrad=0.0,
                illumination=0.0,
                brisque=None,
                composite=0.0,
                brisque_source='none'
            )
            selected.append(SelectedFrame(
                rank=rank,
                sampled=sf,
                score=dummy_score,
                output_filename=filename
            ))

    # 4. Aprimoramento e salvamento
    salvos = 0
    falhas = 0
    if apply_enhancement:
        print(f"    Aprimorando e salvando frames...")
    else:
        print(f"    Salvando frames originais (sem aprimoramentos)...")

    for sel in selected:
        output_path = output_dir / sel.output_filename
        ok = enhance_and_save(sel.sampled.image_bgr, output_path, apply_enhancement=apply_enhancement)
        if ok:
            salvos += 1
            if use_scoring:
                score_val = sel.score.composite
                print(f"      [OK] {sel.output_filename}  (score: {score_val:.1f})")
            else:
                print(f"      [OK] {sel.output_filename}")
        else:
            falhas += 1
            print(f"      [ERRO] Falha ao salvar frame #{sel.rank}")

    return salvos, falhas


def _fluxo_processar_pasta() -> None:
    """Fluxo completo de processamento de uma pasta de vídeos."""
    from core.video_scanner import scan_folder

    # 1. Solicitar pasta
    print()
    pasta = _solicitar_pasta()

    # 2. Varrer vídeos
    print(f"\n  Varrendo vídeos em: {pasta}")
    try:
        videos = scan_folder(pasta, VIDEO_EXTENSIONS)
    except Exception as exc:
        print(f"  [ERRO] Erro ao varrer pasta: {exc}")
        return

    if not videos:
        print("  [ERRO] Nenhum video encontrado na pasta.")
        return

    print(f"\n  {len(videos)} vídeo(s) encontrado(s):")
    _exibir_lista_videos(videos)

    # 3. Configurar quantidade de frames
    total_duration = sum(v.duration_sec for v in videos)
    avg_duration = total_duration / len(videos)
    suggested = suggest_frame_count(avg_duration)
    print(f"  Duração média dos vídeos: {int(avg_duration)}s")
    frame_count = _ask_int(
        f"Quantos frames por vídeo extrair",
        default=suggested,
        min_val=1,
        max_val=100,
    )

    # 4. Configurar pasta de saída
    default_output = pasta / OUTPUT_SUBDIR
    raw_output = _ask_input(
        "Pasta de saída dos frames",
        str(default_output),
    )
    output_dir = normalize_unc_path(raw_output)

    # 4.5. Configurar aprimoramento de imagem
    raw_enhance = _ask_input(
        "Aplicar aprimoramentos nos frames (CLAHE, Denoise, Sharpen)? (S/N)",
        "S"
    ).upper()
    apply_enhancement = (raw_enhance == 'S')

    # 4.6. Configurar escolha de melhores frames (Scoring de Qualidade)
    raw_scoring = _ask_input(
        "Escolher melhores frames automaticamente por IA (Scoring)? (S/N)",
        "S"
    ).upper()
    use_scoring = (raw_scoring == 'S')

    allow_groq = False
    if use_scoring:
        from config import resolve_groq_api_key
        if resolve_groq_api_key():
            print("\n  [ATENCAO] Chave do Groq Cloud detectada no ambiente.")
            print("  Se o processador de qualidade local (pyiqa) falhar, o sistema")
            print("  pode enviar frames para analise externa (nuvem Groq).")
            raw_groq = _ask_input(
                "  Permitir envio de frames para nuvem Groq se necessario? (S/N)",
                "N"
            ).upper()
            allow_groq = (raw_groq == 'S')

    print(f"\n  Configuração confirmada:")
    print(f"    Vídeos    : {len(videos)}")
    print(f"    Frames/vídeo: {frame_count}")
    print(f"    Saída     : {output_dir}")
    print(f"    Escolha IA: {'Ativa (Scoring)' if use_scoring else 'Desativada (Divisão uniforme pura)'}")
    print(f"    Enviar Nuvem: {'Sim (se necessário)' if allow_groq else 'Não (totalmente offline)'}")
    print(f"    Aprimorar : {'Sim' if apply_enhancement else 'Não (rigorosamente fiel ao vídeo)'}")
    print()

    confirmar = _ask_input("Iniciar processamento? (S/N)", "S").upper()
    if confirmar != 'S':
        print("  Processamento cancelado.")
        return

    # 5. Processar cada vídeo
    inicio = time.time()
    total_salvos = 0
    total_falhas = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos, 1):
        print(f"\n  {_progress_bar(i - 1, len(videos))} - {video.name}")
        try:
            salvos, falhas = _processar_video(video, frame_count, output_dir, apply_enhancement, use_scoring, allow_groq=allow_groq)
            total_salvos += salvos
            total_falhas += falhas
        except Exception as exc:
            logger.exception("Erro ao processar %s: %s", video.name, exc)
            print(f"  [ERRO] Erro inesperado ao processar {video.name}: {exc}")
            total_falhas += 1

    # 6. Resumo final
    elapsed = time.time() - inicio
    print(f"\n  {_progress_bar(len(videos), len(videos))}")
    print()
    print("  +------------------------------------------+")
    print("  |          PROCESSAMENTO CONCLUIDO         |")
    print(f"  |  Frames salvos  : {total_salvos:<24}|")
    print(f"  |  Erros          : {total_falhas:<24}|")
    print(f"  |  Tempo total    : {elapsed:.1f}s{' ':<21}|")
    print(f"  |  Pasta de saida : ver pasta indicada    |")
    print("  +------------------------------------------+")
    print()
    print(f"  Frames salvos em: {output_dir}")
    print()


def _fluxo_diagnostico() -> None:
    """Exibe diagnóstico de hardware e ambiente."""
    from install.env_detector import build_environment_report

    print()
    print("  -- Diagnostico de Ambiente ------------------------------")
    report = build_environment_report()
    print(f"    Python        : {report['python_version']} {'OK' if report['python_ok'] else 'VERSAO INSUFICIENTE'}")
    print(f"    Plataforma    : {report['platform']}")
    print(f"    Executavel    : {report['executable']}")
    print()
    cuda = report['cuda']
    if cuda['available']:
        print(f"    GPU           : {cuda['device_name']}")
        print(f"    CUDA          : {cuda['cuda_version']}")
        print(f"    VRAM          : {cuda['vram_gb']} GB")
    else:
        print("    GPU/CUDA      : Nao disponivel - processamento via CPU")
    print(f"    Pacotes inst. : {report['installed_packages_count']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Evita crashes de encoding em consoles Windows cp1252/cp850
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(errors='replace')
        except Exception:
            pass

    from config import LOCAL_DATA_DIR, LOG_DIR
    _configure_logging(LOG_DIR)

    while True:
        _clear()
        opcao = _menu_principal()

        if opcao == '1':
            _fluxo_processar_pasta()
            input("  Pressione ENTER para continuar...")
        elif opcao == '2':
            _fluxo_diagnostico()
            input("  Pressione ENTER para continuar...")
        elif opcao == '3':
            print("\n  Ate logo!\n")
            sys.exit(0)
        else:
            print("  [ERRO] Opcao invalida. Tente novamente.")
            time.sleep(1)


if __name__ == '__main__':
    main()
