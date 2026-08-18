# run_all_tests.ps1
# Executa todos os testes automatizados do VideoFrameExtractor.
# Salvo em disco para compatibilidade com executor PowerShell.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "============================================================"
Write-Host "        VideoFrameExtractor - Suite de Testes               "
Write-Host "============================================================"
Write-Host ""

# Detectar Python disponivel
$PythonExe = $null

# 1. Tentar venv local do projeto
$VenvPython = Join-Path (Join-Path $env:LOCALAPPDATA "VideoFrameExtractor") "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "  Python: venv local ($VenvPython)"
}

# 2. Fallback para Python do sistema (ignorando o stub da MS Store em WindowsApps)
if (-not $PythonExe) {
    $SysPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($SysPython -and $SysPython -notlike "*WindowsApps*") {
        $PythonExe = $SysPython
        Write-Host "  Python: sistema ($SysPython)"
    }
}

# 3. Fallback para Python do Google Drive (instalacao customizada detectada)
if (-not $PythonExe) {
    $GDrivePython = "d:\Meu Drive\dev\programs\Python311\python.exe"
    if (Test-Path $GDrivePython) {
        $PythonExe = $GDrivePython
        Write-Host "  Python: Google Drive ($GDrivePython)"
    }
}

if (-not $PythonExe) {
    Write-Host "  ERRO: Python nao encontrado. Instale Python 3.10+ e tente novamente."
    exit 1
}

# Verificar dependencias minimas para testes
Write-Host "  Verificando dependencias de teste..."
$DepCheck = & $PythonExe -c "import cv2, numpy; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0 -or $DepCheck -notmatch "OK") {
    Write-Host "  AVISO: opencv-python ou numpy nao instalados. Alguns testes serao pulados."
}

Write-Host ""
Write-Host "  Executando testes..."
Write-Host ("  " + ("-" * 60))

# Lista de modulos de teste
$TestModules = @(
    "tests.test_video_scanner",
    "tests.test_frame_sampler",
    "tests.test_quality_scorer",
    "tests.test_frame_selector",
    "tests.test_image_enhancer"
)

$TotalPassed = 0
$TotalFailed = 0
$TotalErrors = 0

foreach ($Module in $TestModules) {
    Write-Host ""
    Write-Host "  >> $Module"

    # Executa cada modulo de teste
    $TestScript = @"
import sys, unittest
sys.path.insert(0, r'$ProjectDir')
loader = unittest.TestLoader()
suite = loader.loadTestsFromName('$Module')
runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"@

    # Salva script temporario para evitar problemas com caracteres especiais
    $TempScript = Join-Path $env:TEMP "vfe_test_runner.py"
    $TestScript | Out-File -FilePath $TempScript -Encoding utf8

    & $PythonExe $TempScript 2>&1 | ForEach-Object { Write-Host "     $_" }
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        $TotalPassed++
        Write-Host "     RESULTADO: PASSOU" -ForegroundColor Green
    } else {
        $TotalFailed++
        Write-Host "     RESULTADO: FALHOU (codigo $ExitCode)" -ForegroundColor Red
    }
}

# Limpar arquivo temporario
if (Test-Path $TempScript) {
    Remove-Item $TempScript -Force
}

# Resumo final
Write-Host ""
Write-Host ("  " + ("-" * 60))
Write-Host "  RESUMO DOS TESTES:"
Write-Host "  Modulos aprovados : $TotalPassed / $($TestModules.Count)"
Write-Host "  Modulos com falha : $TotalFailed / $($TestModules.Count)"
Write-Host ""

if ($TotalFailed -eq 0) {
    Write-Host "  [OK] TODOS OS TESTES PASSARAM!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "  [FALHA] $TotalFailed MODULO(S) COM FALHA" -ForegroundColor Red
    Write-Host "  Verifique os erros acima e corrija antes de distribuir."
    exit 1
}
