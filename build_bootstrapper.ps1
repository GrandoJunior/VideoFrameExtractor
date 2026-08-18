# build_bootstrapper.ps1
# Script de build do VideoFrameExtractor.exe
# Salvo em disco para compatibilidade com executor PowerShell do Antigravity.
# Restrição: nunca usar -Command inline com variaveis PowerShell.

param(
    [switch]$Clean = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DistDir = Join-Path $ProjectDir "dist"
$BuildDir = Join-Path $ProjectDir "build"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║       VideoFrameExtractor — Build do Bootstrapper        ║"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""

# Limpar builds anteriores se solicitado
if ($Clean) {
    Write-Host "  Limpando builds anteriores..."
    if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
    if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
    Write-Host "  Limpeza concluída."
}

# Verificar Python
Write-Host "  [1/4] Verificando Python..."
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Host "  ERRO: Python nao encontrado no PATH."
    exit 1
}
$PythonVersion = & python --version 2>&1
Write-Host "  Python: $PythonVersion"

# Verificar/instalar PyInstaller
Write-Host "  [2/4] Verificando PyInstaller..."
$PyInstallerCheck = & python -c "import PyInstaller; print(PyInstaller.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  PyInstaller nao encontrado. Instalando..."
    & python -m pip install pyinstaller --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERRO: Falha ao instalar PyInstaller."
        exit 1
    }
}
Write-Host "  PyInstaller: $PyInstallerCheck"

# Compilar o bootstrapper
Write-Host "  [3/4] Compilando bootstrapper.py..."
Push-Location $ProjectDir
try {
    & python -m PyInstaller bootstrapper.spec --noconfirm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERRO: Falha na compilacao com PyInstaller."
        exit 1
    }
} finally {
    Pop-Location
}

# Verificar o executável gerado
$ExePath = Join-Path $DistDir "VideoFrameExtractor.exe"
Write-Host "  [4/4] Verificando executavel gerado..."
if (Test-Path $ExePath) {
    $ExeSize = (Get-Item $ExePath).Length / 1MB
    Write-Host ""
    Write-Host "  ✓ Build concluido com sucesso!"
    Write-Host "  Executavel: $ExePath"
    Write-Host "  Tamanho   : $([math]::Round($ExeSize, 1)) MB"
    Write-Host ""
    Write-Host "  Para distribuir, copie VideoFrameExtractor.exe para a mesma pasta"
    Write-Host "  do projeto (junto com config.py, requirements.txt, core/, install/)."
    Write-Host ""
} else {
    Write-Host "  ERRO: Executavel nao encontrado em: $ExePath"
    exit 1
}
