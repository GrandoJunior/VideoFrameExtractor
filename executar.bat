@echo off
:: VideoFrameExtractor — Inicializador
:: pushd mapeia automaticamente UNC para letra de drive temporaria
title VideoFrameExtractor

pushd "%~dp0"

if exist VideoFrameExtractor.exe (
    VideoFrameExtractor.exe
    set EXIT_CODE=%errorlevel%
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python nao encontrado no PATH. Instale Python 3.10 ou superior, ou use o executavel compilado VideoFrameExtractor.exe.
        pause
        popd
        exit /b 1
    )
    python -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo Python nao configurado corretamente [alias da Microsoft Store ativo ou erro]. Instale Python 3.10+ ou use o executavel VideoFrameExtractor.exe.
        pause
        popd
        exit /b 1
    )
    python bootstrapper.py
    set EXIT_CODE=%errorlevel%
)

popd

if %EXIT_CODE% neq 0 pause
