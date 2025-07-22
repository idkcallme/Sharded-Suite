@echo off
REM Simple batch script to build GGUF Shard Suite on Windows
REM Usage: build.bat [target]

setlocal

set TARGET=%1
if "%TARGET%"=="" set TARGET=build

echo GGUF Shard Suite Build Script (Batch)

REM Check for PowerShell
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found. Please install PowerShell.
    exit /b 1
)

REM Run the PowerShell script
echo Running PowerShell build script...
powershell.exe -ExecutionPolicy Bypass -File ".\build.ps1" -Target %TARGET%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo SUCCESS: Build completed successfully!
