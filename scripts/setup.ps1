[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"
$AllowedRoot = "C:\LLM-Workspace"

function Get-BootstrapPython {
    $PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        $Version = & $PyLauncher.Source -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0) {
            return @($PyLauncher.Source, "-3", $Version)
        }
    }

    $Python = Get-Command "python" -ErrorAction SilentlyContinue
    if ($null -ne $Python) {
        $Version = & $Python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0) {
            return @($Python.Source, "", $Version)
        }
    }

    throw "Python 3.11 or later was not found. Install Python, then run this script again."
}

$Bootstrap = Get-BootstrapPython
$PythonExecutable = $Bootstrap[0]
$PythonSelector = $Bootstrap[1]
$PythonVersion = [version]$Bootstrap[2]
if ($PythonVersion -lt [version]"3.11") {
    throw "Python 3.11 or later is required. Detected: $PythonVersion"
}

Write-Host "Python ${PythonVersion}: $PythonExecutable"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host "Creating virtual environment: $VenvPath"
    if ([string]::IsNullOrEmpty($PythonSelector)) {
        & $PythonExecutable -m venv $VenvPath
    }
    else {
        & $PythonExecutable $PythonSelector -m venv $VenvPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}

Write-Host "Updating pip in the project virtual environment..."
& $VenvPython -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip update failed."
}

Write-Host "Installing the project and development dependencies..."
& $VenvPython -m pip install --no-cache-dir --editable "${ProjectRoot}[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "Project installation failed."
}

if (-not (Test-Path -LiteralPath $AllowedRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $AllowedRoot | Out-Null
    Write-Host "Created allowed workspace: $AllowedRoot"
}

if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    Write-Host "Created .env from .env.example."
}

Write-Host ""
Write-Host "Setup completed."
Write-Host "1. Open $EnvFile and set OLLAMA_MODEL to an installed tool-capable model."
Write-Host "2. Start Ollama."
Write-Host "3. Run scripts\verify.ps1, then scripts\run-agent.ps1."
