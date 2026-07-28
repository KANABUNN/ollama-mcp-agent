[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $VenvPython -m ruff check .
    if ($LASTEXITCODE -ne 0) {
        throw "ruff check failed."
    }

    & $VenvPython -m pytest
    if ($LASTEXITCODE -ne 0) {
        throw "pytest failed."
    }
}
finally {
    Pop-Location
}

