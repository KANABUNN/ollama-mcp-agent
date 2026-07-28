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
    & $VenvPython -I -m ollama_mcp_agent.pc_tools_server
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
