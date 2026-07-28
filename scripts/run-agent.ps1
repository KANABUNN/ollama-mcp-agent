[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw ".env not found. Run scripts\setup.ps1 first."
}

Push-Location $ProjectRoot
try {
$ConfigJson = & $VenvPython -I -m ollama_mcp_agent.config 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "The runtime configuration is invalid. Check .env and try again."
}
try {
    $RuntimeConfig = $ConfigJson | ConvertFrom-Json
}
catch {
    throw "The validated runtime configuration could not be read."
}
if (-not $RuntimeConfig.model_configured) {
    throw "OLLAMA_MODEL is empty. Set it to an installed tool-capable model."
}

$OllamaUri = [uri]$RuntimeConfig.ollama_host
$TagsUri = [uri]::new($OllamaUri, "/api/tags")
$Response = $null
Add-Type -AssemblyName System.Net.Http
$HttpHandler = [System.Net.Http.HttpClientHandler]::new()
$HttpHandler.AllowAutoRedirect = $false
$HttpHandler.UseProxy = $false
$HttpClient = [System.Net.Http.HttpClient]::new($HttpHandler)
$HttpClient.Timeout = [TimeSpan]::FromSeconds(5)
try {
    $Response = $HttpClient.GetAsync($TagsUri).GetAwaiter().GetResult()
    if (-not $Response.IsSuccessStatusCode) {
        throw "Ollama returned HTTP $([int]$Response.StatusCode)."
    }
}
catch {
    throw "Ollama is not reachable at the configured loopback address. Start Ollama and try again."
}
finally {
    if ($null -ne $Response) {
        $Response.Dispose()
    }
    $HttpClient.Dispose()
    $HttpHandler.Dispose()
}

    & $VenvPython -I -m ollama_mcp_agent.agent
    $AgentExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $AgentExitCode
