param(
    [ValidateSet('pilot', 'long')]
    [string]$Profile = 'long',
    [int[]]$TrainingSeeds = @(7, 11, 19),
    [int[]]$EvaluationSeeds = @(101, 102, 103, 104, 105),
    [int]$Updates = 0,
    [string]$PythonPath = '',
    [string]$ResumeCampaign = ''
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$experimentRoot = Join-Path (Split-Path -Parent $project) '06_EXPERIMENTS/long_campaigns'
New-Item -ItemType Directory -Force -Path $experimentRoot | Out-Null

if (-not $PythonPath) {
    $localPython = Join-Path $project '.venv/Scripts/python.exe'
    if (Test-Path $localPython) {
        $PythonPath = $localPython
    } else {
        $PythonPath = (Get-Command python).Source
    }
}

$arguments = @('scripts/run_training_campaign.py')
if ($ResumeCampaign) {
    $campaign = (Resolve-Path $ResumeCampaign).Path
    $arguments += @('--resume-campaign', ('"' + $campaign + '"'))
    $name = Split-Path $campaign -Leaf
} else {
    $name = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'") + "_$Profile"
    $config = if ($Profile -eq 'pilot') { 'configs/train_pilot.yaml' } else { 'configs/train_long.yaml' }
    $arguments += @('--config', $config, '--name', $name)
    $arguments += @('--training-seeds') + ($TrainingSeeds | ForEach-Object { [string]$_ })
    $arguments += @('--evaluation-seeds') + ($EvaluationSeeds | ForEach-Object { [string]$_ })
    if ($Updates -gt 0) {
        $arguments += @('--updates', [string]$Updates)
    }
}

$stdout = Join-Path $experimentRoot "$name.controller.stdout.log"
$stderr = Join-Path $experimentRoot "$name.controller.stderr.log"
$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments `
    -WorkingDirectory $project -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$pidFile = Join-Path $experimentRoot "$name.controller.pid"
Set-Content -LiteralPath $pidFile -Value $process.Id

[pscustomobject]@{
    ProcessId = $process.Id
    Campaign = if ($ResumeCampaign) { $campaign } else { Join-Path $experimentRoot $name }
    StandardOutput = $stdout
    StandardError = $stderr
    PidFile = $pidFile
}
