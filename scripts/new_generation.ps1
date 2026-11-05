<#
.SYNOPSIS
  Ejecuta una generacion completa: juega partidas con la red actual, entrena una red
  nueva con todos los datos y la enfrenta a la anterior.

.DESCRIPTION
  Cada generacion necesita un nombre nuevo y entrena con todos los archivos data\*.bin.
  -Threads limita tanto la generacion de datos como el entrenamiento.

  Para decidir, mira el resultado final. Si SPRT no llega a un veredicto, la red nueva es
  mejor cuando su Elo menos el margen sigue siendo positivo; si el margen cruza el cero,
  repite el match con mas partidas (-MatchGames).

.EXAMPLE
  .\scripts\new_generation.ps1 -Name gen4 -Minutes 60

  Guarda data\gen4.bin y nets\gen4-*.nnue. Si la red nueva gana el match, copiala a
  nets\default.nnue y recompila para integrarla en el ejecutable.
#>
param(
  [Parameter(Mandatory = $true)][ValidatePattern('^[\w-]+$')][string]$Name,
  [ValidateRange(1, 100000)][int]$Minutes = 60,
  [ValidateRange(1, 1024)][int]$Threads = 10,
  [ValidateRange(1, 100000000)][int]$Nodes = 5000,
  [ValidateRange(1, 1000)][int]$Epochs = 20,
  [ValidateRange(1, 1000000)][int]$Batch = 4096,
  [ValidateRange(0.0, 1.0)][double]$Wdl = 0,
  [ValidateRange(1, 2000)][int]$Scale = 200,
  [string]$CurrentNet = "nets\default.nnue",
  [ValidateRange(2, 100000)][int]$MatchGames = 400
)

$ErrorActionPreference = "Stop"
$bin = "build\release"

Push-Location (Join-Path $PSScriptRoot "..")
try {
  foreach ($tool in "datagen", "train", "match", "ayano") {
    if (-not (Test-Path "$bin\$tool.exe")) { throw "Falta $tool.exe: compila con scripts\build.cmd" }
  }
  if (-not (Test-Path $CurrentNet)) { throw "No existe la red $CurrentNet" }
  if ((Test-Path "data\$Name.bin") -or (Test-Path "nets\$Name-*.nnue")) {
    throw "Ya existe una generacion llamada ${Name}: elige otro nombre"
  }

  $seed = Get-Random -Maximum 1000000
  Write-Host "== 1/3 Generando datos durante $Minutes minutos con $CurrentNet"
  & "$bin\datagen.exe" --out "data\$Name.bin" --threads $Threads --nodes $Nodes `
    --seed $seed --minutes $Minutes --net $CurrentNet
  if ($LASTEXITCODE -ne 0) { throw "datagen fallo" }

  Write-Host "== 2/3 Entrenando con todos los datos de data\"
  $dataArgs = Get-ChildItem data\*.bin | ForEach-Object { "--data"; $_.FullName }
  & "$bin\train.exe" @dataArgs --out "nets\$Name" --epochs $Epochs --threads $Threads `
    --batch $Batch --wdl $Wdl --scale $Scale --seed $seed
  if ($LASTEXITCODE -ne 0) { throw "train fallo" }

  $candidate = "nets\$Name-$Epochs.nnue"
  Write-Host "== 3/3 Enfrentando $candidate contra $CurrentNet"
  & "$bin\match.exe" --engine1 "$bin\ayano.exe" --engine2 "$bin\ayano.exe" `
    --opt1 "EvalFile=$candidate" --opt2 "EvalFile=$CurrentNet" `
    --games $MatchGames --concurrency 5 --tc 8+0.08 --sprt 0 10 --seed $seed
  if ($LASTEXITCODE -ne 0) { throw "match fallo: el resultado no es valido" }
}
finally {
  Pop-Location
}
