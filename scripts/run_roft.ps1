# Run RoFT on the UniMTS 18-dataset few-shot protocol.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$DataPath = if ($env:DATA_PATH) { $env:DATA_PATH } else { "./UniMTS_data" }
$Ckpt = if ($env:CKPT) { $env:CKPT } else { "./checkpoint/UniMTS.pth" }
$Out = if ($env:OUT) { $env:OUT } else { "./logs/bench" }
$Ks = if ($env:KS) { $env:KS.Split(" ") } else { @(1, 2, 3, 5, 10) }
$Seeds = if ($env:SEEDS) { $env:SEEDS.Split(" ") } else { @(0, 1, 2) }

New-Item -ItemType Directory -Force -Path $Out, "checkpoint" | Out-Null

foreach ($k in $Ks) {
  foreach ($s in $Seeds) {
    python tools/fewshot_bench.py `
      --method unimts_ft `
      --tag ft_small_ens `
      --aug small --aug_angle 30 --ens 1 `
      --k $k --seed $s `
      --epochs 200 --batch_size 64 --lr 1e-4 `
      --data_path $DataPath `
      --checkpoint $Ckpt `
      --out_dir $Out
  }
}
