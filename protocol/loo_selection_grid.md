# LODO Development Grid (seed 0)

Selection metric: mean **accuracy** on the 17 non-held-out datasets.
Reported averages below are over all 18 datasets (seed 0) for reference; LODO never scores the held-out dataset when choosing a recipe.

| Candidate | \(k=1\) Acc (w/o ens / w ens) | \(k=5\) Acc (w/o ens / w ens) | LOO selections \(k=1\) | LOO selections \(k=5\) |
|---|---:|---:|---:|---:|
| UniMTS FT (official) | 61.7 / 62.2 | 76.7 / 76.8 | 0 | 0 |
| + uniform SO(3) | 54.2 / 54.4 | 66.2 / 66.3 | 0 | 0 |
| + small-angle \(30^\circ\) (**RoFT** when ensembled) | 66.4 / 66.7 | 80.6 / 80.7 | **18** | **18** |
| 30 epochs | 59.5 / 60.7 | 71.7 / 72.2 | 0 | 0 |
| TipGuided | 61.1 / — | 74.9 / — | 0 | 0 |
| TipGuided + SO(3) | 57.1 / 57.8 | 68.4 / 69.2 | 0 | 0 |
| TipGuided + small-angle | 62.9 / 62.9 | 77.1 / 77.5 | 0 | 0 |
| TipGuided, head lr \(10^{-3}\) | 60.2 / 62.6 | 75.4 / 76.5 | 0 | 0 |
| TipGuided, 30 epochs | 59.4 / 61.9 | 73.7 / 74.5 | 0 | 0 |

Reproduce selection counts after placing JSON logs under `logs/bench/`:

```bash
python tools/loo_select.py \
  --family all=unimts_ft,ft_so3,ft_so3_ens,ft_small,ft_small_ens,ft_noaug_ens \
  --k 1 5 --seed 0 --metric acc
```

Note: diagnostic angle tables may prefer \(60^\circ\) on five-shot **macro-F1**; LODO used **accuracy**, which selected \(30^\circ\) + ensemble for every held-out fold.
