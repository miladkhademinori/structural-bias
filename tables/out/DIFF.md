<!-- tables/build_all.py via pipeline.tables.diff_report -->
<!-- payload provenance A: ['synthetic']; B: ['synthetic']; synthetic rows are NOT paper numbers -->
# Printed vs reproduced

| table | cell | printed | reproduced | note |
|---|---|---|---|---|
| tab:gamma-min | vit_in21k/cifar100 | 8.3 ± 0.4 | $3.2 \pm 0.4$ | |
| tab:gamma-min | vit_in21k/imagenet_r | 5.2 ± 0.3 | $1.5 \pm 0.2$ | |
| tab:gamma-min | vit_in21k/cub200 | 12.1 ± 0.7 | -- | |
| tab:gamma-min | vit_in21k/omnibenchmark | 9.6 ± 0.5 | -- | |
| tab:gamma-min | dinov2_b14/cifar100 | 47.2 ± 1.2 | $10.8 \pm 0.9$ | |
| tab:gamma-min | dinov2_b14/imagenet_r | 39.1 ± 1.0 | $6.8 \pm 0.8$ | |
| tab:gamma-min | dinov2_b14/cub200 | 51.8 ± 1.4 | -- | |
| tab:gamma-min | dinov2_b14/omnibenchmark | 44.6 ± 1.1 | -- | |
| tab:gamma-min | clip_b16/cifar100 | 23.1 ± 0.8 | $4.9 \pm 0.3$ | |
| tab:gamma-min | clip_b16/imagenet_r | 18.9 ± 0.6 | $2.5 \pm 0.5$ | |
| tab:gamma-min | clip_b16/cub200 | 31.4 ± 0.9 | -- | |
| tab:gamma-min | clip_b16/omnibenchmark | 26.3 ± 0.7 | -- | |
| tab:random-anchor | cifar100 | γ 0.31, TC 0.189 | γ $0.15 \pm 0.02$, TC $0.391 \pm 0.005$ | |
| tab:random-anchor | imagenet_r | γ 0.28, TC 0.211 | γ $0.07 \pm 0.00$, TC $0.224 \pm 0.000$ | |
| tab:random-anchor | cub200 | γ 0.58, TC 0.159 | -- | |
| tab:random-anchor | omnibenchmark | γ 0.42, TC 0.174 | -- | |
| slopes | vit_in21k | -0.141 | -0.093 | R2 1.00 |
| slopes | dinov2_b14 | -0.099 | 0.027 | R2 1.00 |
| slopes | clip_b16 | -0.118 | 0.215 | R2 1.00 |
| slopes | pooled 12 | (-0.115, -0.128, -0.102) | 0.055 [0.032985734728648984, 0.08552174440547715] | |
| slopes | pooled 16 | (-0.119, -0.134, -0.105) | 0.057 [0.03326722030797961, 0.08839304730194607] | |
| predictor | ΔR2, p | 0.06, 0.042 | 0.000, 0.391 | R2 single {'gamma': 0.4811364264177189, 'probe': 0.9765837394912827, 'ncm': 0.9959216849368531} |
| tab:alpha-fits | vit_in21k | 1.32 | 40.52 | 33.18..50.65 |
| tab:alpha-fits | dinov2_b14 | 1.28 | 28.57 | 23.57..35.79 |
| tab:alpha-fits | clip_b16 | 1.35 | 34.08 | 27.93..43.25 |
| tab:ecf-rates | vit_in21k | clipped -0.31, unclipped -0.1 | clipped -1.667, unclipped -1.669 | |
| tab:ecf-rates | dinov2_b14 | clipped -0.31, unclipped -0.1 | clipped -0.810, unclipped -0.807 | |
| tab:fix-benchmark | dinov2_b14/cifar100 | 82.3/86.5/86.8 | 99.6/75.6/100.0 | |
| tab:fix-benchmark | dinov2_b14/imagenet100 | 80.5/84.7/85.0 | -- | |
| tab:fix-benchmark | dinov2_b14/imagenet1k | 76.8/81.8/82.3 | -- | |
| tab:fix-benchmark | clip_b16/cifar100 | 81.5/85.7/86.0 | 90.9/75.1/98.1 | |
| tab:fix-benchmark | clip_b16/imagenet100 | 79.2/83.5/83.8 | -- | |
| tab:fix-benchmark | mae_b16/imagenet100 | 65.3/77.2/78.5 | -- | |
| tab:fix-benchmark | resnet50/imagenet100 | 62.1/73.8/75.2 | -- | |
| tab:fix-benchmark | vit_in21k/imagenet1k | 78.5/83.2/83.5 | -- | |
| tab:fix-benchmark | dinov2_b14/domainnet | 58.2/68.5/70.8 | -- | |
| tab:fix-benchmark | clip_b16/domainnet | 56.8/66.8/68.5 | -- | |
| tab:bias-direction | dinov2_b14|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.344, CV 0.199; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | clip_b16|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.378, CV 0.202; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | mae_b16|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.591, CV 0.158; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | resnet50|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.468, CV 0.194; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | dinov2_b14|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.245, CV 0.199; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | clip_b16|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.275, CV 0.230; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | vit_in21k|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.510, CV 0.218; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:bias-direction | vit_in21k|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.409, CV 0.264; averaged-cos 1.0000 | averaged statistic is the gauge identity |
| tab:rate-fits | clip_b16 | (see paper) | slope 0.032 R2 0.02 n 176 | predicted -0.2849658285166929 |
| tab:rate-fits | dinov2_b14 | (see paper) | slope -0.009 R2 0.02 n 29 | predicted -0.26901119945613994 |
| tab:rate-fits | mae_b16 | (see paper) | slope 0.060 R2 0.01 n 177 | predicted -0.13705830438531777 |
| tab:rate-fits | resnet50 | (see paper) | slope 0.001 R2 0.00 n 170 | predicted -0.2174898333925899 |
| tab:rate-fits | vit_in21k | (see paper) | slope -0.170 R2 0.09 n 327 | predicted -0.25932189795255567 |
