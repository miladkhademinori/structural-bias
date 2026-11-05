# Printed vs reproduced

| table | cell | printed | reproduced | note |
|---|---|---|---|---|
| tab:fix-benchmark | dinov2_b14/cifar100 | 82.3/86.5/86.8 | 98.2/51.3/99.9 | |
| tab:fix-benchmark | dinov2_b14/imagenet100 | 80.5/84.7/85.0 | -- | |
| tab:fix-benchmark | dinov2_b14/imagenet1k | 76.8/81.8/82.3 | -- | |
| tab:fix-benchmark | clip_b16/cifar100 | 81.5/85.7/86.0 | 88.1/43.0/97.5 | |
| tab:fix-benchmark | clip_b16/imagenet100 | 79.2/83.5/83.8 | -- | |
| tab:fix-benchmark | mae_b16/imagenet100 | 65.3/77.2/78.5 | -- | |
| tab:fix-benchmark | resnet50/imagenet100 | 62.1/73.8/75.2 | -- | |
| tab:fix-benchmark | vit_in21k/imagenet1k | 78.5/83.2/83.5 | -- | |
| tab:fix-benchmark | dinov2_b14/domainnet | 58.2/68.5/70.8 | -- | |
| tab:fix-benchmark | clip_b16/domainnet | 56.8/66.8/68.5 | -- | |
| tab:bias-direction | dinov2_b14|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.314, CV 0.186; averaged-cos 0.9999 | averaged statistic is the gauge identity |
| tab:bias-direction | clip_b16|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.304, CV 0.175; averaged-cos 0.9998 | averaged statistic is the gauge identity |
| tab:bias-direction | mae_b16|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.350, CV 0.183; averaged-cos 0.9996 | averaged statistic is the gauge identity |
| tab:bias-direction | resnet50|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.319, CV 0.171; averaged-cos 0.9995 | averaged statistic is the gauge identity |
| tab:bias-direction | dinov2_b14|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.227, CV 0.189; averaged-cos 0.9998 | averaged statistic is the gauge identity |
| tab:bias-direction | clip_b16|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.226, CV 0.199; averaged-cos 0.9996 | averaged statistic is the gauge identity |
| tab:bias-direction | vit_in21k|cifar100 | cos 0.92-0.98 (natural) | per-pair cos 0.321, CV 0.174; averaged-cos 0.9996 | averaged statistic is the gauge identity |
| tab:bias-direction | vit_in21k|imagenet_r | cos 0.92-0.98 (natural) | per-pair cos 0.246, CV 0.198; averaged-cos 0.9994 | averaged statistic is the gauge identity |
| tab:rate-fits | clip_b16 | (see paper) | slope -0.009 R2 0.00 n 260 | predicted -0.28968468004672143 |
| tab:rate-fits | dinov2_b14 | (see paper) | slope 0.003 R2 0.00 n 64 | predicted -0.22812758894826576 |
| tab:rate-fits | mae_b16 | (see paper) | slope -0.045 R2 0.01 n 177 | predicted -0.13705830438531777 |
| tab:rate-fits | resnet50 | (see paper) | slope 0.021 R2 0.00 n 180 | predicted -0.21989554382956542 |
| tab:rate-fits | vit_in21k | (see paper) | slope -0.176 R2 0.16 n 349 | predicted -0.25932189795255567 |