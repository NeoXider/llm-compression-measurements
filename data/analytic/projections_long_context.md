Calibration (3090): e_w = 0.48, e_kv(F16) = 0.92; decode excess per token of depth: f16 0.00 ns, q8_0 205.72 ns, q4_0 253.64 ns; EXP3 148.19 ns (non-fused)

| Card | ctx | F16 tok/s | Q8_0 / F16 | Q4_0 / F16 | EXP3 lossless non-fused / fused | F16 fits? | max ctx F16 / Q8 / Q4 | confidence |
|---|---|---|---|---|---|---|---|---|
| RTX 3090 | 4K | 159 | 0.92x | 0.92x | 0.92x / 1.01x | yes | 132K / 249K / 471K | measured basis |
| RTX 3090 | 32K | 89 | 0.73x | 0.72x | 0.72x / 1.04x | yes | 132K / 249K / 471K | measured basis |
| RTX 3090 | 120K | 37 | 0.63x | 0.63x | 0.62x / 1.06x | yes | 132K / 249K / 471K | measured basis |
| RTX 5090* | 4K | 304 | 0.97x | 0.97x | 0.95x / 1.01x | yes | 185K / 349K / 659K | medium |
| RTX 5090* | 32K | 171 | 0.87x | 0.89x | 0.80x / 1.04x | yes | 185K / 349K / 659K | medium |
| RTX 5090* | 120K | 72 | 0.80x | 0.84x | 0.72x / 1.06x | yes | 185K / 349K / 659K | medium |
| L40S* | 4K | 147 | 1.00x | 1.02x | 0.97x / 1.01x | yes | 291K / 548K / 1036K | low |
| L40S* | 32K | 82 | 1.02x | 1.10x | 0.89x / 1.04x | yes | 291K / 548K / 1036K | low |
| L40S* | 120K | 35 | 1.03x | 1.18x | 0.84x / 1.06x | yes | 291K / 548K / 1036K | low |
| A100 80GB* | 4K | 346 | 0.68x | 0.63x | 0.73x / 0.79x | yes | 503K / 947K / 1789K | medium |
| A100 80GB* | 32K | 194 | 0.32x | 0.28x | 0.37x / 0.45x | yes | 503K / 947K / 1789K | medium |
| A100 80GB* | 120K | 82 | 0.23x | 0.20x | 0.27x / 0.34x | yes | 503K / 947K / 1789K | medium |
| H100 SXM* | 4K | 569 | 0.83x | 0.81x | 0.85x / 0.93x | yes | 503K / 947K / 1789K | medium |
| H100 SXM* | 32K | 319 | 0.52x | 0.49x | 0.56x / 0.76x | yes | 503K / 947K / 1789K | medium |
| H100 SXM* | 120K | 134 | 0.41x | 0.38x | 0.45x / 0.66x | yes | 503K / 947K / 1789K | medium |
| H200* | 4K | 815 | 0.76x | 0.73x | 0.80x / 0.87x | yes | 907K / 1708K / 3226K | medium |
| H200* | 32K | 457 | 0.42x | 0.38x | 0.47x / 0.60x | yes | 907K / 1708K / 3226K | medium |
| H200* | 120K | 192 | 0.31x | 0.28x | 0.36x / 0.48x | yes | 907K / 1708K / 3226K | medium |
| B200* | 4K | ? | ? | ? | ? | yes | 1245K / 2343K / 4427K | low |
| B200* | 32K | ? | ? | ? | ? | yes | 1245K / 2343K / 4427K | low |
| B200* | 120K | ? | ? | ? | ? | yes | 1245K / 2343K / 4427K | low |
