Cell format: baseline (F16 KV, Q4_K_M weights) total tok/s -> speedup for KV lossless 0.907 / KV Q8_0 / KV Q4_0 / W 0.934 + KV 0.907; '(C)' = compute-bound; 'fits'/'no' = whether the variant fits when F16 does not.

### 8B-class model (Q4_K_M weights 4.8 GB)

| Card | BW GB/s | VRAM GB | B=1, 8K | B=1, 128K | B=16, 32K | B=64, 8K | Max KV tokens F16 / lossless 0.907 / Q8 / Q4 |
|---|---|---|---|---|---|---|---|
| RTX 3090 | 936 | 24 | 79 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 21 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | 134K / 148K / 253K / 478K |
| RTX 5090 | 1792 | 32 | 151 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 41 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | F16 KV: does not fit -> no / no / fits / no | F16 KV: does not fit -> no / no / fits / no | 195K / 215K / 368K / 695K |
| L40S | 864 | 48 | 73 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 20 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | F16 KV: does not fit -> no / fits / fits / no | F16 KV: does not fit -> no / fits / fits / no | 317K / 350K / 598K / 1129K |
| RTX PRO 6000 Blackwell | 1792 | 96 | 151 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 41 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | 195 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 779 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 683K / 754K / 1287K / 2431K |
| A100 80GB | 2039 | 80 | 172 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 46 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | 222 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 887 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 561K / 619K / 1057K / 1997K |
| H100 SXM | 3350 | 80 | 283 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 76 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | 364 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 1457 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 561K / 619K / 1057K / 1997K |
| H200 | 4800 | 141 | 405 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 109 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | 522 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 2088 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 1027K / 1132K / 1933K / 3652K |
| B200 | 8000 | 192 | 675 tok/s -> 1.02x / 1.09x / 1.15x / 1.08x | 182 tok/s -> 1.08x / 1.58x / 2.28x / 1.10x | 870 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 3480 tok/s -> 1.10x / 1.78x / 3.04x / 1.10x | 1416K / 1561K / 2666K / 5036K |

### 70B-class model (Q4_K_M weights 42.4 GB)

| Card | BW GB/s | VRAM GB | B=1, 8K | B=1, 128K | B=16, 32K | B=64, 8K | Max KV tokens F16 / lossless 0.907 / Q8 / Q4 |
|---|---|---|---|---|---|---|---|
| RTX 3090 | 936 | 24 | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | weights do not fit |
| RTX 5090 | 1792 | 32 | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | weights do not fit |
| L40S | 864 | 48 | 10 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | 12K / 13K / 23K / 44K |
| RTX PRO 6000 Blackwell | 1792 | 96 | 20 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | 10 tok/s -> 1.05x / 1.31x / 1.57x / 1.09x | F16 KV: does not fit -> no / no / fits / no | F16 KV: does not fit -> no / no / fits / no | 158K / 175K / 299K / 564K |
| A100 80GB | 2039 | 80 | 23 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | F16 KV: does not fit -> no / fits / fits / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | 110K / 121K / 207K / 391K |
| H100 SXM | 3350 | 80 | 37 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | F16 KV: does not fit -> no / fits / fits / no | F16 KV: does not fit -> no / no / no / no | F16 KV: does not fit -> no / no / no / no | 110K / 121K / 207K / 391K |
| H200 | 4800 | 141 | 53 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | 28 tok/s -> 1.05x / 1.31x / 1.57x / 1.09x | F16 KV: does not fit -> no / fits / fits / no | F16 KV: does not fit -> no / fits / fits / no | 296K / 326K / 557K / 1053K |
| B200 | 8000 | 192 | 89 tok/s -> 1.01x / 1.03x / 1.04x / 1.07x | 47 tok/s -> 1.05x / 1.31x / 1.57x / 1.09x | F16 KV: does not fit -> no / fits / fits / no | F16 KV: does not fit -> no / fits / fits / no | 451K / 498K / 850K / 1606K |

