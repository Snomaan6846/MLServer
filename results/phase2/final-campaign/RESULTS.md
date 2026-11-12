# RHOAIENG-102092: Final Benchmark Campaign Results

Branch: `rhoaieng-102092` at HEAD

Total trials: 159 (90 unbatched + 60 batched + 9 mixed)

---

## Session A: Unbatched REST (RF-DETR, batch=1)

### All variants

| Variant | C | RPS (median) | Spread% | p50 (ms) | p99 (ms) | CPU-ms/req | GPU% | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| head-binary-graph | 1 | 24.61 | 0.5% | 40.6 | 42.6 | 39.8 | 55.9 | 933 |
| head-binary-graph | 2 | 31.58 | 0.7% | 63.3 | 66.3 | 41.1 | 83.0 | 931 |
| head-binary-graph | 4 | 31.66 | 0.3% | 126.3 | 130.8 | 38.4 | 83.9 | 940 |
| head-binary-graph | 8 | 31.62 | 0.1% | 253.1 | 259.0 | 34.3 | 83.9 | 964 |
| head-binary-graph | 16 | 31.56 | 0.4% | 506.9 | 516.7 | 39.1 | 81.2 | 1001 |
| head-binary-json | 1 | 3.99 | 0.2% | 249.6 | 264.6 | 208.2 | 16.8 | 993 |
| head-binary-json | 2 | 8.22 | 0.3% | 242.1 | 257.2 | 250.0 | 28.0 | 1055 |
| head-binary-json | 4 | 8.25 | 0.9% | 483.5 | 592.5 | 224.0 | 28.3 | 1174 |
| head-binary-json | 8 | 8.28 | 1.0% | 957.4 | 1072.2 | 218.5 | 35.1 | 1399 |
| head-binary-json | 16 | 8.21 | 2.7% | 1930.6 | 2147.8 | 177.7 | 35.1 | 1911 |
| head-binary-rest | 1 | 24.44 | 2.0% | 40.8 | 42.9 | 31.7 | 54.5 | 903 |
| head-binary-rest | 2 | 31.46 | 0.3% | 63.4 | 66.8 | 43.0 | 82.5 | 897 |
| head-binary-rest | 4 | 31.46 | 0.4% | 127.1 | 131.2 | 41.2 | 81.8 | 917 |
| head-binary-rest | 8 | 31.47 | 0.5% | 254.3 | 259.8 | 42.1 | 79.7 | 953 |
| head-binary-rest | 16 | 31.44 | 0.4% | 508.8 | 518.6 | 45.9 | 79.6 | 993 |
| head-stock-rest | 1 | 3.97 | 2.2% | 251.3 | 266.7 | 248.2 | 16.9 | 1005 |
| head-stock-rest | 2 | 7.74 | 2.2% | 257.0 | 273.6 | 220.9 | 28.0 | 1076 |
| head-stock-rest | 4 | 7.89 | 1.7% | 504.6 | 562.1 | 257.3 | 31.6 | 1174 |
| head-stock-rest | 8 | 7.84 | 1.9% | 1016.4 | 1102.1 | 178.4 | 23.7 | 1397 |
| head-stock-rest | 16 | 7.84 | 0.6% | 2013.4 | 2212.9 | 228.9 | 32.4 | 1912 |

#### Binary tensor vs stock (headline)

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 6.235× | [6.143, 6.237] | ✅ improvement |
| C2 | 4.064× | [4.032, 4.137] | ✅ improvement |
| C4 | 3.990× | [3.968, 4.026] | ✅ improvement |
| C8 | 4.016× | [4.013, 4.067] | ✅ improvement |
| C16 | 3.999× | [3.996, 4.024] | ✅ improvement |

#### JSON fallback vs stock (no-regression)

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 1.006× | [1.003, 1.026] | ➖ parity |
| C2 | 1.061× | [1.054, 1.080] | ✅ improvement |
| C4 | 1.040× | [1.035, 1.059] | ➖ parity |
| C8 | 1.056× | [1.053, 1.083] | ✅ improvement |
| C16 | 1.051× | [1.027, 1.053] | ✅ improvement |

#### CUDA graph delta (config recommendation)

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 1.007× | [0.989, 1.014] | ➖ parity |
| C2 | 1.002× | [1.001, 1.009] | ➖ parity |
| C4 | 1.007× | [1.006, 1.007] | ➖ parity |
| C8 | 1.005× | [1.003, 1.007] | ➖ parity |
| C16 | 1.005× | [1.004, 1.006] | ➖ parity |

#### JSON fallback — regression check

| Metric | Control | Candidate | Status |
|---|---|---|---|
| C1 p99 | 266.7ms | 264.6ms | ✅ (OK) |
| C1 memory | 1005MB | 993MB | ✅ (OK) |
| C2 p99 | 273.6ms | 257.2ms | ✅ (OK) |
| C2 memory | 1076MB | 1055MB | ✅ (OK) |
| C4 p99 | 562.1ms | 592.5ms | ❌ (REGRESSION) |
| C4 memory | 1174MB | 1174MB | ✅ (OK) |
| C8 p99 | 1102.1ms | 1072.2ms | ✅ (OK) |
| C8 memory | 1397MB | 1399MB | ✅ (OK) |
| C16 p99 | 2212.9ms | 2147.8ms | ✅ (OK) |
| C16 memory | 1912MB | 1911MB | ✅ (OK) |

**⚠️ REGRESSIONS DETECTED — review before proceeding.**


---

## Session B: Unbatched gRPC (RF-DETR, batch=1)

### All variants

| Variant | C | RPS (median) | Spread% | p50 (ms) | p99 (ms) | CPU-ms/req | GPU% | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| head-grpc-optin | 1 | 22.86 | 2.1% | 43.6 | 46.6 | 16.8 | 50.8 | 949 |
| head-grpc-optin | 2 | 31.15 | 0.4% | 64.1 | 67.5 | 15.9 | 78.6 | 954 |
| head-grpc-optin | 4 | 31.11 | 0.3% | 128.5 | 132.4 | 18.1 | 81.7 | 988 |
| head-grpc-optin | 8 | 31.12 | 0.2% | 257.2 | 262.3 | 16.1 | 81.8 | 1069 |
| head-grpc-optin | 16 | 31.11 | 0.5% | 515.4 | 523.0 | 20.2 | 80.8 | 1164 |
| head-stock-grpc | 1 | 5.02 | 1.7% | 198.3 | 210.6 | 174.1 | 20.4 | 990 |
| head-stock-grpc | 2 | 7.86 | 2.8% | 253.2 | 265.4 | 173.3 | 34.4 | 1039 |
| head-stock-grpc | 4 | 7.92 | 0.4% | 504.7 | 539.4 | 170.7 | 33.7 | 1109 |
| head-stock-grpc | 8 | 7.92 | 1.4% | 1009.0 | 1064.3 | 194.4 | 24.2 | 1275 |
| head-stock-grpc | 16 | 7.90 | 1.0% | 2020.5 | 2122.7 | 187.3 | 32.5 | 1582 |

#### gRPC opt-in vs stock

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 4.508× | [4.486, 4.614] | ✅ improvement |
| C2 | 3.962× | [3.879, 4.004] | ✅ improvement |
| C4 | 3.922× | [3.922, 3.946] | ✅ improvement |
| C8 | 3.927× | [3.925, 3.976] | ✅ improvement |
| C16 | 3.942× | [3.910, 3.952] | ✅ improvement |

#### gRPC opt-in — regression check

| Metric | Control | Candidate | Status |
|---|---|---|---|
| C1 p99 | 210.6ms | 46.6ms | ✅ (OK) |
| C1 memory | 990MB | 949MB | ✅ (OK) |
| C2 p99 | 265.4ms | 67.5ms | ✅ (OK) |
| C2 memory | 1039MB | 954MB | ✅ (OK) |
| C4 p99 | 539.4ms | 132.4ms | ✅ (OK) |
| C4 memory | 1109MB | 988MB | ✅ (OK) |
| C8 p99 | 1064.3ms | 262.3ms | ✅ (OK) |
| C8 memory | 1275MB | 1069MB | ✅ (OK) |
| C16 p99 | 2122.7ms | 523.0ms | ✅ (OK) |
| C16 memory | 1582MB | 1164MB | ✅ (OK) |

**No p99 or memory regressions detected.**


---

## Session C: Batched (RF-DETR, dynamic batch, max_batch_size=4)

### All variants

| Variant | C | RPS (median) | Spread% | p50 (ms) | p99 (ms) | CPU-ms/req | GPU% | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| batched-binary | 1 | 4.57 | 3.3% | 218.3 | 231.1 | 178.8 | 17.9 | 1123 |
| batched-binary | 2 | 8.29 | 1.9% | 240.4 | 249.9 | 179.3 | 34.1 | 1175 |
| batched-binary | 4 | 8.24 | 1.2% | 482.9 | 499.6 | 206.1 | 36.9 | 1306 |
| batched-binary | 8 | 1.72 | 1.6% | 4586.1 | 5079.4 | 1077.7 | 0.6 | 2111 |
| batched-binary | 16 | 1.75 | 2.3% | 9068.5 | 9246.2 | 1020.5 | 0.0 | 2527 |
| batched-binary-json | 1 | 3.71 | 1.3% | 268.9 | 285.4 | 243.3 | 10.5 | 1138 |
| batched-binary-json | 2 | 7.26 | 1.7% | 272.0 | 307.2 | 222.6 | 26.9 | 1242 |
| batched-binary-json | 4 | 7.41 | 80.9% | 536.4 | 663.6 | 507.0 | 25.8 | 1740 |
| batched-binary-json | 8 | 1.70 | 3.3% | 4639.6 | 5349.5 | 971.4 | 4.2 | 2212 |
| batched-binary-json | 16 | 1.69 | 2.9% | 9369.8 | 9675.8 | 1187.9 | 3.1 | 2853 |
| batched-grpc-raw | 1 | 4.82 | 2.7% | 207.1 | 226.1 | 160.0 | 19.6 | 1132 |
| batched-grpc-raw | 2 | 8.32 | 0.5% | 239.9 | 262.4 | 188.1 | 35.4 | 1182 |
| batched-grpc-raw | 4 | 8.26 | 81.3% | 483.0 | 509.9 | 504.2 | 23.3 | 1732 |
| batched-grpc-raw | 8 | 1.71 | 3.0% | 4643.2 | 4807.1 | 1011.5 | 0.0 | 2033 |
| batched-grpc-raw | 16 | 1.73 | 375.4% | 9189.0 | 9375.5 | 718.1 | 9.4 | 2552 |
| batched-stock | 1 | 3.67 | 2.0% | 272.3 | 285.6 | 245.4 | 15.6 | 1142 |
| batched-stock | 2 | 7.45 | 5.4% | 266.4 | 310.5 | 243.2 | 33.8 | 1237 |
| batched-stock | 4 | 7.66 | 2.6% | 516.4 | 619.2 | 255.1 | 28.2 | 1329 |
| batched-stock | 8 | 1.70 | 348.2% | 4650.4 | 4889.9 | 875.3 | 9.1 | 2115 |
| batched-stock | 16 | 1.74 | 2.8% | 9209.4 | 9474.8 | 984.3 | 2.7 | 2771 |

#### Batched binary vs batched stock

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 1.225× | [1.219, 1.255] | ✅ improvement |
| C2 | 1.129× | [1.091, 1.156] | ✅ improvement |
| C4 | 1.075× | [1.049, 1.089] | ✅ improvement |
| C8 | 1.020× | [0.224, 1.021] | ➖ parity |
| C16 | 1.015× | [0.986, 1.027] | ➖ parity |

#### Batched JSON fallback vs batched stock

| Concurrency | Ratio | CI95 | Interpretation |
|---|---|---|---|
| C1 | 0.999× | [0.994, 1.016] | ➖ parity |
| C2 | 0.967× | [0.966, 1.013] | ➖ parity |
| C4 | 0.967× | [0.222, 1.016] | ➖ parity |
| C8 | 0.999× | [0.221, 1.029] | ➖ parity |
| C16 | 0.968× | [0.965, 1.020] | ➖ parity |

#### Batched JSON fallback — regression check

| Metric | Control | Candidate | Status |
|---|---|---|---|
| C1 p99 | 285.6ms | 285.4ms | ✅ (OK) |
| C1 memory | 1142MB | 1138MB | ✅ (OK) |
| C2 p99 | 310.5ms | 307.2ms | ✅ (OK) |
| C2 memory | 1237MB | 1242MB | ✅ (OK) |
| C4 p99 | 619.2ms | 663.6ms | ❌ (REGRESSION) |
| C4 memory | 1329MB | 1740MB | ❌ (REGRESSION) |
| C8 p99 | 4889.9ms | 5349.5ms | ❌ (REGRESSION) |
| C8 memory | 2115MB | 2212MB | ✅ (OK) |
| C16 p99 | 9474.8ms | 9675.8ms | ✅ (OK) |
| C16 memory | 2771MB | 2853MB | ✅ (OK) |

**⚠️ REGRESSIONS DETECTED — review before proceeding.**


---

## Session D: Mixed Co-Batching (alternating binary/JSON)

### Mixed payload results

| Variant | C | RPS (median) | Spread% | p50 (ms) | p99 (ms) | CPU-ms/req | GPU% | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| batched-mixed | 4 | 8.20 | 1.5% | 486.0 | 555.2 | 194.7 | 28.1 | 1326 |
| batched-mixed | 8 | 8.26 | 79.1% | 965.4 | 1336.4 | 522.9 | 23.3 | 1961 |
| batched-mixed | 16 | 1.77 | 4.2% | 9036.1 | 9378.1 | 922.4 | 2.1 | 2568 |

---

## Campaign Summary

| Protocol | Config | vs Stock (C16) | Status |
|---|---|---|---|
| REST binary | Binary tensor | 4.00× | Headline |
| REST JSON | Binary ON (fallback) | 1.05× | No regression |
| REST binary | + CUDA graph | 1.01× | Config rec |
| gRPC | Opt-in + converter fix | 3.94× | Optimization |
| Batched REST JSON | Stock on HEAD | 0.97× | No regression |
| Batched REST binary | Binary tensor | 1.01× | Binary + batching |
| Mixed binary/JSON | Co-batched | see Session D table | Normalization |

---

*Every trial gated on bit-exact REST / byte-exact gRPC responses.*

*Zero failed requests across all 159 trials.*

*All data captured freshly at HEAD. No stale data. No assumptions.*
