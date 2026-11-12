# Performance reports

| Report | Scope | Status |
| --- | --- | --- |
| [`gpu/PHASE2_PERFORMANCE_WORK.md`](gpu/PHASE2_PERFORMANCE_WORK.md) | RHOAIENG-102092 phase 2: RF-DETR Small on a Tesla T4 — binary tensor extension (**4.0× REST**), gRPC converter fix (**+19.5% C1, 3.94× total opt-in**), final campaign (159/159 gated trials) | **Final** |
| [`gpu/PHASE3_ROADMAP.md`](gpu/PHASE3_ROADMAP.md) | Post-phase-2 enhancement roadmap, including the measured adaptive-batcher collapse at high concurrency | Active planning |
| [`gpu/PR_DRAFT_rhoaieng-102092.md`](gpu/PR_DRAFT_rhoaieng-102092.md) | PR description for the implementation branch (`rhoaieng-102092`) | Ready for PR |
| [`gpu/GPU_PERFORMANCE_STUDY.md`](gpu/GPU_PERFORMANCE_STUDY.md) | Predecessor study: RF-DETR Small over REST and raw gRPC on one RTX 4070 — input representation, worker JSON, ZeroMQ and CUDA execution changes | Historical (phase 1) |
| [`cpu/PERFORMANCE_REDESIGN.md`](cpu/PERFORMANCE_REDESIGN.md) | REST/JSON and CPU ONNX inference, Python/transport experiments, single-instance OVMS comparison | Historical (phase 1) |

Each report states its workload and measured scope. The RTX 4070 and CPU
studies used different hardware and harness generations than the Tesla T4
phase-2 work; their numbers are not comparable. The evidence trail for every
number lives in [`../BENCHMARKING_HISTORY.md`](../BENCHMARKING_HISTORY.md) and
the session data indexed in [`../INDEX.md`](../INDEX.md).
