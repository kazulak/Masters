# Phase P6: UPMEM Cost-Guided Path Optimization — Results and Audit Evidence

This package contains the complete, canonical experimental evidence, model parameter progressions, physical execution manifests, and final readout analysis for the **Phase P6 Cost-Guided Path Optimization** campaign against the frozen UPMEM DPU execution engine.

---

## 1. Audit Anchors & Provenance

| Property | Value | Description |
| :--- | :--- | :--- |
| **Qualified Software Commit** | `2beea27411c16e90ed76988613ddb00bcc09f942` | Audited Git commit for all research software |
| **Software Git Tag** | `thesis-upmem-cost-guided-software-v1` | Immutable Git tag pinned to qualified commit |
| **Frozen Executor Commit** | `459935f586fdd16c82013838e6d27a12604c3093` | Ancestor commit containing verified T8 runtime binaries |
| **Target Hardware** | `safari-baguette1.ethz.ch` | ETH Zürich UPMEM testbed server |
| **UPMEM SDK Version** | `2023.1.0` | Exact pinned SDK version verified via `dpu-pkg-config` |
| **Python Runtime** | `3.10.12` | Strict CPython environment |
| **Hardware Rank** | `/dev/dpu_rank1` | Exclusively locked during all physical stages |

---

## 2. Campaign Summary & Physical Attempt Accounting

Under the formal contract established in Section 11 of `thesis/upmem-system-and-path-optimization-plan-v2.md`, physical hardware executions were strictly bounded to $\le 768$ attempts across four once-only stages:

| Stage | Budgeted Cap | Actual Executed | Valid / Qualified | Retries | Replacements | Physical Tarball SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Initial Calibration** | 192 | 192 | 192 (100%) | 0 | 0 | `2ef9fdfbf44fc68d9e1b712855ffdf362822ad16caf8cd6a3b73934791a769c8` |
| **Feedback Round 1** | 144 | 144 | 144 (100%) | 0 | 0 | `264ebe0a32cd2c8495025c4873454f638db56e52317409f9dc2d67790e5ec05d` |
| **Feedback Round 2** | 144 | 144 | 144 (100%) | 0 | 0 | `a82fa9302d634ae4758bcb67e7cb0a22440d3366094a44156406e7e04878cea5` |
| **Final Evaluation** | 288 | 198 | 198 (100%) | 0 | 0 | `b57f0a55c00becbf8a2b9349feea113e3c17244bdf20b6ef0edf1d88ed533e46` |
| **Total Campaign** | **$\le 768$** | **678** | **678 (100%)** | **0** | **0** | *All stages accepted canonically* |

---

## 3. Launch Cost Model Evolution

The 5-term launch-cost model `upmem_launch_cost_v1` evaluates lowered contraction plans using integer weights summing to 10:
$$\text{Cost}(\text{plan}) = \theta_H \frac{H}{S_H} + \theta_P \frac{P}{S_P} + \theta_N \frac{N}{S_N} + \sum_{l \in \text{launches}} \max_{d \in \text{DPUs}} \left( \theta_M \frac{M_{l,d}}{S_M} + \theta_W \frac{W_{l,d}}{S_W} \right)$$

Normalization scales $(S_H, S_P, S_N, S_M, S_W)$ were derived deterministically from the 12 training greedy circuits:
- $S_H = 1{,}087{,}544$ (Host transfer bytes)
- $S_P = 12{,}324{,}976$ (Host preparation facts)
- $S_N = 65$ (Kernel launch count)
- $S_M = 1{,}140{,}368$ (DPU MRAM-WRAM traffic)
- $S_W = 270{,}568$ (DPU real MAC operations)

### Fitted Weight Progression
1. **Uninformed Prior:** $[2, 2, 2, 2, 2]$ (Uniform 20% weight per term)
2. **Post-Initial Calibration:** $[2, 1, 7, 0, 0]$ (Kernel launch count dominated initial latencies)
3. **Post-Feedback Round 1:** $[2, 6, 1, 0, 1]$ (Host request preparation became primary factor)
4. **Post-Feedback Round 2 (Frozen Pretest):** $[1, 2, 1, 1, 5]$ (DPU compute work $W$ and host preparation balance)

---

## 4. Primary Readout Findings

The evaluation measured 4 contraction path generation methods across all 12 test cells (6 circuit families $\times$ 2 topologies):
- **$G$**: Baseline Greedy heuristic
- **$F$**: Cotengra FLOP-guided adaptive search
- **$R$**: FLOP-guided search reranked with the UPMEM cost model
- **$U$**: Cotengra adaptive search guided by the UPMEM cost model during generation

### Aggregated Performance Summary (Family-Balanced Geometric Ratios)

| Contrast | Metric | Ratio | 95% Bootstrap CI | Finding |
| :--- | :--- | :---: | :---: | :--- |
| **$R / U$** (Primary) | `session_inclusive_s` | **1.0013** | [0.9948, 1.0085] | Statistical parity with slight edge to $U$ (same path in 8/12 cells) |
| **$R / U$** (Primary) | `total_wall_s` | **1.0039** | [0.9963, 1.0130] | Pure kernel & transfer wall-time parity |
| **$F / U$** (FLOP vs. UPMEM) | `session_inclusive_s` | **1.0404** | [1.0319, 1.0534] | **UPMEM-guided is 4.0% faster overall** |
| **$F / U$** (4-DPU topology) | `session_inclusive_s` | **1.0818** | [1.0651, 1.1006] | **UPMEM-guided is 8.2% faster on 4-DPU** |
| **$F / U$** (4-DPU topology) | `total_wall_s` | **1.1154** | [1.0938, 1.1200] | **UPMEM-guided is 11.5% faster pure wall-time** |
| **$G / U$** (Greedy vs. UPMEM) | `session_inclusive_s` | **1.2665** | [1.2585, 1.2783] | **UPMEM-guided is 26.7% faster than Greedy** |
| **$G / U$** (4-DPU topology) | `total_wall_s` | **1.4315** | [1.4151, 1.4517] | **UPMEM-guided is 43.1% faster pure wall-time** |

---

## 5. Package Contents & Audit Verification

```
thesis_results/upmem_cost_guided_path_v1/
├── README.md                      # This document
├── SHA256SUMS                     # Checksums for all package files
├── readout/
│   ├── report.md                  # Executive readout report
│   ├── aggregates.csv             # Geometric ratios with 95% bootstrap intervals
│   ├── contrasts.csv              # Pairwise method comparisons per test cell
│   ├── methods.csv                # Detailed timing breakdown by cell, arm, and metric
│   ├── observations.csv           # Raw measurements across all accepted blocks
│   ├── search.csv                 # Cotengra search timings, proposals, and phase durations
│   ├── selected_plan_facts.csv    # Fact values (H, P, N, M, W) for selected candidate paths
│   ├── model_diagnostics.json     # Feature correlation matrix and tied grid configurations
│   ├── provenance.json            # Audit hashes of input study, manifests, and archives
│   ├── accepted_evaluation_rows.json.gz # Complete gzip-compressed accepted evaluation measurements
│   └── SHA256SUMS                 # Checksums for readout files
├── evidence_manifests/
│   ├── adapter-verification.json  # Preflight CPU & SDK simulator qualification report
│   ├── normalization.json         # Development-only normalization scales and greedy baselines
│   ├── initial_profile.json       # Uninformed prior profile
│   ├── initial_round.json         # Stage 1 candidate selection manifest
│   ├── initial.accepted.json      # Stage 1 canonical acceptance verification record
│   ├── initial_fit.json           # Stage 1 fitted model weights
│   ├── feedback_1_round.json      # Stage 2 candidate selection manifest
│   ├── feedback_1.accepted.json   # Stage 2 canonical acceptance verification record
│   ├── feedback_1_fit.json        # Stage 2 fitted model weights
│   ├── feedback_2_round.json      # Stage 3 candidate selection manifest
│   ├── feedback_2.accepted.json   # Stage 3 canonical acceptance verification record
│   ├── feedback_2_fit.json        # Stage 3 fitted model weights
│   ├── pretest_profile.json       # Gate A frozen model profile for evaluation
│   ├── evaluation_round.json      # Stage 4 candidate selection manifest
│   ├── evaluation.accepted.json   # Stage 4 canonical acceptance verification record
│   └── stage_archives.json        # Physical hardware tarball checksums, sizes, and attempt counts
└── tools/
    ├── p6_operator.sh             # Bounded operational execution harness
    ├── p6_readout.py              # Canonical readout and bootstrap calculation engine
    └── test_p6_readout.py         # Unit tests for readout logic (all 5 passing)
```

### Verification Command
To verify the integrity of the audit package:
```bash
cd thesis/implementation/thesis_results/upmem_cost_guided_path_v1
sha256sum -c SHA256SUMS
```
To run the synthetic readout tests:
```bash
python3 tools/test_p6_readout.py
```
