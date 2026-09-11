# P6 execution runbook — operational appendix to the existing v2 plan

**Prepared 10 September 2026. Source-bound to `2beea27411c16e90ed76988613ddb00bcc09f942`.**

**File rule:** this download contains an operational appendix, not a replacement for the repository's complete `thesis/upmem-system-and-path-optimization-plan-v2.md`. Keep the repository filename and Sections 1–14. Keep this appendix and its helper scripts outside the tracked checkout during the campaign; integrate reporting documentation afterward. Do not overwrite the complete repository plan with this appendix.

## P6.0 — Decision, scope, and completion conditions

The default implementation budget is **zero changes to production source**. The audited branch already has the cost function, metadata extractor, adaptive cotengra/Optuna loop, deterministic fitter, frozen-path replay, packet preparation, raw-evidence acceptance, remote handoff, and once-only physical stage invocation. This runbook operates those components. It does not implement a second optimizer or runner. [S1–S7]

The three supplied operator-side files are optional mechanical conveniences, not repository changes: `p6_operator.sh` spells out the existing CLI/SSH/transfer calls; `p6_readout.py` invokes existing verification before producing final descriptive tables; `test_p6_readout.py` exercises its reporting mathematics on synthetic rows. Their complete source is included in the appendices below. Keep them under the durable operator directory, not `src/` or `native/`.

There are two completion gates:

**Evaluation-ready:** exact-source CPU/SDK adapter qualification accepted; initial and both bounded feedback stages accepted or validly declared empty; final coefficients/scales/search settings fixed; all 24 final F/U traces complete; all 12 G/F/R/U cell mappings frozen; exact selected-path CPU qualification passes; final physical packet and handoff are hash-bound. No final physical timing has been observed yet.

**Milestone-complete:** the frozen final evaluation is executed at most once, both raw copies pass canonical acceptance, the final readout and numerical outcomes are produced, and the portable results package has verified copies on two hosts. Positive speedup is not a completion criterion.

### Immutable study identity

| Item | Required value |
|---|---|
| Branch | `feature/upmem-final-system-path-search-v2` |
| Audited study source | `2beea27411c16e90ed76988613ddb00bcc09f942` |
| Executor source | `459935f586fdd16c82013838e6d27a12604c3093` |
| Executor tag | `thesis-upmem-kernel-schedule-system-v1` |
| Runtime SHA-256 | `b7168cd09f007978622346fd9954bdda54beb9ce48d870e4df8d158ed8681c02` |
| Workload SHA-256 | `f2c85f27508d9b9fb5f55d13b89334939aa424330e27555dfafe3216452382d7` |
| Existing study file | `configs/upmem_cost_guided_path_study_v1.json` |
| Current controlling contract | Section 11 of the existing repository v2 plan |
| Cost model | `upmem_launch_cost_v1` |
| Numerical policy | `split_complex_float32_v1` |
| Schedule / transport | `static_dag_waves_v1` / `packed_wave_v1` |
| Complex / geometry policy | Fusion when admitted / `panel_only_v1` |
| Intermediates | Existing host-roundtrip policy |
| Resources | One rank, 1 DPU/T8 and 4 DPUs/T8 |
| Research libraries | Python 3.10.12, cotengra 0.7.5, Optuna 4.5.0; all packages in the existing research lock |
| SDK | Existing ETH SDK 2023.1.0 |
| Latest independently checked CI | Successful GitHub Actions run `34370052538` at the audited study source |

The record's stale `software_implementation_pending` text is not a reason to rewrite the optimizer. Do not edit that JSON merely to improve its status label before starting: it is part of the study hash. Runtime, kernel, scheduler, precision, resource, and historical evidence work remain closed. [S1, S2, S8]

### Workload and budget

The manifest—not hand-entered circuit names—is authoritative. It contains BB84, BV, EDC, HS, QRNG, and XOR, with one training and one distinct test instance per family and two topologies per instance. The inference is instance/size transfer within six represented families, not family-held-out generalization. [S2]

| Stage | Maximum unique executed paths per cell | Warmups + measurements | Cells | Maximum attempts |
|---|---:|---:|---:|---:|
| Initial | 4 | 1 + 3 | 12 development | 192 |
| Feedback 1 | 3 | 1 + 3 | 12 development | 144 |
| Feedback 2 | 3 | 1 + 3 | 12 development | 144 |
| Evaluation | 4 method roles before deduplication | 1 + 5 | 12 test | 288 |
| **Maximum** | | | | **768** |

The previous 792 authorization leaves 24 **unallocated** attempts. They are not a retry or refill reserve. Role coincidences and empty feedback reduce actual use. There is no extra training-confirmation stage, validation campaign, int8 campaign, or tuning run in this runbook.

The source-specific adapter adds **two CPU observations, zero CPU native sessions, and four SDK-simulator observations/sessions**. These are not physical calibration attempts. Each actual round additionally receives one CPU-reference replay per distinct selected circuit/path, not per duplicate topology. No extra hardware smoke run is introduced. [S3, S4]

## P6.1 — SMART checkpoints

Here “time-bound” means an explicit finite operation count or the existing timeout, not a promised elapsed delivery time. Never extend a limit because the measured result is disappointing.

| Step | Specific action | Measurable exit | Relevance / achievable scope | Bound and stop rule |
|---|---|---|---|---|
| 1 | Bind the actual local/ETH worktrees, interpreter paths, source, and frozen binaries | Source/lock binding equality; executor tag and binary digests match; `inspect` reports 768 | Makes existing code executable without scientific redesign | One read-only binding pass. A mismatch is deployment repair, not automatic source modification |
| 2 | Qualify the small frozen-path adapter | Exactly 2 CPU observations/0 sessions and 4 SDK observations/4 sessions; all raw qualification checks pass | Proves the new path-input/evidence boundary, not the old executor again | One source-specific fixture invocation per target; existing 120-second attempt limit |
| 3 | Initialize scales and collect development search traces | 12 greedy scale inputs; 12 complete 128-proposal initial traces | Establishes immutable model inputs and a diverse initial physical set | 128 proposals/cell; 300 seconds/proposal, 60 seconds/lowering, 7,200 seconds/search |
| 4 | Freeze and prepare the initial stage | Exact initial manifest, at most 192 attempts; selected-path CPU replay; exported handoff | Establishes the initial physical calibration packet | One freeze; no manual candidate additions or refills |
| 5 | Execute, retrieve, and accept the initial stage | Physical/canonical success, exact sample/session set, two matching accepted archives | First real hardware-to-model data | One invocation; rank admission wait at most 15 minutes; 120 seconds/attempt |
| 6 | Fit once through the accepted initial stage | 1,001-row grid; integer coefficient tuple summing to 10; source and observation hashes | Provides the first physically fitted objective | One complete deterministic grid enumeration; no ETH calls |
| 7 | Execute feedback 1 and feedback 2 in order | Each has complete traces, a frozen manifest, accepted raw data or trace-proven empty feedback, and one fit | Completes the specified feedback-guided search POC | At most two feedback rounds, at most 144 attempts each; no extra round |
| 8 | Freeze the final model | `pretest_profile.json` binds the final fit, inputs, search settings, and 12 test cells | Prevents evaluation-driven changes | One exclusive write; no fitting afterward |
| 9 | Generate and freeze all final paths | 24 F/U traces, exactly 128 proposals each; 12 G/F/R/U role maps; at most 288 attempts | Separates guided generation from conventional-pool reranking | Finish every final selection before any final physical timing |
| 10 | Prepare and verify final evaluation handoff | Source-specific adapter and all selected CPU replays valid; final packet and handoff immutable | **Evaluation-ready gate** | One final prepared packet; no performance-based candidate substitution |
| 11 | Run and accept final evaluation | Exactly the frozen deduplicated observations; full accuracy/replay/source/resource checks; two verified raw copies | Obtains final experimental outcome | One invocation; cumulative physical child-process time at most 86,400 seconds |
| 12 | Produce readout and portable package | Per-cell tables, paired contrasts, geometric summaries, trace diagnostics, profile and raw provenance, checksums on two hosts | **Milestone-complete gate** | One non-adaptive analysis using the fixed readout convention |

Do not rerun the 2,489-test suite merely to recreate a number already established at this exact source. Reuse its record and successful exact-HEAD CI. A changed study source requires the focused regressions plus the normal full suite/Ruff/CI. Actual CPU/SDK adapter evidence remains necessary even when CI is green. [S1]

## P6.2 — Mathematical contract — implemented, not a request to reimplement

### P6.2.1 — Object being optimized

For fixed circuit/TN input and complete pre-measurement statevector contract `c`, a candidate complete contraction path `p`, and the retained executor profile `e`, use

\[
\Pi_e(c,p)=\operatorname{Lower}_e(c,p),\qquad
C_\theta(c,p\mid e)=C_\theta(\Pi_e(c,p)).
\]

Only the path and, during development, model coefficients vary. A path is lowered through the real scheduling, tiling, fusion and admission logic. The model does not choose a new scheduler, invent residency, or change DPU counts. Neither a local tree-node surrogate nor a different annealing algorithm is added.

### P6.2.2 — Five feature families and fixed normalization

`H` is planned aligned H2D+D2H bytes; `P` is the documented named host array-pass volume proxy; `N` is the implemented coordination count; `M[l,d]` is estimated local MRAM↔WRAM traffic for DPU d in physical launch l; `W[l,d]` is the real multiply-accumulate count on that DPU in that launch.

Four-product complex fusion changes launch/transfer organization, not the total four-product arithmetic. Work assigned to a DPU must not be divided by the DPU count again. Do not introduce a speculative tasklet-speedup division. Local traffic and host passes are model proxies, not measured bus counters or exact instruction counts. The documented omitted host passes remain omitted for this milestone. [S5, S7]

For development-greedy plans only,

\[
X(g_j)=\left(H,P,N,\sum_{l,d}M[l,d],\sum_{l,d}W[l,d]\right),\quad
s_k=\max\left(1,\operatorname{median}_{j\in D}X_k(g_j)\right).
\]

Freeze these scales once. Do not normalize per candidate, topology, new round, or test workload.

\[
C_\theta(\Pi)=\theta_HH/s_H+\theta_PP/s_P+\theta_NN/s_N+
\sum_l\max_d\left(\theta_MM[l,d]/s_M+\theta_WW[l,d]/s_W\right).
\]

For slots `(M,W)=(10,0),(0,10)`, unit scales, and half weights on M/W, one concurrent launch costs `max(5,5)=5`, not 10. With compute-only weights, concurrent work `(4,4)` costs 4, whereas two serial launches cost 8. These are mathematical regression fixtures, not physical measurements.

The scalar is dimensionless. There is no conversion to seconds, no promise of globally optimal paths, and no claim that fitted coefficients are unique physical constants.

### P6.2.3 — Fitting

Enumerate

\[
\mathcal K=\{(k_H,k_P,k_N,k_M,k_W)\in\mathbb Z_{\ge0}^5:\sum_i k_i=10\},
\quad |\mathcal K|=\binom{14}{4}=1001,
\quad \theta_i=k_i/10.
\]

Initial tuple: `(2,2,2,2,2)`.

For each raw observation, primary time is computed **before** summarizing:

\[
T=t_{\rm open}+t_{\rm steady}+t_{\rm close}.
\]

For a measured eligible path p in development cell j,

\[
\ell_{jp}=\operatorname{median}_{(r,b)\in O_{jp}}
\bigl(\log T_{j,G,r,b}-\log T_{j,p,r,b}\bigr).
\]

The control is from the same cell, round and measured block. Warmups are not fit inputs. Missing controls are errors, not imputation opportunities.

For each tuple, minimize `(cost, path_id)` over physically measured eligible development paths, then maximize

\[
J(\theta)=\sum_j a_j\ell_{j,p_j(\theta)},\qquad a_j=1/(F n_{f(j)}).
\]

Tie-break: greatest `round(J,12)`, greatest worst-cell log-speedup, least `sum((k_i-2)^2)`, lexicographically smallest integer tuple. The existing fitter implements this. Do not insert another optimizer for coefficients.

`exp(J)` is a family-balanced geometric **training** score on the measured pool. It is not held-out performance, not a seconds predictor, and not proof of generalization.

### P6.2.4 — Search feedback

There are 128 serial asks per cell per search call, with 16 startup trials. Parameters are `costmod ∈ [0.1,4.0]` (linear) and `temperature ∈ [0.001,1]` (logarithmic). Every ask uses one fresh `RandomGreedyOptimizer(max_repeats=1, accel=False, parallel=False, simplify=True, seed=...)`. Complete-plan scoring is followed by `tell` **before the next ask**. [S3]

This is UPMEM-guided generator-parameter search; it is not literal simulated annealing of local tree mutations. No new tree engine is needed. The same master seed 20260909 and hash-based domain separation pair F and U; objective/profile identity does not change the paired proposal-seed schedule.

Known deterministic infeasibility consumes its proposal. A duplicate consumes its proposal. Unexpected exceptions and timeouts abort the trace. The loop must not continue until it accumulates 128 distinct or successful paths.

### P6.2.5 — Initial and feedback selection

Initial selection includes G, the conventional trace's FLOP-best F and uniform-model reranked R. Duplicate roles share a path. Fill only to four paths using the existing farthest-first L1 rule on `X/s`, with path-ID ties.

Feedback selects the lowest-score previously unmeasured path, one diverse additional path when available, and a fresh G control. No new candidate means the cell is skipped with the required trace proof. Whole empty feedback stages still have a manifest and acceptance record, but no execution packet and zero physical attempts.

### P6.2.6 — Evaluation and fixed readout convention

G is greedy; F is FLOP-best from its 128-proposal conventional trace plus G; R is lowest final UPMEM score from **that same F trace** plus G; U is lowest score from a separate 128-proposal UPMEM-guided trace plus G. Only after these separate selections may their union be used to deduplicate physical execution.

The repository requires medians/MADs, paired intervals and geometric summaries but does not make a newly written reporter a prerequisite to search. The supplied readout uses this explicit convention; freeze its bytes before the final timing:

\[
S_j^{A/B}=\frac{\operatorname{median}_{b=1}^5T_{j,A,b}}
{\operatorname{median}_{b=1}^5T_{j,B,b}},\qquad
S^{A/B}=\exp\left(\sum_j a_j\log S_j^{A/B}\right).
\]

`A/B > 1` favors B. Primary contrast is **R/U**, primary time is session-inclusive. Secondary contrasts are F/U, G/U, F/R, G/R and G/F; steady time is secondary. Report each topology separately as well as the family-balanced combined view. Raw MAD is `median(abs(T - median(T)))`, without an unannounced consistency multiplier.

For each of 10,000 bootstrap draws with seed 20260910, sample five complete block indices with replacement. Use the **same sampled indices for all methods and cells**, recompute arm medians and their ratios, then recompute the geometric aggregate. The 2.5th/97.5th percentiles are descriptive paired-block intervals. Families and circuits are fixed, not resampled. Five timing blocks and one paired search-seed schedule do not quantify broad circuit-population or optimizer-seed uncertainty.

When two methods choose the same physical path, their ratio is exactly one; their labels are not independent observations. Never construct missing method samples by rerunning that path separately.

The readout also emits search costs, distinct proposal counts, rejected/duplicate counts, post-startup F/U parameter differences, selected-plan facts, and descriptive model identifiability diagnostics. A real cell may legitimately have identical F/U traces or selected paths. Do not force a difference to make the POC look successful.

There is no new minimum-speedup adoption threshold. Lower intervals, regressions, and neutral outcomes are reported, not used to authorize another round.

### P6.2.7 — Numerical and timing boundaries

Use `default_validation_policy()` and its identity from this exact source. Do not type a new tolerance into YAML or a wrapper. The existing runner and raw verifier enforce full-precision accuracy, same-policy replay, resource and binary identities, and complete statevector semantics. This milestone is float32 only.

Session-inclusive time is not complete job time. The recorded search wall time also does not necessarily include circuit construction, every qualification operation, or the whole tuning campaign. Report those boundaries rather than declaring end-to-end acceleration. A conditional amortization estimate is meaningful only when matched execution savings are positive and the extra preprocessing/search cost has actually been measured:

\[
N_{\rm break-even}>\frac{S_{\rm new}-S_{\rm reference}}
{T_{\rm reference}-T_{\rm new}}.
\]

Do not manufacture a number when those quantities do not have compatible boundaries. No amortization estimate is required to finish the POC.

## P6.3 — Installation and deployment binding — one-time operator inputs

The repository cannot supply the local machine's authorized SSH alias, the actual ETH checkout location, or its interpreter path. Those are the only deployment decisions required. They are not scientific choices. Use the values already known in the ETH-enabled Codex workspace; do not guess an unconfigured connection.

The workflow has a **local coordinator** and the existing **single ETH hardware controller**. Offline path search and fitting happen locally. Small CPU/SDK qualification is run remotely so the source-bound packet sees the same binary paths and research environment as deployment. Only the existing `execute-stage` command invokes physical hardware.

### P6.3.1 — Locate the active local branch without switching or merging

Run in the local terminal:

```bash
set -Eeuo pipefail
export SOURCE=2beea27411c16e90ed76988613ddb00bcc09f942
export BRANCH=feature/upmem-final-system-path-search-v2
LOCAL_WORKTREE="$(python3 - <<'PY'
import os,subprocess
from pathlib import Path
repo=Path.home()/"repos/Masters"
text=subprocess.check_output(["git","-C",str(repo),"worktree","list","--porcelain"],text=True)
matches=[]
for block in text.strip().split("\n\n"):
    values=dict(line.split(" ",1) for line in block.splitlines() if " " in line)
    if values.get("branch")=="refs/heads/"+os.environ["BRANCH"]:
        matches.append(values["worktree"])
if len(matches)!=1:
    raise SystemExit(f"Expected one active worktree for the branch, found {matches}. Do not substitute main.")
print(Path(matches[0]).resolve())
PY
)"
export LOCAL_WORKTREE
export I="$LOCAL_WORKTREE/thesis/implementation"
cd "$I"
test "$(git rev-parse HEAD)" = "$SOURCE"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor 459935f586fdd16c82013838e6d27a12604c3093 "$SOURCE"
git rev-parse 'thesis-upmem-kernel-schedule-system-v1^{commit}'
```

The last command must print `459935f586fdd16c82013838e6d27a12604c3093`. Missing local tag metadata can be fetched read-only with `git fetch origin tag thesis-upmem-kernel-schedule-system-v1`; it does not justify a merge or checkout reset.

### P6.3.2 — Bind paths once

Choose an existing accepted operator directory when continuing an already initialized study. Before choosing a fresh directory, inspect the actual current worktree's ignored P6 state and the global remote invocation ledger. A new directory must not be used to bypass a previous invocation.

For a genuinely unstarted study:

```bash
export RUN="$HOME/evidence/upmem-cost-guided-path-v1/$SOURCE"
export PY="$I/runs/p6-preparation/cost-guided-research/.venv/bin/python"
export ETH=tkazulak@safari-baguette1
export RRUN="/home/tkazulak/evidence/upmem-cost-guided-path-v1/$SOURCE"
# Set these two paths to the existing authorized ETH checkout/interpreter.
# Shell prompts prevent an invented path being silently accepted.
read -r -p 'Absolute resolved thesis/implementation directory on ETH: ' RIMPL
read -r -p 'Absolute pinned research Python executable on ETH: ' RPY
export RIMPL RPY
mkdir -p "$RUN/tools" "$RUN/logs"
test -x "$PY"
```

`RIMPL` must be the resolved physical directory, not a symlink alias. The SSH login shell must already provide the accepted SDK. A missing SDK command is an environment problem; do not install a different SDK.

Extract the delivered ZIP outside the checkout and copy its operator files. The command below assumes the browser saved the ZIP in `~/Downloads`; change only `BUNDLE` when the actual download location differs. This is a filesystem input, not a research parameter.

```bash
BUNDLE="$HOME/Downloads/upmem-path-search-runbook.zip"
test -f "$BUNDLE"
test ! -e "$RUN/delivery"
unzip -q "$BUNDLE" -d "$RUN/delivery"
(cd "$RUN/delivery/p6_runbook" && sha256sum -c SHA256SUMS)
for name in p6_operator.sh p6_readout.py test_p6_readout.py upmem-system-and-path-optimization-plan-v2.md; do
    test ! -e "$RUN/tools/$name"
    cp "$RUN/delivery/p6_runbook/$name" "$RUN/tools/$name"
done
```

Do not place these files inside the tracked checkout. Set `TOOLS="$RUN/tools"` and record deployment values:

```bash
export TOOLS="$RUN/tools"
for name in I PY RUN ETH RIMPL RPY RRUN SOURCE TOOLS; do
    printf 'export %s=%q\n' "$name" "${!name}"
done > "$RUN/deployment.env"
chmod 600 "$RUN/deployment.env"
bash -n "$TOOLS/p6_operator.sh"
(cd "$TOOLS" && PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest -v test_p6_readout.py)
sha256sum "$TOOLS/p6_operator.sh" "$TOOLS/p6_readout.py" "$TOOLS/test_p6_readout.py" \
    > "$RUN/logs/operator-tools.sha256"
```

These tests use synthetic data. They are not adapter, SDK, or hardware qualification.

If the isolated research environment is absent, do not modify the historical CI constraints or executor environment. Create a separate environment with the recorded Python and install the exact existing lock:

```bash
# Only when the existing pinned environment is genuinely absent.
# Run in the implementation directory using an installed Python 3.10.12.
python3.10 -c 'import sys; assert sys.version_info[:3] == (3,10,12)'
python3.10 -m venv "$HOME/.venvs/upmem-p6-$SOURCE"
"$HOME/.venvs/upmem-p6-$SOURCE/bin/python" -m pip install -r requirements-upmem-path-search.txt
"$HOME/.venvs/upmem-p6-$SOURCE/bin/python" -m pip check
```

Set `PY` to that interpreter and update `deployment.env` **before** initializing any study state. Apply the same isolated-install procedure on ETH only when necessary, using its existing Python 3.10.12. Do not loosen pinned versions to make installation succeed. A missing pin is a reproducibility/deployment blocker to document.

### P6.3.3 — Verify the exact environment and binary binding

```bash
source "$RUN/deployment.env"
export PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
bash "$TOOLS/p6_operator.sh" check
```

This calls the repository's existing `inspect` and `research_binding`, compares local and ETH source/dependency bindings, requires the recorded Python/SDK, and checks all three frozen binary hashes. It does not allocate a DPU.

Expected `inspect` counts are 192/144/144/288, maximum 768, and 24 unallocated. The local and remote binding JSON files are retained in `logs/`.

Record the existing validation policy without altering it:

```bash
cd "$I"
PYTHONPATH=src "$PY" - <<'PY' > "$RUN/logs/validation-policy.json"
import json
from quantum_bench.experiment import default_validation_policy, default_validation_policy_id
print(json.dumps({"id":default_validation_policy_id(),"policy":dict(default_validation_policy())},indent=2))
PY
```

Do not rebuild the native binaries on the study commit: the expected hashes are the retained binaries built for the frozen executor. Restore exact retained binary bytes when missing. Debug information can make rebuilt hashes differ even when arithmetic source is unchanged.

## P6.4 — Step 2 — qualify the existing small adapter

```bash
bash "$TOOLS/p6_operator.sh" adapter
```

This expands only to existing operations:

```bash
"$PY" scripts/upmem_cost_guided_path.py write-adapter-packets \
    --output "$RUN/qualification" --execution-root "$RIMPL"
# Transfer the packet unchanged to ETH.
# On ETH, for cpu and sdk separately:
"$RPY" -m quantum_bench.cli run \
    --config "$RRUN/qualification/TARGET/preregistration/physical.yml" \
    --output "$RRUN/qualification/TARGET/raw"
"$RPY" -m quantum_bench.cli verify --input "$RRUN/qualification/TARGET/raw"
# Retrieve and call the existing verify_qualification for each target.
```

The uppercase `TARGET` lines above illustrate the wrapper's expansion; the executable wrapper already substitutes `cpu` and `sdk`. They are not additional manual commands to execute a second time.

The filenames `physical.yml` are inherited packet names: the CPU packet contains only `numpy_dag`, and the SDK packet contains only `upmem_sdk_simulator`. No `--allow-physical` is passed and the physical authorization environment is cleared.

**Exit condition:** `logs/adapter-verification.json` has `all_passed=true` for both targets, with CPU `(samples,sessions)=(2,0)` and SDK `(4,4)`. These counts are derived from the two exact six-qubit adapter paths and the two topologies. They do not consume the 768 hardware-attempt ceiling.

Existing completed raw qualification may be reused only when the original verifier accepts its exact source, policy, binary and selection identities. Failed or incomplete raw directories are preserved; the wrapper does not refill them.

## P6.5 — Step 3 — initialize and generate initial development traces

```bash
bash "$TOOLS/p6_operator.sh" initialize
bash "$TOOLS/p6_operator.sh" search initial
bash "$TOOLS/p6_operator.sh" freeze initial
bash "$TOOLS/p6_operator.sh" backup initial-frozen
```

The wrapper's `initialize` calls the existing `initialize --directory "$RUN/control"`. It refuses to replace an existing control directory. On a resumed study, inspect that directory and start at the first not-yet-completed step rather than reinitializing it.

The `search` wrapper obtains the 12 development cell IDs from `normalization.json`, not from a second hard-coded workload list. It invokes `initial-search` once per cell. It can skip an existing completed trace because `freeze-initial` revalidates every persisted trace against its frozen invocation. A started/failed directory without a valid completed trace stops the operation; it is not restarted.

The trace path for a cell is `control/initial/<record_hash(cell_id)>/`. `search_trace.jsonl` is flushed for every attempted proposal; `completed.json` exists only after complete successful bounded search.

**Exit conditions:** all 12 cell traces contain exactly 128 proposals; `initial_round.json` contains every development cell and at most four selected paths/cell; `expected_attempts <= 192`. Normalization uses development-greedy plans only. No physical calibration timing has been used.

Inspect only the frozen summary:

```bash
"$PY" - "$RUN/control/initial_round.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert len(m["cells"])==12 and 0<m["expected_attempts"]<=192
assert m["warmup_blocks"]==[0] and m["measurement_blocks"]==[1,2,3]
for key,cell in sorted(m["cells"].items()):
    assert 1<=len(cell["candidates"])<=4
    assert {"G","F","R"}<=set(cell["selection"]["roles"])
    print(key,len(cell["candidates"]),cell["selection"]["roles"])
print("attempts",m["expected_attempts"])
PY
```

Aliases mean fewer distinct mandatory paths; do not insist on three different role-selected paths. The existing diversity procedure supplies additional candidates only within the cap.

## P6.6 — Step 4 — prepare the initial CPU reference and hardware packet

```bash
bash "$TOOLS/p6_operator.sh" prepare initial
```

The wrapper uses the existing `write-packet --cpu-reference` input, executes one CPU reference per unique selected DAG on ETH, retrieves and verifies the raw qualification, then prepares the physical packet and exports a handoff with the existing `export-handoff`. It does not run hardware.

Exact physical packet/handoff calls are:

```bash
"$PY" scripts/upmem_cost_guided_path.py write-packet \
    --directory "$RUN/control" --stage initial \
    --output "$RUN/packets/initial" --execution-root "$RIMPL"
"$PY" scripts/upmem_cost_guided_path.py export-handoff \
    --directory "$RUN/control" --stage initial \
    --packet "$RUN/packets/initial" --qualification "$RUN/qualification" \
    --candidate-cpu "$RUN/stages/initial/cpu/raw"
```

The wrapper already issues them. Do not also run these expanded lines after the wrapper.

**Exit condition:** `control/initial_handoff.json` exists, the physical packet is present unchanged on ETH, and all source-specific qualification and selected-path CPU evidence has been reopened by `export-handoff`. No calibration sample has been executed.

Deployment-specific binary paths are created by `--execution-root "$RIMPL"`, not edited afterward in YAML. QASM relocation is handled by the existing private normalized comparison. Never manually rewrite the archived manifest, paths or hashes.

## P6.7 — Step 5 — one physical invocation, then verified acceptance

### P6.7.1 — Hardware readiness is not a source-code task

The controller uses `/home/tkazulak/evidence/upmem-experiment.lock` and rank 1. Its current preflight requires **every visible rank to be unowned**, not only rank 1. It also checks competing process names, source, binaries, SDK, CPU affinity/governor, memory and writable evidence storage. Keep that condition unchanged in this milestone. [S6]

Use one read-only occupancy check. When occupied, wait or retry the admission check for at most 15 minutes in the authorized workspace, then stop cleanly. Do not reserve another user's resources, kill their process, change a system-wide governor, or relax the predicate after seeing data.

The decisive once-only marker is:

```text
/home/tkazulak/evidence/cost-guided-invocations/
  upmem_cost_guided_path_study_v1-<study-source>-<stage>.json
```

Its identity does **not** depend on the output directory. Creating a new directory does not authorize another invocation.

### P6.7.2 — Execute and retrieve

```bash
bash "$TOOLS/p6_operator.sh" execute initial
```

The exact existing physical call on ETH is:

```bash
"$RPY" scripts/upmem_cost_guided_path.py execute-stage \
    --packet "$RRUN/packets/initial" \
    --output "$RRUN/stages/initial/physical" \
    --handoff "$RRUN/handoffs/initial_handoff.json" \
    --handoff-sha256 '<digest printed/computed from the frozen local handoff>'
```

The wrapper computes and passes the real digest; the angle-bracket text above is explanatory, not an executable placeholder to paste. No agent supplies a digest from memory.

The existing controller takes the lock, validates the external handoff and packet, performs preflight, writes its durable invocation marker, calls `quantum_bench.cli qualify` **once**, runs canonical verification, records terminal inspection, and archives the stage. It returns `accepted:false` even on successful execution; that is intentional. Retrieval and two-copy acceptance are still required. [S6]

The wrapper retrieves the archive even after a failed invocation when the archive exists. If archive finalization failed but a partial stage directory exists, it retrieves that directory to `incidents/` and stops. It never interprets a missing archive as permission to run again.

### P6.7.3 — Accept and fit

Only after successful physical execution and retrieval:

```bash
bash "$TOOLS/p6_operator.sh" accept initial
bash "$TOOLS/p6_operator.sh" fit initial
bash "$TOOLS/p6_operator.sh" backup initial-accepted-and-fit
```

The local acceptance reads two separately downloaded archive files:

```text
archive-A/initial/physical.tar.gz
archive-B/initial/physical.tar.gz
```

Both have matching outer digests, complete relative checksums and independently executed canonical verification. They are not symlinks or two names for the same inode. The original remote archive remains on ETH, providing the second host/failure domain in addition to local storage. Two local files alone would not establish two-host durability.

Accepted paths are recorded absolutely in the control state. **Do not move the active `RUN` directory or these archives during the campaign.** Use the separate portable export at closure rather than rewriting acceptance records.

**Exit condition:** `initial.accepted.json` derives rows from the accepted raw archives; `initial_fit/{fit,grid,profile}.json` exist; the grid contains 1,001 rows; the integer coefficients sum to ten. The first operational hardware→fit POC is now demonstrated. Its fitted training score is not a final result and cannot justify stopping with a generalization claim.

## P6.8 — Steps 6–7 — two bounded feedback stages

Execute the following separately for `feedback_1`, then for `feedback_2`:

```bash
STAGE=feedback_1              # Use feedback_2 only after feedback_1 is accepted and fit.
bash "$TOOLS/p6_operator.sh" search "$STAGE"
bash "$TOOLS/p6_operator.sh" freeze "$STAGE"
bash "$TOOLS/p6_operator.sh" backup "$STAGE-frozen"
"$PY" - "$RUN/control/${STAGE}_round.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));print("expected_attempts",m["expected_attempts"])
print("active_cells",len(m["cells"]),"skipped_cells",len(m.get("skipped_cells",{})))
assert m["expected_attempts"]<=144
PY
```

When `expected_attempts > 0`:

```bash
bash "$TOOLS/p6_operator.sh" prepare "$STAGE"
bash "$TOOLS/p6_operator.sh" execute "$STAGE"
bash "$TOOLS/p6_operator.sh" accept "$STAGE"
bash "$TOOLS/p6_operator.sh" fit "$STAGE"
bash "$TOOLS/p6_operator.sh" backup "$STAGE-accepted-and-fit"
```

When `expected_attempts == 0`, only for a fully frozen trace-proven empty feedback stage:

```bash
bash "$TOOLS/p6_operator.sh" accept "$STAGE"
bash "$TOOLS/p6_operator.sh" fit "$STAGE"
bash "$TOOLS/p6_operator.sh" backup "$STAGE-empty-accepted-and-fit"
```

Do **not** prepare a physical packet for an empty stage. Do not create an empty acceptance by hand. The existing accept command checks all skipped cells against complete search traces and earlier measured-path sets.

After feedback 2, no more adaptive rounds exist. Greedy controls are paired within their actual rounds and blocks; historical pilot observations and the previous family-aligned study's observations are not fit inputs. A weaker fitted objective is reported, not repaired by extending the experiment.

## P6.9 — Steps 8–10 — freeze and reach evaluation-ready

```bash
bash "$TOOLS/p6_operator.sh" pretest
bash "$TOOLS/p6_operator.sh" backup pretest-frozen
bash "$TOOLS/p6_operator.sh" search evaluation
bash "$TOOLS/p6_operator.sh" freeze evaluation
bash "$TOOLS/p6_operator.sh" backup evaluation-frozen
bash "$TOOLS/p6_operator.sh" prepare evaluation
```

`pretest` calls `freeze-pretest`, producing the actual implemented `pretest_profile.json`. Do not create a redundant `final_freeze.json` merely to match an older generic artifact list.

Evaluation search invokes `evaluation-search` for each test cell and each of exactly two objectives: `cotengra_tree_flops_v1` and `upmem_launch_cost_v1`. G and R are derived by the existing final selection logic, not given their own extra trace budgets.

Freeze every evaluation path before final selected-path CPU replay and before physical evaluation. If a selected path fails numerical qualification, stop and preserve the failure. Do not replace it with a faster or more accurate path after inspecting evaluation data.

### Evaluation-ready checks

```bash
cd "$I"
PYTHONPATH=src PYTHONOPTIMIZE=0 "$PY" - "$I" "$RUN/control" "$RUN/packets/evaluation" <<'PY'
import hashlib,json,sys
from pathlib import Path
impl,d,packet=map(Path,sys.argv[1:]);sys.path.insert(0,str(impl/"scripts"))
import upmem_cost_guided_path as p
s,w,b=p.load_study()
p._load_preparation(d,s,w)
profile=p._pretest_profile(d,s,w)
m=json.loads((d/"evaluation_round.json").read_text())
h=json.loads((d/"evaluation_handoff.json").read_text())
assert len(profile["evaluation_cells"])==len(m["cells"])==12
assert 0<m["expected_attempts"]<=288
assert m["profile_hash"]==p.record_hash(profile)==h["profile_hash"]
assert h["round_manifest_hash"]==p.record_hash(m)
assert h["packet_checksums_sha256"]==hashlib.sha256((packet/"SHA256SUMS").read_bytes()).hexdigest()
assert h["qualification_receipt"]["all_passed"] is True
for name in ("cpu","sdk","candidate_cpu"):
    assert h["qualification_receipt"][name]["all_passed"] is True
for cell in m["cells"].values():
    assert set(cell["selection"]["roles"])=={"G","F","R","U"}
assert not (d/"evaluation.accepted.json").exists()
print("EVALUATION-READY: final roles, model, packet, selected-path qualification and budget are frozen")
print("planned physical attempts",m["expected_attempts"])
PY
```

These assertions are operator checks on already generated files, not replacement evidence validation. The actual `execute-stage` still revalidates the handoff and deployment before admission.

**This is the requested pre-final-evaluation milestone.** No new code or experiment is required between this gate and the final physical stage, unless admission or a demonstrated correctness defect blocks it.

## P6.10 — Step 11 — frozen final evaluation

When the authorized campaign proceeds beyond evaluation-ready:

```bash
bash "$TOOLS/p6_operator.sh" execute evaluation
bash "$TOOLS/p6_operator.sh" accept evaluation
bash "$TOOLS/p6_operator.sh" backup evaluation-accepted
```

There are no `fit evaluation`, `feedback_3`, extra seed searches, changed tolerances, or candidate substitutions. The controller enforces the relevant guards; the operator must not bypass them.

The cumulative 86,400-second cap is the sum of the recorded physical runner child-process durations, including work inside that runner's boundary. It is not a guarantee that all offline work, transfers and writing fit within one day. The existing runner's 120-second attempt limit and once-only marker remain authoritative.

## P6.11 — Step 12 — produce the readout, interpret, and archive

### P6.11.1 — Run the read-only report

```bash
cd "$I"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" "$TOOLS/p6_readout.py" \
    --implementation "$I" --directory "$RUN/control" --output "$RUN/readout"
(cd "$RUN/readout" && sha256sum -c SHA256SUMS)
```

The script loads the existing research binding, final profile, and accepted evaluation through the repository's current functions. It reopens the accepted archives before computing anything. It has no SDK call, path search, fitter, mutation of accepted evidence, or physical execution path.

Outputs:

| File | Meaning |
|---|---|
| `accepted_evaluation_rows.json` | Reverified normalized rows, including retained warmups |
| `observations.csv` | Five measured observations per method/cell; aliases identify the same physical sample |
| `methods.csv` | Method/path mapping, medians and raw MADs; available kernel/transfer/host metrics retained |
| `contrasts.csv` | Within-cell ratios, paired intervals, reductions and same-path flags |
| `aggregates.csv` | Family-balanced combined and per-topology geometric ratios, worst cells and regression counts |
| `search.csv` | Proposal counts, duplicates/infeasibility, search duration components and F/U parameter-trace differences |
| `selected_plan_facts.csv` | H/P/N, aggregate M/W and launch counts for selected plans |
| `model_diagnostics.json` | Constant features, descriptive correlations, and grid selection-equivalence/tie counts |
| `provenance.json` | Source/acceptance/profile/readout bindings and exact statistical convention |
| `report.md` | Compact numerical result and scope limitations |
| `SHA256SUMS` | Readout file checksums |

Optional timing columns remain null when unavailable. Do not fill them with zero, infer CPU-only time by arbitrary subtraction, or add overlapping timing envelopes as if they were disjoint.

### P6.11.2 — Interpret without another optimization loop

The primary claim is conditional on the fixed family instances, executor, profiles, resource cells, seeds and timing boundary. Use precise formulations:

- **U better than R:** feedback-guided generation improved execution beyond the particular conventional-pool reranking control under this budget.
- **R better than G/F, U not better than R:** hardware-aware selection helped, but this bounded generator adaptation did not establish additional benefit.
- **All similar or identical paths:** little measured improvement in this tested search/domain, not evidence that the implementation must be expanded.
- **U slower:** a valid failure of this fitted model/search protocol to select faster execution on the tested instance; report it.

A neutral cell with little measured candidate diversity differs from a poor selector in a pool with substantial measured headroom. Do not call generated-but-untimed candidates a physical oracle. Do not add extra timing to make that distinction after the final freeze.

The report's pooled feature correlations are descriptive. Correlated features or many grid tuples selecting identical paths are evidence against treating the chosen tuple as uniquely identified physical penalties. They do not justify switching model form after evaluation.

### P6.11.3 — Portable delivery without changing live acceptance paths

Leave the live operator directory and both raw-copy paths in place. Build a separate portable export containing one copy of each raw archive plus controls, qualification evidence, packet inputs, tools, readout, and logs. The active second archive and the ETH originals remain retained.

```bash
export EXPORT="$HOME/evidence/upmem-p6-export-$SOURCE"
test ! -e "$EXPORT"
mkdir -p "$EXPORT"
for part in control qualification packets stages readout logs tools; do
    cp -a "$RUN/$part" "$EXPORT/$part"
done
cp -a "$RUN/archive-A" "$EXPORT/raw-archives"
cp "$RUN/deployment.env" "$EXPORT/deployment.env"
# Publish the already inspected exact repository source as a portable snapshot.
git -C "$LOCAL_WORKTREE" archive --format=tar.gz \
    --output="$EXPORT/source-$SOURCE.tar.gz" "$SOURCE" \
    thesis/implementation thesis/upmem-system-and-path-optimization-plan-v2.md
```

`git archive` does not include submodule contents or ignored native binaries. Include the three exact frozen executables separately from ETH. The raw manifests and research binding retain environment facts; the older executor evidence archives remain separately identified by their recorded digests. Do not claim this export contains the SDK installation or automatically embeds the earlier executor archives.

```bash
mkdir "$EXPORT/frozen-binaries"
for name in dpu_wave_v5_t8 host_upmem_execution_plan_v4_t8 dpu_simplepim_management_init_t8; do
    scp "$ETH:$RIMPL/native/upmem/runtime/bin/$name" "$EXPORT/frozen-binaries/$name"
done
"$PY" - "$I" "$EXPORT" <<'PY'
import hashlib,json,sys
from pathlib import Path
impl,root=map(Path,sys.argv[1:]);s=json.loads((impl/"configs/upmem_cost_guided_path_study_v1.json").read_text())
for name,expected in s["executor"]["binaries"].items():
    if hashlib.sha256((root/"frozen-binaries"/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit("Frozen export binary mismatch")
for p in root.rglob("*"):
    if p.is_symlink():raise SystemExit(f"Inspect and resolve export symlink explicitly, do not silently dereference: {p}")
with (root/"SHA256SUMS").open("x") as out:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p!=root/"SHA256SUMS":
            h=hashlib.sha256()
            with p.open("rb") as src:
                for block in iter(lambda:src.read(1024*1024),b""):h.update(block)
            out.write(f"{h.hexdigest()}  {p.relative_to(root).as_posix()}\n")
PY
(cd "$EXPORT" && sha256sum -c SHA256SUMS)
tar -C "$(dirname "$EXPORT")" -czf "$EXPORT.tar.gz" "$(basename "$EXPORT")"
(cd "$(dirname "$EXPORT")" && sha256sum "$(basename "$EXPORT").tar.gz" > "$(basename "$EXPORT").tar.gz.sha256")
ssh "$ETH" "test ! -e '$RRUN/final-export.tar.gz'"
scp "$EXPORT.tar.gz" "$ETH:$RRUN/final-export.tar.gz"
LOCAL_EXPORT_SHA=$(sha256sum "$EXPORT.tar.gz" | cut -d' ' -f1)
REMOTE_EXPORT_SHA=$(ssh "$ETH" "sha256sum '$RRUN/final-export.tar.gz'" | cut -d' ' -f1)
test "$LOCAL_EXPORT_SHA" = "$REMOTE_EXPORT_SHA"
ssh "$ETH" "set -C; printf '%s  final-export.tar.gz\n' '$LOCAL_EXPORT_SHA' > '$RRUN/final-export.tar.gz.sha256'; cd '$RRUN'; sha256sum -c final-export.tar.gz.sha256"
printf '%s\n' "$LOCAL_EXPORT_SHA" > "$RUN/logs/final-export-sha256.txt"
```

The final bundle preserves original absolute acceptance metadata for provenance. It is not a promise that the live coordinator can be resumed at a new arbitrary pathname without a restoration step. A general portable-resume layer is outside this milestone.

Public release is a separate authorized write action. Its description should identify study source, executor source, execution binary hashes, observed conditions, raw/readout archive digests, the six-family instance-transfer scope, and whether G/F/R/U coincide. Do not relabel this as universal PIM acceleration.

## P6.12 — Failure decisions — no agent improvisation

| Observed condition | Required action | Prohibited shortcut |
|---|---|---|
| Local/ETH SHA, research lock, SDK or binary mismatch | Restore or select the exact already qualified deployment; rerun read-only checks | Upgrade dependencies, rebuild arbitrary binaries, merge old branches |
| Stale README/status prose | Record it outside the campaign; correct in a reporting descendant after completion | Treat it as evidence the optimizer must be rewritten |
| All ranks occupied or known competing workload before invocation | Stop admission; at most 15 minutes of read-only availability checking; later retry only when marker is absent | Change rank ownership, kill another user's job, relax the predicate |
| Invocation marker exists | Inspect its referenced output and terminal state; preserve evidence | Delete marker, change output directory/source to disguise a retry |
| Search directory has `started.json` but no complete trace | Stop, retain all flushed proposals and logs; diagnose | Delete directory or restart to get a nicer/completed set |
| Known infeasible proposal | Preserve status/reason and count it within 128 | Replace it with an extra proposal |
| Empty feedback with complete proof | Use existing zero-attempt acceptance, then fit through the stage | Invent rows or launch a packet with no candidates |
| Physical attempt fails, times out, violates numerical policy or ownership | Stop, retrieve complete partial evidence, no fit from that stage | Splice a replacement sample/cell or automatically start a new campaign |
| Physical run succeeds but archive transfer fails | Repeat transfer/checksum verification only; leave hardware alone | Run hardware again because local evidence is missing |
| Archive finalization fails after invocation | Retrieve raw directory and markers; bounded archive-only recovery after diagnosis | Claim acceptance without checksums or rerun the data collection |
| `accepted:false` after `execute-stage` | Run the existing acceptance after retrieval | Treat this intentional field as a hardware failure |
| A completed stage has an acceptance/profile already | Reverify and continue at the next unfinished operation | Re-execute or overwrite it |
| Final evaluation is neutral or negative | Complete readout and archive, then stop | Additional coefficients, trials, families, seeds or thresholds |

### Critical-source-repair rule

A production patch is justified only by a reproducible failure that prevents exact qualification, valid invocation/acceptance, or correct reporting of this declared study. It needs one failing regression and the smallest repair. Environmental path/permission problems are fixed in deployment, not by changing scientific rules.

Before the first physical calibration, a necessary study-adapter repair creates a new exact study source, receives focused and normal full-suite/Ruff/CI qualification, and is rebound before any source-specific traces/receipts are used. Executor behavior and its binary identities stay frozen.

After physical calibration begins, do not silently advance source, change features, or bypass the consumed-stage marker. Preserve the incident and state which observations remain scientifically valid. A replacement campaign is a new explicitly bounded decision, not an automatic option in this runbook.

Do not plan hypothetical fixes in advance. At the inspected commit, no missing cost-function/search/fitter implementation has been demonstrated. The first action is deployment/qualification, not editing those modules.

## P6.13 — Evidence checklist and final research statement

The delivered milestone must identify the exact study and executor sources separately; all complete proposal traces and candidate/fact hashes; initial normalization; all three fit tables/profiles; every stage manifest and acceptance; warmups and measurements; actual invocation/budget/terminal records; numeric/source/resource outcomes; final role deduplication; final readout convention/code; and two-host retained raw/final archives.

The final thesis statement can then be precise:

> A frozen one-rank UPMEM executor was coupled to a deterministic launch-aware ranking objective. Physical development measurements selected coefficients from a finite grid. Those coefficients guided subsequent cotengra generator-parameter proposals through a bounded adaptive loop. A frozen G/F/R/U evaluation measured whether guided generation improved physical execution beyond conventional generation and hardware-aware reranking on held-out instances from six represented circuit families.

The statement remains valid whether the empirical outcome is positive, neutral or negative. It does not require a new kernel, residency engine, slicing strategy, multi-rank scheduler, CPU/GPU placement, or a second numerical campaign.

## P6.14 — Source and verification notes

Sources below were inspected at the pinned commit through the GitHub connector on 10 September 2026. The branch HEAD was freshly checked and remained `2beea27411c16e90ed76988613ddb00bcc09f942`. This planning task did not modify the repository, execute its tests, connect to ETH, or launch experiments.

- **S1 — Current checkpoint:** [docs/upmem_cost_guided_path_v1.md](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/docs/upmem_cost_guided_path_v1.md). Existing software/qualification status and source bindings.
- **S2 — Study/manifest:** [configs/upmem_cost_guided_path_study_v1.json](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/configs/upmem_cost_guided_path_study_v1.json), and its hashed family workload. Exact budgets, methods, policies and search parameters.
- **S3 — Existing coordinator:** [scripts/upmem_cost_guided_path.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/scripts/upmem_cost_guided_path.py). Argument names, state files, fitting/search/freezing, archive acceptance, CPU/SDK handoff.
- **S4 — Exact path packet and evidence adapters:** [scripts/qualify_upmem_path_candidates.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/scripts/qualify_upmem_path_candidates.py), [scripts/upmem_cost_guided_evidence.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/scripts/upmem_cost_guided_evidence.py). Frozen configuration, two CPU/four SDK adapter observations, exact raw validation and returned timing fields.
- **S5 — Existing mathematics:** [src/quantum_bench/upmem/path_heuristic.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/src/quantum_bench/upmem/path_heuristic.py), [src/quantum_bench/upmem/execution_features.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/src/quantum_bench/upmem/execution_features.py).
- **S6 — Existing physical invocation:** [scripts/upmem_cost_guided_execution.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/scripts/upmem_cost_guided_execution.py). Exact all-ranks-idle check, private flock, invocation ledger, owned-process cleanup, archive and separate acceptance.
- **S7 — Controlling research plan:** [thesis/upmem-system-and-path-optimization-plan-v2.md, §11](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/upmem-system-and-path-optimization-plan-v2.md). Mathematical and bounded experiment contract.
- **S8 — CI:** [successful exact-source run 34370052538](https://github.com/kazulak/Masters/actions/runs/34370052538). Software CI, not SDK/physical qualification.
- **S9 — CLI and environment:** [src/quantum_bench/cli.py](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/src/quantum_bench/cli.py), [Makefile](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/Makefile), [requirements-upmem-path-search.txt](https://github.com/kazulak/Masters/blob/2beea27411c16e90ed76988613ddb00bcc09f942/thesis/implementation/requirements-upmem-path-search.txt).

**Delivery validation:** `bash -n` was run on the supplied operator wrapper. The supplied Python readout compiled, and five synthetic mathematical tests passed, covering exact contrasts, method aliases, warmup exclusion, family weighting, missing/duplicate rows, timing-boundary mismatch, and deterministic output. The full repository and the actual ETH command chain were not executed here. Live success must not be claimed until the explicit source-specific qualification and acceptance gates pass.

## Appendix A — Complete operator wrapper source

This wrapper uses existing CLI commands; no new sampling, fitting or accepted-evidence state machine is introduced.

```bash
#!/usr/bin/env bash
# Operator-side command wrappers. No sampling loop, model, fitter, or new state machine.
# Keep this file OUTSIDE the tracked checkout. Source deployment.env before invoking.
set -Eeuo pipefail
: "${I:?absolute local thesis/implementation path}"
: "${PY:?absolute pinned local research Python}"
: "${RUN:?durable absolute local operator directory}"
: "${ETH:?authorized SSH destination, e.g. tkazulak@safari-baguette1}"
: "${RIMPL:?absolute ETH thesis/implementation path}"
: "${RPY:?absolute pinned research Python on ETH}"
: "${RRUN:?durable absolute ETH operator directory}"
SOURCE=${SOURCE:-2beea27411c16e90ed76988613ddb00bcc09f942}
EXECUTOR=459935f586fdd16c82013838e6d27a12604c3093
export I PY RUN ETH RIMPL RPY RRUN SOURCE
for value in "$I" "$PY" "$RUN" "$RIMPL" "$RPY" "$RRUN"; do
  [[ "$value" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo 'Use absolute Linux paths without spaces/shell metacharacters.' >&2; exit 2; }
done
[[ "$ETH" =~ ^[A-Za-z0-9][A-Za-z0-9_.@:-]*$ ]] || exit 2
[[ "$SOURCE" =~ ^[0-9a-f]{40}$ ]] || exit 2
cd "$I"
[[ "$(git rev-parse HEAD)" == "$SOURCE" && -z "$(git status --porcelain)" ]] || { echo 'Wrong or dirty local source; stop.' >&2; exit 2; }
[[ "$(git rev-parse 'thesis-upmem-kernel-schedule-system-v1^{commit}')" == "$EXECUTOR" ]] || exit 2
git merge-base --is-ancestor "$EXECUTOR" "$SOURCE" || exit 2
mkdir -p "$RUN/logs" "$RUN/packets" "$RUN/stages" "$RUN/archive-A" "$RUN/archive-B"
D="$RUN/control"
export D PYTHONPATH="$I/src" PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset UPMEM_ALLOW_PHYSICAL_HARDWARE DPU_BACKEND UPMEM_REQUIRE_SDK_SIMULATOR || true

p6() { "$PY" "$I/scripts/upmem_cost_guided_path.py" "$@"; }
valid_stage() { [[ "$1" =~ ^(initial|feedback_1|feedback_2|evaluation)$ ]]; }

remote() {
  # Every argument is a validated space-free token; script content uses quoted expansions.
  ssh "$ETH" bash -l -s -- "$RIMPL" "$RPY" "$RRUN" "$SOURCE" "$@" <<'REMOTE'
set -Eeuo pipefail
ri=$1; py=$2; rr=$3; source=$4; action=$5; stage=${6:-}
cd "$ri"
[[ "$(pwd -P)" == "$ri" ]] || { echo "RIMPL must be its resolved physical directory" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$source" && -z "$(git status --porcelain)" ]] || exit 2
export PYTHONPATH="$ri/src" PYTHONDONTWRITEBYTECODE=1 PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset UPMEM_ALLOW_PHYSICAL_HARDWARE DPU_BACKEND UPMEM_REQUIRE_SDK_SIMULATOR || true
"$py" -c 'import sys; sys.path.insert(0,"scripts"); import upmem_cost_guided_path as p; p.research_binding(p.load_study()[0])'
mkdir -p "$rr" "$rr/packets" "$rr/stages" "$rr/handoffs"
run_nonphysical() {
  target=$1
  if [[ -e "$target/raw" ]]; then
    "$py" - "$target/raw/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if m.get("status") != "completed": raise SystemExit("Incomplete/failed qualification exists: retain it; do not rerun here.")
PY
  else
    taskset -c 0 "$py" -m quantum_bench.cli run \
      --config "$target/preregistration/physical.yml" --output "$target/raw" \
      > "$target/run.log" 2>&1
  fi
  "$py" -m quantum_bench.cli verify --input "$target/raw"
}
case "$action" in
  check)
    "$py" - <<'PY'
import hashlib,json,pathlib,subprocess,sys
sys.path.insert(0,"scripts")
import upmem_cost_guided_path as p
s,_,_=p.load_study()
if sys.version_info[:3] != (3,10,12): raise SystemExit("Use the recorded Python 3.10.12 environment")
sdk=subprocess.check_output(["dpu-pkg-config","--modversion","dpu"],text=True).strip()
if sdk != "2023.1.0": raise SystemExit("Wrong/missing SDK environment; do not replace the SDK")
for name,expected in s["executor"]["binaries"].items():
 f=pathlib.Path("native/upmem/runtime/bin")/name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
  raise SystemExit(f"Missing or wrong frozen binary: {f}; restore exact retained bytes, do not rebuild blindly")
print(json.dumps({"binding":p.research_binding(s),"sdk":sdk,"python":sys.version.split()[0]},sort_keys=True))
PY
    ;;
  mkdir-adapter) mkdir -p "$rr/qualification" ;;
  run-adapter)
    run_nonphysical "$rr/qualification/cpu"
    run_nonphysical "$rr/qualification/sdk"
    ;;
  mkdir-cpu) mkdir -p "$rr/stages/$stage/cpu" ;;
  run-cpu) run_nonphysical "$rr/stages/$stage/cpu" ;;
  mkdir-packet) mkdir -p "$rr/packets/$stage" "$rr/stages/$stage" ;;
  execute)
    hash=$7
    "$py" scripts/upmem_cost_guided_path.py execute-stage \
      --packet "$rr/packets/$stage" --output "$rr/stages/$stage/physical" \
      --handoff "$rr/handoffs/${stage}_handoff.json" --handoff-sha256 "$hash"
    ;;
  *) echo 'Unknown remote operation' >&2; exit 2 ;;
esac
REMOTE
}

verify_adapter() {
  "$PY" - "$I" "$RUN/qualification" <<'PY'
import json,sys
from pathlib import Path
impl,q=map(Path,sys.argv[1:]);sys.path[:0]=[str(impl/"scripts"),str(impl/"src")]
import upmem_cost_guided_path as p
from upmem_cost_guided_evidence import verify_qualification
s,_,_=p.load_study();b=json.loads((q/"binding.json").read_text());sel=json.loads((q/"selection.json").read_text());w=json.loads((q/"workload.json").read_text())
if b!=p.research_binding(s): raise SystemExit("Adapter source/environment differs")
reports={t:verify_qualification(q/t/"raw",sel,w,b,s,target=t) for t in ("cpu","sdk")}
if (reports["cpu"]["sample_count"],reports["cpu"]["session_count"],reports["sdk"]["sample_count"],reports["sdk"]["session_count"])!=(2,0,4,4):
 raise SystemExit("Wrong adapter counts")
print(json.dumps(reports,indent=2))
PY
}

retrieve() {
  local stage=$1 base="$RRUN/stages/$1/physical.tar.gz"
  if ! ssh "$ETH" "test -f '$base' && test -f '$base.sha256'"; then
    if ssh "$ETH" "test -d '$RRUN/stages/$stage/physical'"; then
      mkdir -p "$RUN/incidents/$stage"
      rsync -a --ignore-existing "$ETH:$RRUN/stages/$stage/physical/" "$RUN/incidents/$stage/physical/"
      echo 'Partial stage tree retrieved. No accepted archive: do not rerun hardware.' >&2
    else
      echo 'No stage output. Inspect the remote invocation marker before deciding that admission consumed no attempt.' >&2
    fi
    return 2
  fi
  for copy in archive-A archive-B; do
    local dest="$RUN/$copy/$stage"
    mkdir -p "$dest" || return 2
    # Transfer may be repeated; only temporary transfer files are replaced.
    # Published archives and accepted records are never replaced.
    scp "$ETH:$base.sha256" "$dest/expected.sha256.transfer" || return 2
    if [[ ! -f "$dest/physical.tar.gz" ]]; then
      scp "$ETH:$base" "$dest/physical.tar.gz.transfer" || return 2
    fi
    "$PY" - "$dest" <<'PY' || return 2
import hashlib,os,sys
from pathlib import Path
p=Path(sys.argv[1]);expected=(p/"expected.sha256.transfer").read_text().split()
if len(expected)!=2 or expected[1]!="physical.tar.gz": raise SystemExit("Malformed outer checksum")
final=p/"physical.tar.gz";tmp=p/"physical.tar.gz.transfer";source=final if final.exists() else tmp
h=hashlib.sha256()
with source.open("rb") as f:
 for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
if h.hexdigest()!=expected[0]:raise SystemExit("Download checksum mismatch; retain partials, repeat transfer only")
if not final.exists(): os.link(tmp,final);tmp.unlink()
side=p/"physical.tar.gz.sha256"
text=f"{expected[0]}  physical.tar.gz\n"
if side.exists():
 if side.read_text()!=text:raise SystemExit("Published checksum changed")
else:
 with side.open("x") as f:f.write(text);f.flush();os.fsync(f.fileno())
with final.open("rb") as f:os.fsync(f.fileno())
fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
print(str(final),expected[0])
PY
  done
  echo 'Remote original remains retained; both local archive files are ready for canonical acceptance.'
}

command=${1:?command required}; stage=${2:-}
case "$command" in
  check)
    p6 inspect
    "$PY" - "$I" <<'PY' > "$RUN/logs/local-binding.json"
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/"scripts"))
import upmem_cost_guided_path as p
if sys.version_info[:3]!=(3,10,12):raise SystemExit("Use recorded Python 3.10.12")
print(json.dumps(p.research_binding(p.load_study()[0]),sort_keys=True))
PY
    remote check > "$RUN/logs/remote-binding.json"
    "$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);a=json.loads((r/"logs/local-binding.json").read_text());b=json.loads((r/"logs/remote-binding.json").read_text())
if a!=b["binding"]:raise SystemExit("Local/ETH source or research dependency mismatch")
print("Exact local/ETH source, research environment, SDK and frozen binaries match.")
PY
    ;;
  adapter)
    if [[ ! -d "$RUN/qualification" ]]; then
      p6 write-adapter-packets --output "$RUN/qualification" --execution-root "$RIMPL"
    fi
    remote mkdir-adapter
    rsync -a --ignore-existing "$RUN/qualification/" "$ETH:$RRUN/qualification/"
    status=0
    remote run-adapter 2>&1 | tee -a "$RUN/logs/adapter.log" || status=$?
    rsync -a --ignore-existing "$ETH:$RRUN/qualification/" "$RUN/qualification/"
    [[ "$status" == 0 ]] || exit 2
    verify_adapter | tee "$RUN/logs/adapter-verification.json"
    ;;
  initialize) [[ ! -e "$D" ]] || { echo 'Control directory exists: reuse/inspect it; never reinitialize.' >&2; exit 2; }; p6 initialize --directory "$D" ;;
  search)
    valid_stage "$stage" || exit 2
    "$PY" - "$I" "$D" "$stage" <<'PY'
import json,subprocess,sys
from pathlib import Path
impl,d=map(Path,sys.argv[1:3]);stage=sys.argv[3];sys.path.insert(0,str(impl/"scripts"))
import upmem_cost_guided_path as p
if (d/f"{stage}_round.json").exists():raise SystemExit("Round is already frozen; do not search again")
if stage=="evaluation":
 cells=json.loads((d/"pretest_profile.json").read_text())["evaluation_cells"]
 objectives=("cotengra_tree_flops_v1","upmem_launch_cost_v1")
else:
 if (d/"pretest_profile.json").exists():raise SystemExit("Adaptation forbidden after freeze")
 cells=sorted(json.loads((d/"normalization.json").read_text())["greedy_cells"])
 objectives=(None,)
for cell in cells:
 for objective in objectives:
  folder=d/stage/p.record_hash(cell)
  if objective:folder/=objective
  if folder.exists():
   if (folder/"completed.json").is_file() and not (folder/"failed.json").exists():
    print("Reuse completed trace; freeze will revalidate:",folder,flush=True);continue
   raise SystemExit(f"Interrupted/failed trace: {folder}. No refill or rerun.")
  cmd="evaluation-search" if objective else "initial-search" if stage=="initial" else "feedback-search"
  args=[sys.executable,str(impl/"scripts/upmem_cost_guided_path.py"),cmd,"--directory",str(d),"--cell",cell]
  if objective:args += ["--objective",objective]
  elif stage!="initial":args += ["--stage",stage]
  subprocess.run(args,cwd=impl,check=True)
PY
    ;;
  freeze)
    valid_stage "$stage" || exit 2
    if [[ "$stage" == initial ]];then p6 freeze-initial --directory "$D"
    elif [[ "$stage" == evaluation ]];then p6 freeze-evaluation --directory "$D"
    else p6 freeze-feedback --directory "$D" --stage "$stage";fi
    ;;
  prepare)
    valid_stage "$stage" || exit 2
    "$PY" - "$D/${stage}_round.json" <<'PY'
import json,sys
if json.load(open(sys.argv[1]))["expected_attempts"]==0:raise SystemExit("Empty feedback: use accept then fit; no packets or hardware.")
PY
    cpu="$RUN/stages/$stage/cpu"
    if [[ ! -e "$cpu/preregistration" ]]; then
      p6 write-packet --directory "$D" --stage "$stage" --output "$cpu/preregistration" --execution-root "$RIMPL" --cpu-reference
    fi
    remote mkdir-cpu "$stage"
    rsync -a --ignore-existing "$cpu/" "$ETH:$RRUN/stages/$stage/cpu/"
    status=0
    remote run-cpu "$stage" 2>&1 | tee -a "$RUN/logs/$stage-cpu.log" || status=$?
    rsync -a --ignore-existing "$ETH:$RRUN/stages/$stage/cpu/" "$cpu/"
    [[ "$status" == 0 ]] || exit 2
    if [[ ! -e "$RUN/packets/$stage" ]]; then
      p6 write-packet --directory "$D" --stage "$stage" --output "$RUN/packets/$stage" --execution-root "$RIMPL"
    fi
    if [[ ! -e "$D/${stage}_handoff.json" ]]; then
      p6 export-handoff --directory "$D" --stage "$stage" --packet "$RUN/packets/$stage" \
        --qualification "$RUN/qualification" --candidate-cpu "$cpu/raw"
    fi
    remote mkdir-packet "$stage"
    rsync -a --ignore-existing "$RUN/packets/$stage/" "$ETH:$RRUN/packets/$stage/"
    rsync -a --ignore-existing "$D/${stage}_handoff.json" "$ETH:$RRUN/handoffs/"
    echo 'Prepared only. No physical execution performed. The executor revalidates packet and handoff before admission.'
    ;;
  execute)
    valid_stage "$stage" || exit 2
    hash=$(sha256sum "$D/${stage}_handoff.json" | cut -d' ' -f1)
    status=0
    remote execute "$stage" "$hash" 2>&1 | tee -a "$RUN/logs/$stage-execute.log" || status=$?
    # Retrieval is not a retry. Retrieve even when physical execution failed.
    retrieved=0; retrieve "$stage" || retrieved=$?
    [[ "$status" == 0 && "$retrieved" == 0 ]] || exit 2
    ;;
  retrieve) valid_stage "$stage" || exit 2; retrieve "$stage" ;;
  accept)
    valid_stage "$stage" || exit 2
    count=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["expected_attempts"])' "$D/${stage}_round.json")
    if [[ "$count" == 0 ]]; then
      p6 accept --directory "$D" --stage "$stage"
    else
      p6 accept --directory "$D" --stage "$stage" \
        --archive "$RUN/archive-A/$stage/physical.tar.gz" --archive "$RUN/archive-B/$stage/physical.tar.gz"
    fi
    ;;
  fit) valid_stage "$stage" && [[ "$stage" != evaluation ]] || exit 2; p6 fit --directory "$D" --stage "$stage" ;;
  pretest) p6 freeze-pretest --directory "$D" ;;
  backup)
    [[ "$stage" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
    ssh "$ETH" "test ! -e '$RRUN/control-snapshots/$stage' && mkdir -p '$RRUN/control-snapshots/$stage'"
    rsync -a "$D/" "$ETH:$RRUN/control-snapshots/$stage/"
    "$PY" - "$D" <<'PY' > "$RUN/logs/$stage-control-SHA256SUMS"
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1])
for p in sorted(root.rglob("*")):
 if p.is_symlink():raise SystemExit("No symlinks in control snapshot")
 if p.is_file():print(hashlib.sha256(p.read_bytes()).hexdigest()," "+p.relative_to(root).as_posix())
PY
    scp "$RUN/logs/$stage-control-SHA256SUMS" "$ETH:$RRUN/control-snapshots/$stage/SHA256SUMS"
    ssh "$ETH" "cd '$RRUN/control-snapshots/$stage' && sha256sum -c SHA256SUMS"
    ;;
  *) echo 'Commands: check adapter initialize search STAGE freeze STAGE prepare STAGE execute STAGE retrieve STAGE accept STAGE fit STAGE pretest backup LABEL' >&2; exit 2 ;;
esac
```

## Appendix B — Complete read-only reporting adapter

```python
#!/usr/bin/env python3
"""Read-only P6 readout. Uses the repository's verifier; never executes hardware.

Place outside the tracked checkout. This is an operator-side report adapter,
not a new cost model, fitter, evidence-acceptance implementation, or runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = ("G", "F", "R", "U")
# Numerator/denominator. A ratio above one favors the denominator method.
CONTRASTS = (("R", "U"), ("F", "U"), ("G", "U"), ("F", "R"), ("G", "R"), ("G", "F"))
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 10_000
CORE_METRICS = ("session_inclusive_s", "total_wall_s")
OPTIONAL_METRICS = (
    "session_open_s", "session_close_s", "kernel_s", "h2d_s", "d2h_s",
    "request_build_sum_s", "request_wave_wall_sum_s",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest: dict, rows: list[dict]) -> dict:
    """Pure reporting math on rows already accepted by the canonical verifier.

    Point estimates are ratios of within-cell arm medians. Paired bootstrap
    draws resample the same complete block indices for every arm and cell.
    Families/cells are fixed; they are not bootstrap sampling units here.
    """
    require(manifest.get("stage") == "evaluation", "evaluation manifest required")
    blocks = tuple(manifest["measurement_blocks"])
    require(blocks == (1, 2, 3, 4, 5), "require the five frozen evaluation blocks")
    cells = manifest["cells"]
    require(bool(cells), "empty evaluation")
    lookup = {}
    for row in rows:
        if row["attempt_type"] == "warmup":
            continue
        require(row["attempt_type"] == "measurement", "unknown attempt kind")
        require(row["round_id"] == "evaluation" and row["split"] == "test", "wrong split/stage")
        require(row["status"] == "success" and row["validation"] == "pass"
                and row["fallback"] is False, "ineligible observation")
        key = (row["cell_id"], row["path_id"], row["block"])
        require(key not in lookup, "duplicate physical observation")
        inclusive = row["session_open_s"] + row["total_wall_s"] + row["session_close_s"]
        require(math.isclose(inclusive, row["session_inclusive_s"], rel_tol=1e-12, abs_tol=1e-12),
                "inconsistent session-inclusive boundary")
        for field in CORE_METRICS:
            require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] > 0,
                    f"invalid {field}")
        lookup[key] = row
    expected = {(c, p, b) for c, cell in cells.items() for p in cell["candidates"] for b in blocks}
    require(set(lookup) == expected, "measurement set differs from frozen selected paths")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(blocks), size=(BOOTSTRAP_DRAWS, len(blocks))
    )
    observations, method_rows, contrast_rows = [], [], []
    bootstrap_by_key = {}
    matrices = {}
    for cell_id, cell in sorted(cells.items()):
        require(cell["split"] == "test", "non-test cell")
        roles = cell["selection"]["roles"]
        require(set(roles) == set(METHODS), "G/F/R/U roles missing or unexpected")
        metadata = {"cell_id": cell_id, "family": cell["family"], "topology_id": cell["topology_id"]}
        for method in METHODS:
            selected = roles[method]
            chosen = [lookup[(cell_id, selected, b)] for b in blocks]
            base = {**metadata, "method": method, "path_id": selected, "n": len(blocks)}
            for block, row in zip(blocks, chosen):
                observations.append({**base, "block": block,
                                     "sample_id": row.get("sample_id"),
                                     "session_instance_id": row.get("session_instance_id"),
                                     **{f: row.get(f) for f in (*CORE_METRICS, *OPTIONAL_METRICS)}})
            for field in (*CORE_METRICS, *OPTIONAL_METRICS):
                present = all(row.get(field) is not None for row in chosen)
                if not present:
                    base[f"{field}_median"] = None
                    base[f"{field}_mad"] = None
                    continue
                values = np.asarray([row[field] for row in chosen], dtype=np.float64)
                require(np.all(np.isfinite(values)) and np.all(values >= 0), "invalid optional timing")
                med = float(np.median(values))
                base[f"{field}_median"] = med
                base[f"{field}_mad"] = float(np.median(np.abs(values - med)))
                if field in CORE_METRICS:
                    matrices[(cell_id, method, field)] = values
            method_rows.append(base)
        for metric in CORE_METRICS:
            for numerator, denominator in CONTRASTS:
                a, b = matrices[(cell_id, numerator, metric)], matrices[(cell_id, denominator, metric)]
                ratio = float(np.median(a) / np.median(b))
                boot = np.median(a[draws], axis=1) / np.median(b[draws], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                label = f"{numerator}/{denominator}"
                bootstrap_by_key[(cell_id, label, metric)] = boot
                contrast_rows.append({**metadata, "contrast": label, "metric": metric,
                                      "ratio_of_medians": ratio,
                                      "denominator_time_reduction_pct": 100 * (1 - 1 / ratio),
                                      "paired_bootstrap_low": float(lo), "paired_bootstrap_high": float(hi),
                                      "same_selected_path": roles[numerator] == roles[denominator]})
    aggregate_rows = []
    groups = [("all", sorted(cells))] + [
        (topology, sorted(c for c in cells if cells[c]["topology_id"] == topology))
        for topology in sorted({cell["topology_id"] for cell in cells.values()})
    ]
    for group, ids in groups:
        counts = Counter(cells[c]["family"] for c in ids)
        weights = np.asarray([1 / (len(counts) * counts[cells[c]["family"]]) for c in ids])
        require(np.isclose(weights.sum(), 1), "family weights do not sum to one")
        for metric in CORE_METRICS:
            for numerator, denominator in CONTRASTS:
                label = f"{numerator}/{denominator}"
                selected_rows = [next(row for row in contrast_rows if row["cell_id"] == c
                                     and row["contrast"] == label and row["metric"] == metric) for c in ids]
                point = np.asarray([row["ratio_of_medians"] for row in selected_rows])
                combined = np.stack([bootstrap_by_key[(c, label, metric)] for c in ids])
                ratio = float(np.exp(weights @ np.log(point)))
                boot = np.exp(weights @ np.log(combined))
                lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                aggregate_rows.append({"group": group, "contrast": label, "metric": metric,
                                       "cells": len(ids), "families": len(counts),
                                       "family_balanced_geometric_ratio": ratio,
                                       "paired_bootstrap_low": float(lo), "paired_bootstrap_high": float(hi),
                                       "worst_cell_ratio": float(point.min()),
                                       "cells_denominator_slower": int(np.sum(point < 1)),
                                       "cells_same_path": sum(row["same_selected_path"] for row in selected_rows)})
    return {"observations": observations, "methods": method_rows,
            "contrasts": contrast_rows, "aggregates": aggregate_rows}


def trace_readout(directory: Path, manifest: dict, coordinator, study: dict, workload: dict) -> list[dict]:
    binding, normalization, _ = coordinator._load_preparation(directory, study, workload)
    profile = coordinator._pretest_profile(directory, study, workload)
    result = []
    for cell_id, cell in sorted(manifest["cells"].items()):
        traces = {}
        for method, objective in (("F", "cotengra_tree_flops_v1"), ("U", "upmem_launch_cost_v1")):
            trace = coordinator._completed_cell_trace(
                directory, "evaluation", cell_id, cell["circuit_id"], study, workload,
                binding, normalization, profile, objective,
            )
            require(coordinator.record_hash(trace) == cell["trace_hashes"][method], "trace hash mismatch")
            traces[method] = trace
            attempts = trace["trace"]
            result.append({"cell_id": cell_id, "method": method,
                           "proposals": len(attempts),
                           "unique_eligible_paths": len({r["path_id"] for r in attempts if r["status"] == "eligible"}),
                           "duplicate_proposals": sum(bool(r["duplicate"]) for r in attempts),
                           "known_infeasible": sum(r["status"] == "known_infeasible" for r in attempts),
                           "search_wall_s": trace["search_wall_s"],
                           **{name + "_sum_s": sum(float(r["timings_s"].get(name, 0)) for r in attempts)
                              for name in ("adaptive_ask", "adaptive_tell", "candidate_generation", "plan_evaluation")}})
        changes = sum(a["params"] != b["params"]
                      for a, b in zip(traces["F"]["trace"][16:], traces["U"]["trace"][16:]))
        for row in result[-2:]:
            row["F_U_parameter_differences_after_startup"] = changes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    impl, directory, output = args.implementation.resolve(), args.directory.resolve(), args.output.resolve()
    require(not output.exists(), "readout output already exists; do not overwrite")
    require(not output.is_relative_to(directory), "keep reports outside the immutable control directory")
    sys.path[:0] = [str(impl / "scripts"), str(impl / "src")]
    import upmem_cost_guided_path as coordinator

    study, workload, budget = coordinator.load_study(impl / "configs/upmem_cost_guided_path_study_v1.json", root=impl)
    coordinator._load_preparation(directory, study, workload)  # Exact source and research environment.
    profile = coordinator._pretest_profile(directory, study, workload)
    manifest, accepted = coordinator._accepted_round(directory, "evaluation", study, workload)
    require(manifest["profile_hash"] == coordinator.record_hash(profile), "evaluation profile mismatch")
    require(len(manifest["cells"]) == budget["evaluation_cells"], "missing evaluation cells")
    summary = summarize(manifest, accepted["rows"])
    traces = trace_readout(directory, manifest, coordinator, study, workload)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in summary.items():
        write_csv(output / f"{name}.csv", rows)
    write_csv(output / "search.csv", traces)
    write_json(output / "accepted_evaluation_rows.json", accepted["rows"])
    feature_rows = []
    for cell_id, cell in sorted(manifest["cells"].items()):
        for method in METHODS:
            p = cell["selection"]["roles"][method]
            f = cell["candidates"][p]["facts"]
            feature_rows.append({"cell_id": cell_id, "method": method, "path_id": p,
                                 **{k: f[k] for k in ("H", "P", "N")},
                                 "M_total": sum(sum(l["M"]) for l in f["launches"]),
                                 "W_total": sum(sum(l["W"]) for l in f["launches"]),
                                 "launch_count": len(f["launches"])})
    write_csv(output / "selected_plan_facts.csv", feature_rows)
    # Descriptive diagnostics only; these never change coefficients or selection.
    measured = {}
    for stage in ("initial", "feedback_1", "feedback_2"):
        development = read_json(directory / f"{stage}_round.json")
        for cell_id, cell in development["cells"].items():
            for path_id, candidate in cell["candidates"].items():
                f = candidate["facts"]
                vector = (f["H"], f["P"], f["N"],
                          sum(sum(l["M"]) for l in f["launches"]),
                          sum(sum(l["W"]) for l in f["launches"]))
                prior = measured.setdefault((cell_id, path_id), vector)
                require(prior == vector, "development facts changed")
    names = ("H", "P", "N", "M", "W")
    x = np.asarray(list(measured.values()), dtype=np.float64)
    constant = [bool(np.all(x[:, k] == x[0, k])) for k in range(5)]
    correlations = [[None if constant[a] or constant[b]
                     else float(np.corrcoef(x[:, a], x[:, b])[0, 1])
                     for b in range(5)] for a in range(5)]
    grid = read_json(directory / "feedback_2_fit/grid.json")
    decisions = {tuple((cell_id, value["path_id"]) for cell_id, value in sorted(row["cells"].items()))
                 for row in grid}
    best_j = max(row["rounded_J"] for row in grid)
    write_json(output / "model_diagnostics.json", {
        "features": names, "pooled_measured_development_pairs": len(measured),
        "constant_features": [name for name, yes in zip(names, constant) if yes],
        "pooled_total_feature_correlations": correlations,
        "grid_size": len(grid), "distinct_grid_selection_vectors": len(decisions),
        "tuples_tied_at_best_rounded_J": sum(row["rounded_J"] == best_j for row in grid),
        "interpretation": "Pooled aggregate feature totals, not per-cell identifiability or a proof of unique physical coefficients; launch maxima remain in the actual model.",
    })
    write_json(output / "provenance.json", {
        "study_hash": coordinator.record_hash(study), "profile": profile,
        "accepted_evaluation_hash": coordinator.record_hash(accepted),
        "evaluation_manifest_hash": coordinator.record_hash(manifest),
        "verified_archives": accepted["archives"], "report_script_sha256": digest(Path(__file__)),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "point_estimator": "ratio_of_arm_medians_within_cell_then_family_balanced_geometric_aggregation",
        "bootstrap_unit": "five complete block IDs, common resample across cells and methods",
        "primary_contrast": "R/U on session_inclusive_s; >1 favors U",
        "interpretation": "Conditional descriptive intervals, fixed six families/two topologies, one search seed; not population-generalization intervals.",
    })
    primary = next(r for r in summary["aggregates"] if r["group"] == "all"
                   and r["contrast"] == "R/U" and r["metric"] == "session_inclusive_s")
    lines = ["# Frozen P6 evaluation readout", "",
             "Primary comparison: R/U; a ratio above one favors UPMEM-guided generation.", "",
             f"Family-balanced geometric ratio: **{primary['family_balanced_geometric_ratio']:.6f}**.",
             f"Descriptive paired-block 95% bootstrap interval: [{primary['paired_bootstrap_low']:.6f}, {primary['paired_bootstrap_high']:.6f}].", "",
             "## Interpretation boundaries", "",
             "All figures are derived from reverified accepted raw evidence. Warmups are excluded.",
             "Method aliases share a physical sample; they are not additional independent observations.",
             "Session-inclusive time is not complete job time. Search cost is reported separately.",
             "Five timing blocks and one paired search-seed schedule do not establish broad population or optimizer-seed robustness.",
             "A neutral or negative result does not authorize further tuning.", "",
             "See methods.csv, contrasts.csv, aggregates.csv, search.csv and selected_plan_facts.csv for the complete numerical result."]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    with (output / "SHA256SUMS").open("x", encoding="ascii") as stream:
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS":
                stream.write(f"{digest(path)}  {path.name}\n")
    print(json.dumps({"output": str(output), "primary_R_over_U": primary}, indent=2))


if __name__ == "__main__":
    main()
```

## Appendix C — Synthetic readout tests

```python
"""Synthetic tests only: no repository, SDK, hardware, or physical observations."""
from copy import deepcopy
import unittest

from p6_readout import summarize


def fixture():
    manifest = {"stage": "evaluation", "measurement_blocks": [1, 2, 3, 4, 5], "cells": {}}
    rows = []
    for family in ("A", "B"):
        for topology in ("1dpu_t8", "4dpu_t8"):
            cell_id = f"{family}/{topology}"
            manifest["cells"][cell_id] = {
                "family": family, "topology_id": topology, "split": "test",
                "candidates": {p: {} for p in ("g", "r", "u")},
                "selection": {"roles": {"G": "g", "F": "g", "R": "r", "U": "u"}},
            }
            for p, inclusive in (("g", 12.0), ("r", 6.0), ("u", 3.0)):
                for b in (0, 1, 2, 3, 4, 5):
                    t = inclusive if b else 99999.0
                    rows.append({"cell_id": cell_id, "path_id": p, "block": b,
                                 "round_id": "evaluation", "split": "test", "status": "success",
                                 "validation": "pass", "fallback": False,
                                 "attempt_type": "measurement" if b else "warmup",
                                 "session_open_s": 0.5, "total_wall_s": t - 1,
                                 "session_close_s": 0.5, "session_inclusive_s": t})
    return manifest, rows


class ReadoutTests(unittest.TestCase):
    def test_exact_contrast_alias_and_warmup_exclusion(self):
        m, rows = fixture()
        r = summarize(m, rows)
        primary = next(x for x in r["aggregates"] if x["group"] == "all"
                       and x["contrast"] == "R/U" and x["metric"] == "session_inclusive_s")
        self.assertAlmostEqual(primary["family_balanced_geometric_ratio"], 2)
        self.assertAlmostEqual(primary["paired_bootstrap_low"], 2)
        self.assertAlmostEqual(primary["paired_bootstrap_high"], 2)
        aliases = [x for x in r["contrasts"] if x["contrast"] == "G/F"]
        self.assertTrue(all(x["same_selected_path"] and x["ratio_of_medians"] == 1 for x in aliases))
        self.assertEqual(len(r["observations"]), 4 * 4 * 5)
        self.assertTrue(all(x["session_inclusive_s"] < 100 for x in r["observations"]))

    def test_family_weights_not_cell_frequency(self):
        m, rows = fixture()
        # Family A gains 4x in both its cells; B regresses to .25x in one cell.
        remove = "B/4dpu_t8"
        del m["cells"][remove]
        rows = [r for r in rows if r["cell_id"] != remove]
        for r in rows:
            if r["path_id"] == "r" and r["block"]:
                t = 12 if r["cell_id"].startswith("A/") else 0.75
                r.update(session_open_s=0, session_close_s=0, total_wall_s=t, session_inclusive_s=t)
        out = summarize(m, rows)
        primary = next(x for x in out["aggregates"] if x["group"] == "all"
                       and x["contrast"] == "R/U" and x["metric"] == "session_inclusive_s")
        self.assertAlmostEqual(primary["family_balanced_geometric_ratio"], 1)

    def test_missing_or_duplicate_measurement_rejected(self):
        m, rows = fixture()
        measured = next(r for r in rows if r["attempt_type"] == "measurement")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(m, [*rows, deepcopy(measured)])
        rows.remove(measured)
        with self.assertRaisesRegex(ValueError, "measurement set"):
            summarize(m, rows)

    def test_wrong_session_boundary_rejected(self):
        m, rows = fixture()
        next(r for r in rows if r["block"] == 1)["session_inclusive_s"] += 1
        with self.assertRaisesRegex(ValueError, "boundary"):
            summarize(m, rows)

    def test_repeatability(self):
        m, rows = fixture()
        self.assertEqual(summarize(m, rows), summarize(m, rows))


if __name__ == "__main__":
    unittest.main()
```
