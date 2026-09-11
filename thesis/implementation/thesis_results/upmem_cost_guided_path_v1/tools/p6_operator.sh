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
