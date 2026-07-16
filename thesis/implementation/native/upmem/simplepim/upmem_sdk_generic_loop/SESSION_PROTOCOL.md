# Persistent Generic Session Protocol

The host keeps the existing six-argument single-operation invocation unchanged.
The additive session invocation is:

    host --session-manifest session.json --response-manifest response.json

The input is upmem_generic_session_v1 JSON:

    {
      "schema_version": "upmem_generic_session_v1",
      "manifest_kind": "upmem_generic_session_input",
      "session_id": "example",
      "dpu_binary": "build/bin/dpu_generic",
      "requested_dpus": 1,
      "tasklets": 1,
      "tasks": [
        {
          "task_id": "task-0",
          "args_path": "inputs/task-0_args.bin",
          "left_path": "inputs/task-0_left.bin",
          "right_path": "inputs/task-0_right.bin",
          "output_path": "outputs/task-0_output.bin"
        }
      ]
    }

All paths are relative to the session manifest. Absolute paths and .. path
components are rejected. args_path contains the existing packed
upmem_generic_args_t; the task's operand mode selects either float32 inputs
and float32 output or int8 inputs and int32 output. The native metadata and
size/stride validation are the same contract used by the single-operation
path.

The host preflights every task, calls dpu_alloc(1, profile, ...) and
dpu_load(...) once, then processes the array strictly in order. Each task
broadcasts its metadata and operands, uses DPU_SYNCHRONOUS, copies the
result, and writes its output before the next task begins. Any failure stops
the batch; later tasks are returned as not_run. The set is released on every
post-allocation exit path. No CPU, simulator, or other fallback is selected by
this protocol.

The response is upmem_generic_session_response JSON. It reports session
allocation/load/batch/release timings and one ordered task record containing
status, failure stage, SDK error code, stage timings, and an output reference
(path and byte count) for successful tasks. The response is a host-side SDK
record, not a physical bus counter or a performance claim.

## Interactive session

The additive interactive invocation is:

    host --interactive-session --bootstrap-manifest bootstrap.json

The bootstrap manifest is `generic_loop_interactive_session_v1` with
`manifest_kind` `bootstrap`, a safe relative `dpu_binary`,
`requested_dpus: 1`, and `tasklets: 1`. After one allocation and one binary
load, the host writes one JSON readiness event to stdout and flushes it:

    {"schema_version":"generic_loop_interactive_session_v1","event":"ready",
     "status":"ready","requested_dpus":1,"allocated_dpus":1,
     "allocation_time_s":0.0,"binary_load_time_s":0.0}

The controller writes newline-delimited commands to stdin:

    REQUEST request.json response.json
    CLOSE

Both command paths are safe relative paths under the bootstrap manifest
directory. A request manifest uses `manifest_kind` `request`, a `request_id`,
`requested_dpus: 1`, `tasklets: 1`, and an ordered `tasks` array containing
exactly one or four component tasks. Task paths are safe relative paths under
the request manifest directory. The response uses `manifest_kind` `response`
and preserves each task's output path plus input-read, H2D, synchronous kernel,
D2H, output-write, and total timings.

Each request response is flushed to its response path and acknowledged by a
JSON stdout event containing the actual response relative path. `CLOSE`,
request failure, protocol failure, and EOF release the DPU set. The `closed`
event reports `released: true` only after `dpu_free` succeeds. A timeout is not
treated as proof of release. This protocol never selects a simulator or CPU
fallback and does not alter the existing `--session-manifest` batch protocol.

The initial contract deliberately requires one DPU and one tasklet. The
compiled UPMEM_GENERIC_HARDWARE_MVP=1 profile continues to select the existing
physical backend=hw allocation profile; simulator and hardware selection remain
build/environment responsibilities, and this mode never falls back between
them.
