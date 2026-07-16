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

The initial contract deliberately requires one DPU and one tasklet. The
compiled UPMEM_GENERIC_HARDWARE_MVP=1 profile continues to select the existing
physical backend=hw allocation profile; simulator and hardware selection remain
build/environment responsibilities, and this mode never falls back between
them.
