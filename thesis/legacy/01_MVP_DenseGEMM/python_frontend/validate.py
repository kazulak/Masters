#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys

import numpy as np


def read_output(path):
    with open(path, "rb") as f:
        n_elem = struct.unpack("<i", f.read(4))[0]
        real_part = np.frombuffer(f.read(n_elem * 8), dtype="<f8")
        imag_part = np.frombuffer(f.read(n_elem * 8), dtype="<f8")
    return real_part + 1j * imag_part


def compute_metrics(result, reference):
    diff = result - reference
    max_abs_err = float(np.max(np.abs(diff))) if diff.size else 0.0
    max_ref = float(np.max(np.abs(reference))) if reference.size else 0.0
    max_rel_err = max_abs_err / max_ref if max_ref > 1e-15 else max_abs_err
    result_norm = float(np.linalg.norm(result))
    reference_norm = float(np.linalg.norm(reference))
    norm_drift = abs(result_norm - reference_norm)

    fidelity = None
    if result.shape == reference.shape and reference_norm > 1e-15 and result_norm > 1e-15:
        overlap = np.vdot(reference / reference_norm, result / result_norm)
        fidelity = float(min(1.0, np.abs(overlap) ** 2))

    return {
        "max_abs_error": max_abs_err,
        "max_rel_error": float(max_rel_err),
        "norm_drift": float(norm_drift),
        "fidelity": fidelity,
        "result_norm": result_norm,
        "reference_norm": reference_norm,
    }


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(
        description="Validate MVP UPMEM amplitudes against the NumPy reference."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(root, "data_exchange", "output_amplitudes.bin"),
        help="Path to mvp_host output_amplitudes.bin.",
    )
    parser.add_argument(
        "--reference",
        default=os.path.join(root, "data_exchange", "reference_output.npy"),
        help="Path to NumPy reference_output.npy.",
    )
    parser.add_argument(
        "--record",
        default=os.path.join(root, "data_exchange", "validation_record.json"),
        help="Path for the machine-readable validation record.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Maximum relative error tolerated for the int8 MVP baseline.",
    )
    args = parser.parse_args()

    upmem_result = read_output(args.output)
    reference = np.load(args.reference).flatten()

    print("UPMEM result:    ", upmem_result)
    print("NumPy reference: ", reference)

    if upmem_result.shape != reference.shape:
        record = {
            "schema_version": "mvp_validation_record-0.2",
            "status": "fail",
            "reason": "shape_mismatch",
            "output_path": args.output,
            "reference_path": args.reference,
            "result_shape": list(upmem_result.shape),
            "reference_shape": list(reference.shape),
        }
        with open(args.record, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print("[validate] FAIL - shape mismatch")
        return 1

    metrics = compute_metrics(upmem_result, reference)
    passed = metrics["max_rel_error"] < args.tolerance

    print(f"Max absolute error: {metrics['max_abs_error']:.6e}")
    print(f"Max relative error: {metrics['max_rel_error']:.4%}")
    print(f"Norm drift:         {metrics['norm_drift']:.6e}")
    if metrics["fidelity"] is not None:
        print(f"Fidelity:           {metrics['fidelity']:.12f}")

    record = {
        "schema_version": "mvp_validation_record-0.2",
        "status": "pass" if passed else "fail",
        "reference_route": "cpu_reference_numpy",
        "compared_route": "raw_upmem_dense",
        "data_format": "complex_i8_tile_scaled",
        "output_path": args.output,
        "reference_path": args.reference,
        "tolerance": {
            "max_rel_error": args.tolerance,
        },
        "metrics": metrics,
    }
    with open(args.record, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    print(f"[validate] Wrote {args.record}")
    if passed:
        print("[validate] PASS - amplitude within int8 tolerance")
        return 0

    print(
        "[validate] FAIL - error "
        f"{metrics['max_rel_error']:.2%} exceeds tolerance {args.tolerance:.0%}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
