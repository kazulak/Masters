#!/usr/bin/env python3
import os
import struct

import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(root, "data_exchange", "output_amplitudes.bin"), "rb") as f:
    n_elem = struct.unpack("<i", f.read(4))[0]
    real_part = np.frombuffer(f.read(n_elem * 8), dtype="<f8")
    imag_part = np.frombuffer(f.read(n_elem * 8), dtype="<f8")

upmem_result = real_part + 1j * imag_part
ref = np.load(os.path.join(root, "data_exchange", "reference_output.npy")).flatten()

print("UPMEM result:    ", upmem_result)
print("NumPy reference: ", ref)

max_abs_err = np.max(np.abs(upmem_result - ref))
max_ref = np.max(np.abs(ref))
rel_err = max_abs_err / max_ref if max_ref > 1e-15 else max_abs_err

print(f"Max absolute error: {max_abs_err:.6e}")
print(f"Max relative error: {rel_err:.4%}")

TOLERANCE = 0.02
if rel_err < TOLERANCE:
    print("✓ PASS — amplitude within int8 tolerance")
else:
    print(f"✗ FAIL — error {rel_err:.2%} exceeds tolerance {TOLERANCE:.0%}")
