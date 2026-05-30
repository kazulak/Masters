# Data Formats

Future home for reduced-precision and integer-native format experiments.

Responsibilities:

- define host tensor format metadata;
- quantize and dequantize;
- define DPU transfer layout;
- define accumulation type;
- record scale scope and scale values;
- report numerical error;
- reject illegal route-format combinations.

Initial baseline:

- `complex_f64_host` for reference execution;
- `complex_i8_tile_scaled` matching the current MVP behavior.

Candidate next formats:

- fixed-point;
- block-floating-point after fixed-point is stable;
- library-backed integer or mixed-precision format if integration is practical.

No DPU route may hide its numerical format. Any non-reference format must produce
a validation record against CPU reference output.
