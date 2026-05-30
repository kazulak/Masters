# Data Formats

Future home for reduced-precision and integer-native format experiments.

Responsibilities:

- define host tensor format metadata;
- quantize and dequantize;
- define DPU transfer layout;
- define accumulation type;
- report numerical error;
- reject illegal route-format combinations.

Initial baseline:

- `complex_f64_host` for reference execution;
- `complex_i8_tile_scaled` matching the current MVP behavior.

Candidate next formats:

- fixed-point;
- block-floating-point;
- library-backed integer or mixed-precision format if integration is practical.
