"""Private v5 launch controls; not enabled in the active v4 execution route."""

from dataclasses import dataclass
from struct import Struct


VERSION = 5
CONTROL_MAGIC = 0x35574354
MRAM_BYTES = 512 * 1024
NO_OPERATION = (1 << 32) - 1
IDLE = 1
REAL_PANEL = 1
FOUR_PRODUCT_PANEL = 2
MAX_K = 65536
INT8_COMPONENT_PRODUCT = 2 * 127 * 127
CONTROL = Struct("<8I3Q6I16I")
PLANE_NAMES = ("a_real", "a_imag", "b_real", "b_imag", "rr", "ii", "ri", "ir")


def _uint(value: int, bits: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must be uint{bits}")


def aligned_bytes(size: int) -> int:
    _uint(size, 32, "size")
    result = (size + 7) // 8 * 8
    _uint(result, 32, "aligned size")
    return result


def product_layout(m: int, n: int, k: int, *, numeric_mode: int,
                   kernel: int) -> tuple[tuple[int, int], ...]:
    """Pack only admitted planes, without changing tile or reduction geometry."""
    for name, value, limit in (("m", m, 256), ("n", n, 256), ("k", k, MAX_K)):
        _uint(value, 32, name)
        if not 1 <= value <= limit:
            raise ValueError(f"{name} exceeds the admitted tile geometry")
    if type(numeric_mode) is not int or numeric_mode not in (0, 1):
        raise ValueError("unknown numeric mode")
    if type(kernel) is not int or kernel not in (REAL_PANEL, FOUR_PRODUCT_PANEL):
        raise ValueError("unknown kernel")
    if k * INT8_COMPONENT_PRODUCT > (1 << 31) - 1:
        raise ValueError("int8 component accumulation exceeds int32")
    element_bytes = 4 if numeric_mode == 0 else 1
    a, b, c = (aligned_bytes(size) for size in
               (m * k * element_bytes, k * n * element_bytes, m * n * 4))
    sizes = (a, a, b, b, c, c, c, c) if kernel == FOUR_PRODUCT_PANEL else (
        a, 0, b, 0, c, 0, 0, 0)
    cursor = 0
    spans = []
    for size in sizes:
        if size > MRAM_BYTES - cursor:
            raise ValueError("kernel working set exceeds the MRAM arena")
        spans.append((cursor, size) if size else (0, 0))
        cursor += size
    return tuple(spans)


@dataclass(frozen=True)
class WaveControl:
    dpu_id: int
    tasklets: int
    flags: int
    numeric_mode: int
    kernel: int
    operation_index: int
    wave_id: int
    request_sequence: int
    tile_id: int
    batch_index: int
    m: int
    n: int
    k: int
    k_offset: int
    planes: tuple[tuple[int, int], ...]

    def validate(self) -> None:
        for name in ("dpu_id", "tasklets", "flags", "numeric_mode", "kernel",
                     "operation_index", "batch_index", "m", "n", "k", "k_offset"):
            _uint(getattr(self, name), 32, name)
        for name in ("wave_id", "request_sequence", "tile_id"):
            _uint(getattr(self, name), 64, name)
        if self.dpu_id >= 64 or not 1 <= self.tasklets <= 24:
            raise ValueError("invalid DPU/tasklet identity")
        if self.numeric_mode not in (0, 1) or self.flags not in (0, IDLE):
            raise ValueError("unsupported mode or flags")
        if type(self.planes) is not tuple or len(self.planes) != 8:
            raise ValueError("exactly eight immutable plane descriptors required")
        for span in self.planes:
            if type(span) is not tuple or len(span) != 2:
                raise ValueError("invalid immutable plane descriptor")
            _uint(span[0], 32, "plane offset")
            _uint(span[1], 32, "plane length")
        if self.flags == IDLE:
            if (self.kernel != 0 or self.operation_index != NO_OPERATION or
                    any((self.batch_index, self.m, self.n, self.k, self.k_offset,
                         self.tile_id)) or self.planes != ((0, 0),) * 8):
                raise ValueError("idle descriptor contains work")
            return
        if self.operation_index >= 64:
            raise ValueError("operation index exceeds wave capacity")
        if self.k_offset + self.k > MAX_K:
            raise ValueError("reduction range exceeds admitted K")
        expected = product_layout(self.m, self.n, self.k,
                                  numeric_mode=self.numeric_mode, kernel=self.kernel)
        regions = []
        for (offset, length), (_, required) in zip(self.planes, expected):
            if length != required or (length == 0 and offset != 0):
                raise ValueError("plane length differs from kernel geometry")
            if offset > MRAM_BYTES or length > MRAM_BYTES - offset or offset % 8:
                raise ValueError("unaligned or out-of-bounds plane")
            if length:
                regions.append((offset, offset + length))
        regions.sort()
        if any(left[1] > right[0] for left, right in zip(regions, regions[1:])):
            raise ValueError("overlapping input/output planes")

    def to_bytes(self) -> bytes:
        self.validate()
        return CONTROL.pack(
            CONTROL_MAGIC, VERSION, self.dpu_id, self.tasklets, self.flags,
            self.numeric_mode, self.kernel, self.operation_index,
            self.wave_id, self.request_sequence, self.tile_id,
            self.batch_index, self.m, self.n, self.k, self.k_offset, 0,
            *(value for span in self.planes for value in span),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "WaveControl":
        if len(data) != CONTROL.size:
            raise ValueError("wrong control length")
        values = CONTROL.unpack(data)
        if values[:2] != (CONTROL_MAGIC, VERSION) or values[16] != 0:
            raise ValueError("unsupported control identity or reserved field")
        result = cls(*values[2:16], tuple(zip(values[17::2], values[18::2])))
        result.validate()
        return result
