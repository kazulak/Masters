"""Private v5 launch controls and completions; not enabled in active v4."""

from dataclasses import dataclass
from struct import Struct


VERSION = 5
CONTROL_MAGIC = 0x35574354
COMPLETION_MAGIC = 0x35574350
MRAM_BYTES = 512 * 1024
NO_OPERATION = (1 << 32) - 1
NO_PRODUCT = (1 << 32) - 1
IDLE = 1
REAL_PANEL = 1
FOUR_PRODUCT_PANEL = 2
MAX_K = 65536
INT8_COMPONENT_PRODUCT = 2 * 127 * 127
CONTROL = Struct("<8I3Q6I16I")
COMPLETION = Struct("<6I5Q2I")
PLANE_NAMES = ("a_real", "a_imag", "b_real", "b_imag", "rr", "ii", "ri", "ir")

STATUS_PENDING = 0
STATUS_COMPLETED = 1
STATUS_FAILED = 2

FAILURE_NONE = 0
FAILURE_VALIDATION = 1
FAILURE_EXECUTION = 2

COMPLETED_MASK_IDLE = 0
COMPLETED_MASK_REAL = 1
COMPLETED_MASK_FUSED = 15
PRODUCT_COUNT_IDLE = 0
PRODUCT_COUNT_REAL = 1
PRODUCT_COUNT_FUSED = 4

COMPLETION_BYTES = COMPLETION.size
_PREFIX_MASKS = frozenset((0, 1, 3, 7, 15))
_SUCCESS_MASKS = frozenset((COMPLETED_MASK_IDLE, COMPLETED_MASK_REAL,
                            COMPLETED_MASK_FUSED))


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


@dataclass(frozen=True)
class WaveCompletion:
    """Private v5 completion record and its wave-level invariants."""

    status: int
    dpu_id: int
    operation_index: int
    completed_product_mask: int
    wave_id: int
    request_sequence: int
    tile_id: int
    cycles: int
    processed_elements: int
    failure_stage: int
    failing_product: int

    def _validate_structure(self) -> None:
        for name in ("status", "dpu_id", "operation_index", "completed_product_mask",
                     "failure_stage", "failing_product"):
            _uint(getattr(self, name), 32, name)
        for name in ("wave_id", "request_sequence", "tile_id", "cycles",
                     "processed_elements"):
            _uint(getattr(self, name), 64, name)
        if self.status not in (STATUS_PENDING, STATUS_COMPLETED, STATUS_FAILED):
            raise ValueError("invalid completion status")
        if self.failure_stage not in (FAILURE_NONE, FAILURE_VALIDATION,
                                      FAILURE_EXECUTION):
            raise ValueError("invalid completion failure stage")

        if self.status == STATUS_PENDING:
            if (self.completed_product_mask not in _PREFIX_MASKS or
                    self.failure_stage != FAILURE_NONE or
                    self.failing_product != NO_PRODUCT):
                raise ValueError("pending completion contains progress or failure")
            if (self.completed_product_mask == COMPLETED_MASK_IDLE and
                    self.processed_elements != 0):
                raise ValueError("pending completion count disagrees with mask")
            return

        if self.status == STATUS_COMPLETED:
            if self.completed_product_mask not in _SUCCESS_MASKS:
                raise ValueError("invalid completed product mask")
            if self.completed_product_mask == COMPLETED_MASK_IDLE and self.processed_elements:
                raise ValueError("idle completion processed elements must be zero")
            if (self.failure_stage != FAILURE_NONE or
                    self.failing_product != NO_PRODUCT):
                raise ValueError("completed completion contains failure metadata")
            return

        if self.failure_stage == FAILURE_VALIDATION:
            if (self.completed_product_mask != COMPLETED_MASK_IDLE or
                    self.processed_elements != 0 or
                    self.failing_product != NO_PRODUCT):
                raise ValueError("validation failure contains execution progress")
            return
        if self.failure_stage != FAILURE_EXECUTION:
            raise ValueError("failed completion requires a failure stage")
        if not 0 <= self.failing_product < PRODUCT_COUNT_FUSED:
            raise ValueError("failing product exceeds fused product count")
        expected_mask = (1 << self.failing_product) - 1
        if self.completed_product_mask != expected_mask:
            raise ValueError("failed completion mask is not a completed prefix")

    def validate(
        self, control: WaveControl | None = None, *, require_success: bool = False
    ) -> None:
        """Validate this record, optionally correlating it with a wave control."""

        if type(require_success) is not bool:
            raise ValueError("require_success must be bool")
        if require_success and control is None:
            raise ValueError("require_success needs a WaveControl")
        self._validate_structure()
        if control is None:
            return
        if not isinstance(control, WaveControl):
            raise TypeError("completion correlation requires a WaveControl")
        control.validate()
        for name in ("dpu_id", "operation_index", "wave_id", "request_sequence",
                     "tile_id"):
            if getattr(self, name) != getattr(control, name):
                raise ValueError(f"completion {name} does not match control")
        if require_success and self.status != STATUS_COMPLETED:
            raise ValueError("terminal correlation requires completed status")

        if control.flags == IDLE:
            product_count = PRODUCT_COUNT_IDLE
        elif control.kernel == REAL_PANEL:
            product_count = PRODUCT_COUNT_REAL
        else:
            product_count = PRODUCT_COUNT_FUSED
        expected_mask = (1 << product_count) - 1 if product_count else 0
        elements_per_product = control.m * control.n if product_count else 0

        if self.status == STATUS_PENDING:
            completed_count = self.completed_product_mask.bit_count()
            if (completed_count > product_count or
                    self.processed_elements != elements_per_product * completed_count):
                raise ValueError("pending progress does not match control")
        elif self.status == STATUS_COMPLETED:
            if (self.completed_product_mask != expected_mask or
                    self.processed_elements != elements_per_product * product_count):
                raise ValueError("completed progress does not match control")
        elif self.status == STATUS_FAILED and self.failure_stage == FAILURE_EXECUTION:
            if self.failing_product >= product_count:
                raise ValueError("failing product exceeds control product count")
            prefix_mask = (1 << self.failing_product) - 1
            expected_elements = elements_per_product * self.failing_product
            if (self.completed_product_mask != prefix_mask or
                    self.processed_elements != expected_elements):
                raise ValueError("failed progress does not match control")

    def to_bytes(
        self, control: WaveControl | None = None, *, require_success: bool = False
    ) -> bytes:
        self.validate(control, require_success=require_success)
        return COMPLETION.pack(
            COMPLETION_MAGIC, VERSION, self.status, self.dpu_id,
            self.operation_index, self.completed_product_mask, self.wave_id,
            self.request_sequence, self.tile_id, self.cycles,
            self.processed_elements, self.failure_stage, self.failing_product,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        control: WaveControl | None = None,
        *,
        require_success: bool = False,
    ) -> "WaveCompletion":
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("completion data must be bytes-like")
        if len(data) != COMPLETION.size:
            raise ValueError("wrong completion length")
        values = COMPLETION.unpack(bytes(data))
        if values[:2] != (COMPLETION_MAGIC, VERSION):
            raise ValueError("unsupported completion identity")
        result = cls(*values[2:])
        result.validate(control, require_success=require_success)
        return result
