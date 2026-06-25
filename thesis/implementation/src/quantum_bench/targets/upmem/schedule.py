from __future__ import annotations

from dataclasses import dataclass

from quantum_bench.core.records import ContractionTask, JsonDict, TaskGraph


@dataclass(frozen=True)
class UpmemHardwareProfile:
    name: str
    wram_bytes: int
    dpu_count: int | None = None


@dataclass(frozen=True)
class UpmemDataFormat:
    name: str
    input_element_bytes: int
    output_element_bytes: int
    accumulator_element_bytes: int


@dataclass(frozen=True)
class UpmemTaskEstimate:
    task_id: str
    gemm_m: int
    gemm_k: int
    gemm_n: int
    left_bytes: int
    right_bytes: int
    output_bytes: int
    accumulator_bytes: int
    working_set_bytes: int
    host_to_dpu_bytes: int
    dpu_to_host_bytes: int
    mram_to_wram_bytes: int
    fits_wram_without_tiling: bool
    reject_reason: str | None


@dataclass(frozen=True)
class UpmemScheduleEstimate:
    route_family: str
    hardware: UpmemHardwareProfile
    data_format: UpmemDataFormat
    tasks: tuple[UpmemTaskEstimate, ...]
    total_host_to_dpu_bytes: int
    total_dpu_to_host_bytes: int
    total_mram_to_wram_bytes: int
    max_working_set_bytes: int
    tasks_fit_without_tiling: int
    tasks_requiring_tiling: int

    @property
    def all_tasks_fit_without_tiling(self) -> bool:
        return self.tasks_requiring_tiling == 0

    def first_reject_reason(self) -> str | None:
        for task in self.tasks:
            if task.reject_reason:
                return task.reject_reason
        return None

    def notes(self) -> tuple[str, ...]:
        return (
            f"UPMEM target layer: {self.hardware.name}, {self.hardware.wram_bytes} B WRAM",
            f"dense schedule estimate: {self.tasks_fit_without_tiling}/{len(self.tasks)} tasks fit without tiling",
            f"estimated H2D={self.total_host_to_dpu_bytes} B, D2H={self.total_dpu_to_host_bytes} B",
        )

    def metadata(self) -> JsonDict:
        return {
            "target": "upmem",
            "route_family": self.route_family,
            "hardware": {
                "name": self.hardware.name,
                "wram_bytes": self.hardware.wram_bytes,
                "dpu_count": self.hardware.dpu_count,
            },
            "data_format": {
                "name": self.data_format.name,
                "input_element_bytes": self.data_format.input_element_bytes,
                "output_element_bytes": self.data_format.output_element_bytes,
                "accumulator_element_bytes": self.data_format.accumulator_element_bytes,
            },
            "task_count": len(self.tasks),
            "tasks_fit_without_tiling": self.tasks_fit_without_tiling,
            "tasks_requiring_tiling": self.tasks_requiring_tiling,
            "total_host_to_dpu_bytes": self.total_host_to_dpu_bytes,
            "total_dpu_to_host_bytes": self.total_dpu_to_host_bytes,
            "total_mram_to_wram_bytes": self.total_mram_to_wram_bytes,
            "max_working_set_bytes": self.max_working_set_bytes,
            "first_reject_reason": self.first_reject_reason(),
        }


UPMEM_PROFILE = UpmemHardwareProfile(name="UPMEM DPU", wram_bytes=64 * 1024)
DENSE_INT8_FORMAT = UpmemDataFormat(
    name="int8_inputs_int32_accumulator",
    input_element_bytes=1,
    output_element_bytes=4,
    accumulator_element_bytes=4,
)


def estimate_dense_task(
    task: ContractionTask,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemTaskEstimate:
    left_elements = task.gemm_m * task.gemm_k
    right_elements = task.gemm_k * task.gemm_n
    output_elements = task.gemm_m * task.gemm_n
    left_bytes = left_elements * data_format.input_element_bytes
    right_bytes = right_elements * data_format.input_element_bytes
    output_bytes = output_elements * data_format.output_element_bytes
    accumulator_bytes = output_elements * data_format.accumulator_element_bytes
    working_set_bytes = left_bytes + right_bytes + output_bytes + accumulator_bytes
    fits = working_set_bytes <= hardware.wram_bytes
    reject_reason = None
    if not fits:
        reject_reason = (
            f"task {task.id} dense tile needs {working_set_bytes} B WRAM "
            f"before tiling; limit is {hardware.wram_bytes} B"
        )
    return UpmemTaskEstimate(
        task_id=task.id,
        gemm_m=task.gemm_m,
        gemm_k=task.gemm_k,
        gemm_n=task.gemm_n,
        left_bytes=left_bytes,
        right_bytes=right_bytes,
        output_bytes=output_bytes,
        accumulator_bytes=accumulator_bytes,
        working_set_bytes=working_set_bytes,
        host_to_dpu_bytes=left_bytes + right_bytes,
        dpu_to_host_bytes=output_bytes,
        mram_to_wram_bytes=left_bytes + right_bytes + output_bytes,
        fits_wram_without_tiling=fits,
        reject_reason=reject_reason,
    )


def estimate_dense_task_graph(
    graph: TaskGraph,
    hardware: UpmemHardwareProfile = UPMEM_PROFILE,
    data_format: UpmemDataFormat = DENSE_INT8_FORMAT,
) -> UpmemScheduleEstimate:
    tasks = tuple(estimate_dense_task(task, hardware, data_format) for task in graph.tasks)
    return UpmemScheduleEstimate(
        route_family="dense_gemm",
        hardware=hardware,
        data_format=data_format,
        tasks=tasks,
        total_host_to_dpu_bytes=sum(task.host_to_dpu_bytes for task in tasks),
        total_dpu_to_host_bytes=sum(task.dpu_to_host_bytes for task in tasks),
        total_mram_to_wram_bytes=sum(task.mram_to_wram_bytes for task in tasks),
        max_working_set_bytes=max((task.working_set_bytes for task in tasks), default=0),
        tasks_fit_without_tiling=sum(1 for task in tasks if task.fits_wram_without_tiling),
        tasks_requiring_tiling=sum(1 for task in tasks if not task.fits_wram_without_tiling),
    )
