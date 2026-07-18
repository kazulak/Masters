from __future__ import annotations

import pytest

from .support import (
    cpu_gpu_pair_records,
    hardware_evidence_records,
    minimal_real_graph,
    planner_evidence_records,
    resident_suite,
    split_complex_graph,
    tn_upmem_pair_records,
    valid_resident_response,
)


@pytest.fixture
def minimal_graph():
    return minimal_real_graph()


@pytest.fixture
def split_complex_graph_fixture():
    return split_complex_graph()


@pytest.fixture
def valid_resident_response_fixture():
    return valid_resident_response


@pytest.fixture
def cpu_gpu_pair_evidence():
    return cpu_gpu_pair_records()


@pytest.fixture
def tn_upmem_pair_evidence():
    return tn_upmem_pair_records()


@pytest.fixture
def planner_hardware_evidence():
    return {"planner": planner_evidence_records(), "hardware": hardware_evidence_records()}


@pytest.fixture
def resident_hardware_suite():
    return resident_suite()
