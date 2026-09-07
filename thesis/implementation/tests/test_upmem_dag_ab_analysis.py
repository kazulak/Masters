import copy

import pytest

from scripts import analyze_upmem_dag_ab as dag
from scripts import analyze_upmem_fusion_ab as shared


def rows():
    return [dict(case_id=case, dpu_count=d, arm=arm, attempt_kind=kind,
                 block_id=block, sample_index=0 if kind == 'warmup' else block - 1,
                 steady_s=2.0 if arm == 'serial' else 1.0,
                 session_open_s=0.25, session_close_s=0.25, kernel_s=0.5)
            for case in shared.EXPECTED_CASES for d in (2, 4) for arm in ('serial', 'dag')
            for kind, blocks in (('warmup', (0,)), ('measurement', range(1, 6))) for block in blocks]


def test_declared_dag_topologies_and_paired_scopes():
    result = dag.analyze_rows(rows(), bootstrap_resamples=32)
    assert result['dpu_counts'] == [2, 4] and result['arms'] == ['serial', 'dag']
    assert result['row_count'] == 72 and len(result['cells']) == 6
    for cell in result['cells']:
        assert cell['comparisons']['steady_s']['primary_speedup'] == 2
        assert cell['comparisons']['session_inclusive_s']['primary_speedup'] == pytest.approx(2.5 / 1.5)
        assert cell['comparisons']['kernel_s']['primary_speedup'] == 1


def test_reject_old_resource_matrix_in_dag_packet():
    data = rows()
    for row in data:
        if row['dpu_count'] == 2:
            row['dpu_count'] = 1
    with pytest.raises(ValueError, match='unexpected dpu_count'):
        dag.analyze_rows(data, bootstrap_resamples=8)


@pytest.mark.parametrize('counts', [(2, 2), (4, 2), (True, 4), (0, 4), (2,), (2, 4, 8), '24'])
def test_reject_invalid_resource_definition(counts):
    with pytest.raises(ValueError, match='dpu_counts'):
        shared.analyze_rows(rows(), arm_labels=('serial', 'dag'), dpu_counts=counts)


def test_default_matrix_is_unchanged_when_explicit():
    data = copy.deepcopy(rows())
    for row in data:
        row['dpu_count'] = 1 if row['dpu_count'] == 2 else 4
        row['arm'] = 'unfused' if row['arm'] == 'serial' else 'fused'
    assert shared.analyze_rows(data, bootstrap_resamples=128) == shared.analyze_rows(
        data, bootstrap_resamples=128, dpu_counts=(1, 4))
