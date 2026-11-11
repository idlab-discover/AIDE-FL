"""Publishing must reject incomplete or modified experimental evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'wp4_publish', Path(__file__).parents[1] / 'experiments/wp4/publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


def evidence(root):
    (root / 'config.json').write_text(json.dumps({'rounds': 2}))
    (root / 'metrics.jsonl').write_text('{"round": 1}\n{"round": 2}\n')
    (root / 'summary.json').write_text('{"test_f1": 0.5}')
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    (root / 'checksums.json').write_text(json.dumps(checksums))


def test_reads_verified_runs(tmp_path):
    evidence(tmp_path)
    assert len(publish.read_runs(tmp_path)) == 1


def test_rejects_changed_summary(tmp_path):
    evidence(tmp_path)
    (tmp_path / 'summary.json').write_text('{"test_f1": 1.0}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        publish.read_runs(tmp_path)


def test_rejects_duplicate_rounds(tmp_path):
    evidence(tmp_path)
    with pytest.raises(ValueError, match='duplicate training rounds'):
        publish.validate_evidence(tmp_path, {'rounds': 2}, [{'round': 1}, {'round': 1}])


def test_remote_float_serialization_tolerance():
    assert publish.same_measurement(1.23456789012345, 1.2345678901234567)
    assert not publish.same_measurement(0.51, 0.5)
    assert not publish.same_measurement(None, 0.5)
    assert not publish.same_measurement([[1, 2]], [[1, 3]])


def test_nested_summary_matches_public_api_keys():
    assert dict(publish.summary_items({'counts': {'0': 12}, 'f1': .5})) == {
        'counts.0': 12, 'f1': .5}
