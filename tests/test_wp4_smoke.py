"""Scientific split invariants for the WP4 smoke comparison."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('wp4_smoke', Path(__file__).parents[1] / 'experiments/wp4/smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def fixture():
    count = 4000
    frame = pd.DataFrame({'row_id': np.arange(count), 'UTC': np.repeat(np.arange(2000), 2),
                          smoke.LABEL: np.arange(count) % 2})
    for feature in smoke.FEATURES:
        frame[feature] = np.arange(count, dtype=float)
    return frame


def test_cached_source_has_no_shared_extracted_csv(tmp_path):
    frame = fixture()
    csv_bytes = frame.to_csv(index=False).encode()
    with smoke.zipfile.ZipFile(tmp_path / 'smoke-detection-kaggle.zip', 'w') as archive:
        archive.writestr('smoke_detection_iot.csv', csv_bytes)
    # A stale or partly written extraction must not affect a new run.
    (tmp_path / 'smoke_detection_iot.csv').write_text('')
    loaded, provenance = smoke.source_data(tmp_path)
    assert len(loaded) == len(frame)
    assert provenance['csv_sha256'] == smoke.hashlib.sha256(csv_bytes).hexdigest()
    assert (tmp_path / 'smoke_detection_iot.csv').read_bytes() == b''


def test_temporal_split_preserves_timestamps_and_embargo():
    result = smoke.split_data(fixture(), 2, 17, 'iid')
    train = result[result.partition == 'train']
    validation = result[result.partition == 'validation']
    test = result[result.partition == 'test']
    assert train.UTC.max() < validation.UTC.min() - 60
    assert validation.UTC.max() < test.UTC.min() - 60
    assert result.groupby('UTC').partition.nunique().max() == 1
    assert set(train.row_id).isdisjoint(validation.row_id)
    assert set(train.row_id).isdisjoint(test.row_id)
    assert set(validation.row_id).isdisjoint(test.row_id)
    assert set(train.client) == {0, 1}
    assert (result[result.partition != 'train'].client == -1).all()


def test_seed_changes_only_participant_assignment():
    first = smoke.split_data(fixture(), 2, 17, 'iid')
    same = smoke.split_data(fixture(), 2, 17, 'iid')
    other = smoke.split_data(fixture(), 2, 42, 'iid')
    pd.testing.assert_frame_equal(first, same)
    pd.testing.assert_series_equal(first.partition, other.partition)
    assert not first.client.equals(other.client)


def test_chronological_clients_have_disjoint_ordered_training_rows():
    result = smoke.split_data(fixture(), 2, 17, 'chronological')
    assert result[result.client == 0].UTC.max() <= result[result.client == 1].UTC.min()
    assert set(result[result.client == 0].row_id).isdisjoint(result[result.client == 1].row_id)


def test_real_training_records_global_model_metrics_and_reloads(tmp_path):
    frame = fixture().iloc[:-2].copy()
    rng = np.random.default_rng(17)
    for feature in smoke.FEATURES:
        frame[feature] = frame[smoke.LABEL] * 2 + rng.normal(size=len(frame))
    prepared = smoke.split_data(frame, 4, 17, 'iid')
    assert prepared[prepared.partition == 'train'].groupby('client').size().nunique() > 1
    # A shifted future period must not enter the fitted training transform.
    prepared.loc[prepared.partition == 'test', smoke.FEATURES] += 10
    for method in ('local-0', 'centralised', 'fedavg'):
        output = tmp_path / method
        smoke.run_method(output, prepared, {
            'method': method, 'seed': 17, 'rounds': 2, 'local_epochs': 5, 'clients': 4,
        })
        summary = json.loads((output / 'summary.json').read_text())
        history = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        assert summary['coefficient_l2_norm'] > 0
        assert summary['validation_logloss'] < summary['initial_validation_logloss']
        assert [row['round'] for row in history] == [1, 2]
        assert all(np.isfinite(row['global_train_logloss']) for row in history)
        bundle = smoke.joblib.load(output / 'model.joblib')
        train = prepared[prepared.partition == 'train']
        np.testing.assert_allclose(bundle['scaler'].mean_, train[smoke.FEATURES].mean())
        assert bundle['features'] == smoke.FEATURES
        predictions = pd.read_csv(output / 'test-predictions.csv')
        test = prepared[prepared.partition == 'test']
        np.testing.assert_allclose(predictions.probability,
                                  bundle['model'].predict_proba(bundle['scaler'].transform(test[smoke.FEATURES]))[:, 1])
