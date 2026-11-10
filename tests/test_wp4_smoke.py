"""Scientific split invariants for the WP4 smoke comparison."""
import importlib.util
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
