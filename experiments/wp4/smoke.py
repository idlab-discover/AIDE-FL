#!/usr/bin/env python3
"""Reproducible local smoke-sensor experiments with the AIDE adapter and Flower FedAvg."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from flwr.app import ArrayRecord, Message, RecordDict
from flwr.serverapp.strategy import FedAvg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'flower'))
from flower.frameworks.base import ClientDatasets
from flower.frameworks.scikit_learn_adapter import ScikitLearnAdapter

SOURCE = 'https://www.kaggle.com/api/v1/datasets/download/deepcontractor/smoke-detection-dataset'
DATASET_PAGE = 'https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset'
FEATURES = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2',
            'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5']
LABEL = 'Fire Alarm'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_data(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'smoke-detection-kaggle.zip'
    csv = cache / 'smoke_detection_iot.csv'
    if not archive.exists():
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as z:
        csv.write_bytes(z.read('smoke_detection_iot.csv'))
    frame = pd.read_csv(csv)
    required = FEATURES + [LABEL, 'UTC']
    if not set(required).issubset(frame.columns):
        raise ValueError('Smoke source does not have the expected sensor schema')
    if not np.isfinite(frame[required].to_numpy()).all() or set(frame[LABEL]) != {0, 1}:
        raise ValueError('Smoke source contains invalid features or binary labels')
    frame['row_id'] = np.arange(len(frame))
    original_count = len(frame)
    frame = frame.sort_values(['UTC', 'row_id']).drop_duplicates(FEATURES, keep='first')
    return frame, {'source_url': SOURCE, 'dataset_page': DATASET_PAGE,
                   'archive_sha256': sha256(archive), 'csv_sha256': sha256(csv),
                   'source_rows': original_count, 'unique_rows': len(frame),
                   'dropped_duplicate_sensor_rows': original_count - len(frame),
                   'retrieved_at': datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat()}


def split_data(frame, clients, seed, partition, embargo_seconds=60):
    """Use global chronological splits, preserving timestamps and embargoing boundaries."""
    ordered = frame.sort_values(['UTC', 'row_id']).copy()
    first = int(ordered.iloc[int(len(ordered) * .5)]['UTC'])
    second = int(ordered.iloc[int(len(ordered) * .7)]['UTC'])
    ordered['partition'] = 'embargo'
    ordered.loc[ordered.UTC < first - embargo_seconds, 'partition'] = 'train'
    ordered.loc[(ordered.UTC >= first) & (ordered.UTC < second - embargo_seconds), 'partition'] = 'validation'
    ordered.loc[ordered.UTC >= second, 'partition'] = 'test'
    ordered['client'] = -1
    training = ordered[ordered.partition == 'train']
    if partition == 'iid':
        splitter = StratifiedKFold(clients, shuffle=True, random_state=seed)
        assignments = [test for _, test in splitter.split(training[FEATURES], training[LABEL])]
    else:
        assignments = np.array_split(np.arange(len(training)), clients)
    for client, indices in enumerate(assignments):
        ordered.loc[training.iloc[indices].index, 'client'] = client
    for split in ('train', 'validation', 'test'):
        if set(ordered.loc[ordered.partition == split, LABEL]) != {0, 1}:
            raise ValueError(f'{split} must contain both classes; inspect source temporal coverage')
    for client in range(clients):
        if set(ordered.loc[ordered.client == client, LABEL]) != {0, 1}:
            raise ValueError(f'client {client} lacks both classes; reduce clients or use iid partition')
    return ordered


def metrics(model, x, y, prefix):
    probabilities = model.predict_proba(x)[:, 1]
    predicted = model.predict(x)
    return {f'{prefix}_{key}': value for key, value in {
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'logloss': float(log_loss(y, probabilities, labels=[0, 1])),
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'num_examples': len(y),
    }.items()}


def run_method(output, prepared, config):
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', config)
    adapter = ScikitLearnAdapter()
    scenario = {'label_column': LABEL, 'training': {'local_epochs': config['local_epochs'], 'batch_size': 256},
                'scikit_learn': {'model': {'kind': 'logistic_regression', 'n_features': len(FEATURES),
                                         'classes': [0, 1], 'solver': 'saga'}}}
    save_json(output / 'model-scenario.json', scenario)
    model = adapter.build_model_from_scenario(scenario)
    arrays = adapter.initial_arrays(model)
    train_config = adapter.make_config_record(scenario)
    train = prepared[prepared.partition == 'train']
    validation = prepared[prepared.partition == 'validation']
    test = prepared[prepared.partition == 'test']
    # The common transform uses training-only moments; all methods use this same transform.
    scaler = StandardScaler().fit(train[FEATURES])
    x_val, y_val = scaler.transform(validation[FEATURES]), validation[LABEL].to_numpy()
    method = config['method']
    if method == 'centralised':
        subsets = [train]
    elif method.startswith('local-'):
        subsets = [train[train.client == int(method.split('-')[1])]]
    else:
        subsets = [train[train.client == c] for c in range(config['clients'])]
    datasets = [ClientDatasets(scaler.transform(s[FEATURES]), s[LABEL].to_numpy(),
                              x_val, y_val, x_val, y_val) for s in subsets]
    strategy = FedAvg(min_available_nodes=len(subsets))
    if method == 'secagg-fedavg':
        from flower.experimental_secagg import aggregate_secagg
    if method == 'fhe-fedavg':
        from flower.experimental_fhe import (create_private_context, public_context_bytes,
                                             encrypt_weights, aggregate_encrypted, decrypt_weights)
        private_context = create_private_context()
        public_context = public_context_bytes(private_context)
    convergence_warnings = 0
    with (output / 'metrics.jsonl').open('w') as log:
        for rnd in range(1, config['rounds'] + 1):
            replies = []
            encrypted_updates = []
            secure_updates = []
            for client, data in enumerate(datasets):
                local = adapter.build_model_from_scenario(scenario)
                local.random_state = config['seed'] + rnd * 1000 + client
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    result = adapter.train(local, arrays, data, train_config, False)
                convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                if method == 'secagg-fedavg':
                    secure_updates.append((result.arrays.to_numpy_ndarrays()[:2], len(data.x_train)))
                if method == 'fhe-fedavg':
                    encrypted_updates.append(encrypt_weights(public_context, result.arrays.to_numpy_ndarrays()[:2], len(data.x_train)))
                request = Message(RecordDict(), dst_node_id=client + 1, message_type='train')
                replies.append(Message(RecordDict({'arrays': result.arrays, 'metrics': result.metrics}), reply_to=request))
            arrays, training_metrics = strategy.aggregate_train(rnd, replies)
            if arrays is None or training_metrics is None:
                raise RuntimeError('Flower aggregation returned no model')
            encryption_metrics = {}
            if method == 'secagg-fedavg':
                plaintext_reference = arrays.to_numpy_ndarrays()
                secure_arrays, protocol = aggregate_secagg(secure_updates)
                max_error = max(float(np.max(np.abs(a - b))) for a, b in zip(secure_arrays, plaintext_reference[:2]))
                if max_error > 1e-4:
                    raise RuntimeError(f'SecAgg quantization error exceeds tolerance: {max_error}')
                arrays = ArrayRecord([*secure_arrays, plaintext_reference[2]])
                encryption_metrics = {'secagg_max_absolute_error': max_error,
                                      'secagg_stages': protocol['stages'],
                                      'secagg_message_replies': protocol['message_replies'],
                                      'secagg_plaintext_array_values': protocol['plaintext_array_values_in_replies']}
            if method == 'fhe-fedavg':
                plaintext_reference = arrays.to_numpy_ndarrays()
                encrypted = aggregate_encrypted(public_context, encrypted_updates)
                decrypted = decrypt_weights(private_context, encrypted)
                max_error = max(float(np.max(np.abs(a - b))) for a, b in zip(decrypted, plaintext_reference[:2]))
                if max_error > 1e-5:
                    raise RuntimeError(f'CKKS aggregation error exceeds tolerance: {max_error}')
                arrays = ArrayRecord([*decrypted, plaintext_reference[2]])
                encryption_metrics = {'ckks_max_absolute_error': max_error,
                                      'encrypted_upload_bytes': sum(len(c) for u in encrypted_updates for c in u.ciphertexts)}
            adapter._set_model_params(model, arrays.to_numpy_ndarrays())
            row = {'round': rnd, **dict(training_metrics), **encryption_metrics, **metrics(model, x_val, y_val, 'validation')}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
    x_test, y_test = scaler.transform(test[FEATURES]), test[LABEL].to_numpy()
    majority = int(train[LABEL].mode().iloc[0])
    dummy = np.full(len(y_test), majority)
    summary = {**metrics(model, x_test, y_test, 'test'),
               'majority_baseline_label': majority,
               'majority_baseline_test_accuracy': float(accuracy_score(y_test, dummy)),
               'majority_baseline_test_balanced_accuracy': float(balanced_accuracy_score(y_test, dummy)),
               'majority_baseline_test_f1': float(f1_score(y_test, dummy, zero_division=0)),
               'test_confusion_matrix': confusion_matrix(y_test, model.predict(x_test), labels=[0, 1]).tolist(),
               'test_confusion_matrix_labels': [0, 1], 'convergence_warnings': convergence_warnings,
               'selected_round': config['rounds'], 'selection_rule': 'fixed final round; no test-driven selection'}
    joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES, 'label': LABEL}, output / 'model.joblib')
    restored = joblib.load(output / 'model.joblib')
    np.testing.assert_array_equal(restored['model'].predict(restored['scaler'].transform(test[FEATURES])), model.predict(x_test))
    pd.DataFrame({'row_id': test.row_id, 'UTC': test.UTC, 'label': y_test,
                  'probability': model.predict_proba(x_test)[:, 1]}).to_csv(output / 'test-predictions.csv', index=False)
    save_json(output / 'summary.json', summary)
    save_json(output / 'checksums.json', {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps({'run': output.name, **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/tmp/aide-wp4-smoke-data'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 73])
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--secagg', action='store_true', help='Add native Flower SecAgg+ local protocol comparison (3+ clients)')
    parser.add_argument('--fhe', action='store_true', help='Add experimental trusted-decryptor CKKS FedAvg comparison')
    parser.add_argument('--partition', choices=['iid', 'chronological'], default='iid')
    args = parser.parse_args()
    if args.clients < 2 or args.rounds < 1 or args.local_epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require clients >= 2, positive rounds/epochs, and unique seeds')
    if args.secagg and args.clients < 3:
        parser.error('--secagg requires at least 3 clients')
    args.output.mkdir(parents=True, exist_ok=False)
    frame, provenance = source_data(args.cache)
    versions = {name: importlib.metadata.version(name) for name in ['flwr', 'numpy', 'pandas', 'scikit-learn', 'joblib']}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    (args.output / 'working-tree.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    adapter_path = ROOT / 'flower/flower/frameworks/scikit_learn_adapter.py'
    source_hashes = {'runner': sha256(__file__), 'adapter': sha256(adapter_path)}
    if args.fhe:
        versions['tenseal'] = importlib.metadata.version('tenseal')
        source_hashes['fhe'] = sha256(ROOT / 'flower/flower/experimental_fhe.py')
    if args.secagg:
        source_hashes['secagg'] = sha256(ROOT / 'flower/flower/experimental_secagg.py')
    source_dir = args.output / 'source'
    source_dir.mkdir()
    (source_dir / 'smoke.py').write_bytes(Path(__file__).read_bytes())
    (source_dir / 'scikit_learn_adapter.py').write_bytes(adapter_path.read_bytes())
    if args.fhe:
        (source_dir / 'experimental_fhe.py').write_bytes((ROOT / 'flower/flower/experimental_fhe.py').read_bytes())
    if args.secagg:
        (source_dir / 'experimental_secagg.py').write_bytes((ROOT / 'flower/flower/experimental_secagg.py').read_bytes())
    save_json(args.output / 'provenance.json', {**provenance, 'features': FEATURES, 'label': LABEL,
               'excluded_features': ['UTC', 'CNT', 'Unnamed: 0'], 'versions': versions,
               'source_commit': commit, 'git_dirty': dirty, 'source_sha256': source_hashes, 'runner_sha256': sha256(__file__),
               'split': 'UTC ordered 50/20/30 percent; 60-second embargo before validation and test',
               'scope': 'In-process AIDE sklearn adapter and Flower Message/FedAvg aggregation; no network or cluster',
               'preprocessing': 'Training-only shared StandardScaler; training moments pooled; not privacy-preserving',
               'limitations': 'Simulated clients from one sensor dataset; temporal sensor classification, no sequence model or independent sites'})
    run_dirs = []
    for seed in args.seeds:
        prepared = split_data(frame, args.clients, seed, args.partition)
        partition_path = args.output / f'partitions-seed-{seed}.csv'
        prepared[['row_id', 'UTC', LABEL, 'partition', 'client']].to_csv(partition_path, index=False)
        counts = prepared.groupby(['partition', 'client', LABEL]).size().reset_index(name='rows')
        counts.to_csv(args.output / f'partition-counts-seed-{seed}.csv', index=False)
        methods = [*[f'local-{c}' for c in range(args.clients)], 'centralised', 'fedavg']
        if args.fhe:
            methods.append('fhe-fedavg')
        if args.secagg:
            methods.append('secagg-fedavg')
        for method in methods:
            name = f'{args.partition}-seed-{seed}-{method}'
            config = {'seed': seed, 'method': method, 'clients': args.clients, 'rounds': args.rounds,
                      'local_epochs': args.local_epochs, 'partition': args.partition,
                      'dataset_sha256': provenance['csv_sha256'], 'partition_sha256': sha256(partition_path),
                      'source_commit': commit, 'git_dirty': dirty, 'source_sha256': source_hashes, 'runner_sha256': sha256(__file__), 'versions': versions,
                      'execution': 'in-process Flower aggregation', 'split': 'temporal-50-20-30-embargo-60s',
                      'feature_count': len(FEATURES), 'threshold': .5}
            run_method(args.output / name, prepared, config)
            run_dirs.append(name)
    save_json(args.output / 'experiment.json', {'runs': run_dirs, 'created_at': datetime.now(timezone.utc).isoformat(),
               'provenance': 'provenance.json'})


if __name__ == '__main__':
    main()
