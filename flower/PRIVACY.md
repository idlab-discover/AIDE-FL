# Privacy experiments

These mechanisms have different trust boundaries. They are not enabled by default.

| Mechanism | Implementation | Current execution scope |
| --- | --- | --- |
| Central differential privacy | Native Flower fixed server clipping and RDP accountant | Integrated TensorFlow/FedAvg runtime and local smoke experiment |
| Secure aggregation | Native Flower SecAggPlusWorkflow and secaggplus_mod | Local four-stage protocol experiment with three or more clients |
| Encrypted FedAvg | TenSEAL CKKS encrypted weighted sums | Experimental local FedAvg extension with a separate trusted decryptor role |

## Central differential privacy

Add this object to a TensorFlow FedAvg scenario's `aggregation` section:

```json
"privacy": {
  "mode": "central_dp",
  "noise_multiplier": 1.0,
  "clipping_norm": 0.2,
  "num_sampled_clients": 3,
  "population_size": 3,
  "target_delta": 0.00001,
  "max_epsilon": 15.0
}
```

Set `min_available_clients` to three and `fraction_fit` to 1 for this example.
The configured cohort must exactly match `num_sampled_clients`; missing replies
abort the run. Only floating-point TensorFlow model arrays are accepted. XGBoost
trees and scikit-learn class metadata are excluded. Aggregation assigns each client
weight one, regardless of its dataset size.

The trusted server receives individual updates, clips their joint L2 norm, averages
them and adds Gaussian noise. The native RDP accountant uses add/remove-one-client
adjacency and no sampling amplification. It refuses a round that would exceed
`max_epsilon`. That setting is optional. The recorded epsilon applies to released
models within this run, not to the entire system or repeated runs over the same
clients. Budgets reset for a new run and must be composed across repeated runs.
The accountant is not a persistent cross-run privacy ledger.

Client metrics, histories and verbose training output are suppressed. Federated
evaluation and central holdout evaluation are disabled. Saved checkpoints contain
the noisy released model, and `<scenario>_privacy.json` records the cumulative
epsilon, delta and number of releases. Private preprocessing statistics, membership,
timing and metadata are outside this accounting. Model selection or preprocessing
using private data requires a separate analysis. This central mechanism does not
hide client updates from the server and is not combined with SecAgg in this release.

Run the public-data utility diagnostic with:

```sh
python experiments/wp4/privacy_comparison.py --output /tmp/wp4-dp --rounds 5
```

It trains a small TensorFlow classifier on the smoke sensor dataset with three
disjoint training partitions and compares uniform FedAvg against two noise levels.
All methods use fixed feature scales without pooling private sample statistics.
The common chronological validation/test data are treated as public benchmark
data for this diagnostic. Their published metrics are not private-client releases.
Model initialization and shuffle seeds are retained, while privacy noise uses fresh
OS entropy rather than the published experiment seed. Five rounds and one training
seed are a diagnostic, not a stable estimate of utility or a privacy audit. A large
reported epsilon is not evidence of strong privacy.

## Native secure aggregation

`flower.experimental_secagg.aggregate_secagg` runs the actual upstream SecAgg+
workflow and client modifier with isolated client contexts and in-process message
delivery. It checks that protocol replies contain no nonempty model ArrayRecords.
The result approximates sample-weighted FedAvg because the protocol quantizes
floating-point values. It rejects weights outside the configured clipping range.

The experiment uses at least three clients, all-to-all shares and reconstruction
threshold two. It has not validated dropouts, collusion, malicious clients or a
network deployment. The current application uses the modern Message strategy;
SecAgg+ still requires Flower's legacy workflow context, so this experiment uses an
explicit local bridge. Do not label it a deployed secure runtime or a generic MPC
engine. Flower's native protocol is the MPC-style secure-sum capability examined here.

## Experimental encrypted FedAvg

Install `flower/fhe-requirements.txt` into the experiment environment to use the
optional TenSEAL 0.3.16 extension. `experimental_fhe.py` separates key creation,
public-context encryption, ciphertext aggregation and secret-key decryption. The
aggregator rejects contexts containing a secret key. Clients preweight their float
arrays by sample count; the server adds ciphertexts; the decryptor divides the
decrypted sum by the total sample count. Class labels and integer metadata must
remain outside encrypted model weights. Ciphertexts and public sample counts are
not authenticated.

This is leveled CKKS homomorphic encryption with approximate arithmetic, not
bootstrapped arbitrary-depth FHE. It does not provide threshold decryption. The
trusted decryptor could decrypt an individual update if it obtained that ciphertext;
separating Python objects in a local experiment does not establish process or host
isolation. There is no malicious-server protection, attestation, key provisioning,
or dropout protocol. Treat this as a research extension, not a production security
boundary. Keys are generated in memory and are not saved in experiment artifacts.

The smoke runner's `--fhe` option compares real trained weights with plaintext
FedAvg, checks numerical error and stores ciphertext sizes and utility metrics.

## Verification and sources

`python -m pytest tests/test_privacy.py` exercises clipping/noise/accounting,
budget refusal, missing-client failure, private metric suppression, noisy-only
checkpoint persistence, native SecAgg+ masked exchanges and encrypted/plaintext
weighted-average agreement. The TenSEAL test skips if the optional package is absent.

- [Flower 1.35 differential privacy](https://flower.ai/docs/framework/1.35/en/how-to-use-differential-privacy.html)
- [Flower 1.35 secure aggregation protocols](https://flower.ai/docs/framework/1.35/en/explanation-ref-secure-aggregation-protocols.html)
- [TenSEAL source and CKKS examples](https://github.com/OpenMined/TenSEAL)
