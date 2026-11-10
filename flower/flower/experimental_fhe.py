"""Experimental CKKS encrypted FedAvg, with a separate key-holding decryptor.

This uses leveled homomorphic encryption, not arbitrary-depth bootstrapped FHE.
The aggregator receives only a public context. The decryptor is trusted and could
decrypt individual ciphertexts if given them. No threshold or malicious-party
security, ciphertext authentication, or differential privacy is provided.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class EncryptedUpdate:
    context_id: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    ciphertexts: tuple[bytes, ...]
    num_examples: int
    contributors: int = 1


def create_private_context():
    import tenseal as ts

    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1)
    context.global_scale = 2**40
    return context


def public_context_bytes(private_context):
    return private_context.serialize(save_secret_key=False, save_galois_keys=False,
                                     save_relin_keys=False)


def _public_context(serialized):
    import tenseal as ts

    context = ts.context_from(serialized, n_threads=1)
    if context.has_secret_key():
        raise ValueError("aggregator and encryptor must receive a public-only context")
    return context


def encrypt_weights(public_context, arrays, num_examples):
    import tenseal as ts

    if isinstance(num_examples, bool) or not isinstance(num_examples, int) or num_examples <= 0:
        raise ValueError("num_examples must be a positive integer")
    context = _public_context(public_context)
    arrays = [np.asarray(array) for array in arrays]
    if not arrays or any(array.size == 0 or array.dtype.kind != "f" or not np.isfinite(array).all() for array in arrays):
        raise ValueError("encrypted FedAvg accepts only nonempty finite floating-point weights")
    # Preweight before encryption: the server performs only ciphertext additions.
    ciphertexts = tuple(ts.ckks_vector(context, (array.ravel().astype(np.float64) * num_examples).tolist()).serialize()
                        for array in arrays)
    return EncryptedUpdate(sha256(public_context).hexdigest(),
                           tuple(array.shape for array in arrays),
                           tuple(array.dtype.str for array in arrays), ciphertexts, num_examples)


def aggregate_encrypted(public_context, updates):
    import tenseal as ts

    context = _public_context(public_context)
    updates = list(updates)
    if len(updates) < 2:
        raise ValueError("encrypted aggregation requires at least two contributions")
    reference = updates[0]
    context_id = sha256(public_context).hexdigest()
    for update in updates:
        if update.context_id != context_id:
            raise ValueError("encrypted updates use a different public context")
        if (update.shapes, update.dtypes) != (reference.shapes, reference.dtypes):
            raise ValueError("encrypted model layouts differ")
        if len(update.ciphertexts) != len(reference.shapes) or update.num_examples <= 0 or update.contributors != 1:
            raise ValueError("invalid individual encrypted contribution")
    sums = []
    for i in range(len(reference.shapes)):
        vectors = [ts.ckks_vector_from(context, update.ciphertexts[i]) for update in updates]
        summed = vectors[0]
        for vector in vectors[1:]:
            summed += vector
        sums.append(summed.serialize())
    return EncryptedUpdate(context_id, reference.shapes, reference.dtypes, tuple(sums),
                           sum(update.num_examples for update in updates), len(updates))


def decrypt_weights(private_context, aggregate):
    import tenseal as ts

    if not private_context.has_secret_key():
        raise ValueError("the separate decryptor requires the secret key")
    if aggregate.context_id != sha256(public_context_bytes(private_context)).hexdigest():
        raise ValueError("aggregate belongs to a different private context")
    if aggregate.contributors < 2 or aggregate.num_examples <= 0:
        raise ValueError("decrypt_weights expects an aggregated update")
    arrays = []
    for ciphertext, shape, dtype in zip(aggregate.ciphertexts, aggregate.shapes, aggregate.dtypes, strict=True):
        vector = ts.ckks_vector_from(private_context, ciphertext)
        array = np.asarray(vector.decrypt()) / aggregate.num_examples
        if not np.isfinite(array).all():
            raise ValueError("non-finite decrypted aggregate")
        converted = array.reshape(shape).astype(dtype)
        if not np.isfinite(converted).all():
            raise ValueError("decrypted aggregate overflows the model dtype")
        arrays.append(converted)
    return arrays
