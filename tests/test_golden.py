import os
import hashlib
import numpy as np
import pytest
import stim

from qldpc_rs import QldpcDecoder


def extract_dem_info(circuit: stim.Circuit):
    """Extract parity check matrix H, observable matrix L, and error priors p from Stim circuit."""
    dem = circuit.detector_error_model()
    columns = {}
    priors = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key = (frozenset(detectors), frozenset(observables))
        p = instruction.args_copy()[0]
        if key in columns:
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.asarray(priors, dtype=np.float32)


@pytest.mark.timeout(60)
def test_platform_independent_golden_decoding():
    """Platform-independent golden test verifying bit-identical output.

    Loads 2,000 fixed syndrome vectors from tests/data/x_syndromes.npy and
    asserts that decode_batch produces observables matching the golden SHA256
    hash in tests/data/x_expected.sha256.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tests_dir)

    circuit_path = os.path.join(repo_root, "circuits", "X_circuit.stim")
    syndromes_path = os.path.join(tests_dir, "data", "x_syndromes.npy")
    expected_hash_path = os.path.join(tests_dir, "data", "x_expected.sha256")

    assert os.path.exists(circuit_path), f"Missing circuit: {circuit_path}"
    assert os.path.exists(syndromes_path), f"Missing golden syndromes: {syndromes_path}"
    assert os.path.exists(expected_hash_path), f"Missing expected hash file: {expected_hash_path}"

    with open(expected_hash_path, "r", encoding="utf-8") as f:
        expected_hash = f.read().strip()

    circuit = stim.Circuit.from_file(circuit_path)
    H, L, priors = extract_dem_info(circuit)

    dec = QldpcDecoder(H, L, priors, max_bp_iters=25, bp_alpha=0.625)
    syndromes = np.load(syndromes_path)
    assert syndromes.dtype == np.uint8
    assert syndromes.shape == (2000, 70)

    preds = dec.decode_batch(syndromes)
    actual_hash = hashlib.sha256(preds.tobytes()).hexdigest()

    assert actual_hash == expected_hash, (
        f"Golden hash mismatch!\nExpected: {expected_hash}\nActual:   {actual_hash}"
    )
