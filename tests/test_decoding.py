"""Decoding correctness test suite for qldpc-rs.

Validates:
  1. 5-bit repetition code: every single-fault syndrome decodes to the exact logical observable.
  2. [[18,4,4]] circuits: all 2,470 single-fault syndromes on X_circuit.stim (<= 1 failure)
     and 2,465 single-fault syndromes on Z_circuit.stim (<= 4 failures).
  3. Word-boundary bit-packing logic: random codes with M in {1, 63, 64, 65, 128, 129, 200}
     matching deterministic golden references.
  4. Accuracy improvement: beats the all-zeros baseline by >= 70% on a fixed 2,000-shot sample.
"""

import os
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
def test_5bit_repetition_code_all_single_faults():
    """5-bit repetition code: every single-fault syndrome must decode to the correct logical observable."""
    # N = 5 bits, M = 4 checks, k = 1 logical observable
    H = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.uint8)
    # Observable is logical flip on bit 0
    L = np.array([[1, 0, 0, 0, 0]], dtype=np.uint8)
    priors = np.full(5, 0.05, dtype=np.float32)

    dec = QldpcDecoder(H, L, priors, max_bp_iters=20, bp_alpha=0.625)

    for j in range(5):
        e = np.zeros(5, dtype=np.uint8)
        e[j] = 1
        syn = (H @ e % 2).reshape(1, -1)
        y_true = (L @ e % 2).reshape(1, -1)
        y_pred = dec.decode_batch(syn)
        assert y_pred[0, 0] == y_true[0, 0], f"Single fault at bit {j} decoded to {y_pred[0, 0]}, expected {y_true[0, 0]}"


@pytest.mark.timeout(60)
def test_single_fault_syndromes_circuits():
    """All single-fault syndromes of X_circuit (<= 1 failure) and Z_circuit (<= 4 failures)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    circ_dir = os.path.join(base_dir, "circuits")

    cx = stim.Circuit.from_file(os.path.join(circ_dir, "X_circuit.stim"))
    cz = stim.Circuit.from_file(os.path.join(circ_dir, "Z_circuit.stim"))

    Hx, Lx, px = extract_dem_info(cx)
    Hz, Lz, pz = extract_dem_info(cz)

    dec_x = QldpcDecoder(Hx, Lx, px, max_bp_iters=25, bp_alpha=0.625)
    dec_z = QldpcDecoder(Hz, Lz, pz, max_bp_iters=25, bp_alpha=0.625)

    # 1. X circuit: 2,470 single-fault syndromes (columns of Hx)
    syn_x = np.ascontiguousarray(Hx.T)
    y_true_x = np.ascontiguousarray(Lx.T)
    y_pred_x = dec_x.decode_batch(syn_x)
    fail_x = int(np.sum(np.any(y_pred_x != y_true_x, axis=1)))
    assert fail_x <= 1, f"X circuit single-fault failures: {fail_x} (expected <= 1)"

    # 2. Z circuit: 2,465 single-fault syndromes (columns of Hz)
    syn_z = np.ascontiguousarray(Hz.T)
    y_true_z = np.ascontiguousarray(Lz.T)
    y_pred_z = dec_z.decode_batch(syn_z)
    fail_z = int(np.sum(np.any(y_pred_z != y_true_z, axis=1)))
    assert fail_z <= 4, f"Z circuit single-fault failures: {fail_z} (expected <= 4)"


@pytest.mark.timeout(60)
def test_u64_word_boundary_random_codes():
    """Test random codes with M in {1, 63, 64, 65, 128, 129, 200} to guard u64 word-boundary logic."""
    golden_outputs = {
        1: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        63: [[1, 0], [1, 0], [0, 0], [0, 0], [0, 0]],
        64: [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0]],
        65: [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0]],
        128: [[1, 0], [1, 0], [0, 0], [0, 1], [0, 1]],
        129: [[1, 1], [0, 1], [0, 1], [1, 0], [1, 1]],
        200: [[1, 0], [0, 0], [1, 0], [1, 1], [0, 0]],
    }

    for M in [1, 63, 64, 65, 128, 129, 200]:
        rng = np.random.default_rng(M + 42)
        N = M + 20
        k = 2
        H = (rng.random((M, N)) < 0.2).astype(np.uint8)
        for i in range(M):
            if H[i].sum() < min(2, N):
                cols = rng.choice(N, size=min(2, N), replace=False)
                H[i, cols] = 1
        for j in range(N):
            if H[:, j].sum() < min(2, M):
                rows = rng.choice(M, size=min(2, M), replace=False)
                H[rows, j] = 1

        L = (rng.random((k, N)) < 0.1).astype(np.uint8)
        priors = np.full(N, 0.05, dtype=np.float32)

        dec = QldpcDecoder(H, L, priors, max_bp_iters=5, bp_alpha=0.625)

        e_rand = (rng.random((5, N)) < 0.05).astype(np.uint8)
        syn = (e_rand @ H.T % 2).astype(np.uint8)

        y_pred, status = dec.decode_batch(syn, return_status=True)
        assert np.all(status), f"Failed decode status for M={M}"
        assert y_pred.tolist() == golden_outputs[M], f"Mismatch at word-boundary M={M}"


@pytest.mark.timeout(60)
def test_beats_all_zeros_baseline_by_70_percent():
    """Assert decoder beats the all-zeros baseline by >= 70% on a fixed 2,000-shot sample."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    circ_dir = os.path.join(base_dir, "circuits")

    cx = stim.Circuit.from_file(os.path.join(circ_dir, "X_circuit.stim"))
    Hx, Lx, px = extract_dem_info(cx)
    dec_x = QldpcDecoder(Hx, Lx, px, max_bp_iters=25, bp_alpha=0.625)

    sampler = cx.compile_detector_sampler(seed=42)
    shots = 2000
    dets, y_true = sampler.sample(shots, separate_observables=True)

    # All-zeros baseline error rate
    ler_base = float(np.mean(y_true != 0))

    # Decoded error rate
    y_pred = dec_x.decode_batch(dets.astype(np.uint8))
    ler_dec = float(np.mean(y_true != y_pred))

    reduction = (ler_base - ler_dec) / ler_base
    assert reduction >= 0.70, f"Error rate reduction {reduction:.2%} is below 70%"
