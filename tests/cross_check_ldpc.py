"""Optional cross-check test comparing qldpc-rs against the reference C++ ldpc package.

Samples 3,000 shots each from circuits/X_circuit.stim and circuits/Z_circuit.stim,
evaluating per-shot observable prediction agreement between:
  - qldpc_rs.QldpcDecoder (Normalized Min-Sum BP alpha=0.625 + OSD-0)
  - ldpc.BpOsdDecoder(bp_method="minimum_sum", ms_scaling_factor=0.625, max_iter=25, osd_method="osd_0")

Skips cleanly if the optional `ldpc` package is not installed.
"""

import os
import sys
import numpy as np
import stim

try:
    import pytest
except ImportError:
    pytest = None

try:
    from ldpc import BpOsdDecoder
    HAS_LDPC = True
except ImportError:
    HAS_LDPC = False

from qldpc_rs import QldpcDecoder


def extract_dem(circuit: stim.Circuit):
    dem = circuit.detector_error_model()
    cols, priors = {}, []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        dets, obs = set(), set()
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        key = (frozenset(dets), frozenset(obs))
        p = ins.args_copy()[0]
        if key in cols:
            q = priors[cols[key]]
            priors[cols[key]] = q * (1 - p) + p * (1 - q)
        else:
            cols[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (dets, obs), j in cols.items():
        H[list(dets), j] = 1
        L[list(obs), j] = 1
    return H, L, np.asarray(priors, dtype=np.float64)


def run_cross_check(shots: int = 3000, seed: int = 0):
    if not HAS_LDPC:
        print("[!] Optional 'ldpc' package not installed. Skipping cross-check.")
        return True

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    circuits = [
        ("X-basis", os.path.join(repo_root, "circuits", "X_circuit.stim")),
        ("Z-basis", os.path.join(repo_root, "circuits", "Z_circuit.stim")),
    ]

    total_shots = 0
    total_matches = 0

    print("=================================================================")
    print("      qldpc-rs vs C++ ldpc Package Cross-Check Verification     ")
    print("=================================================================")

    for name, path in circuits:
        if not os.path.exists(path):
            print(f"[-] Circuit file missing: {path}")
            return False

        circuit = stim.Circuit.from_file(path)
        H, L, priors = extract_dem(circuit)

        q_dec = QldpcDecoder(H, L, priors.astype(np.float32), max_bp_iters=25, bp_alpha=0.625)
        l_dec = BpOsdDecoder(
            H,
            channel_probs=priors,
            max_iter=25,
            bp_method="minimum_sum",
            ms_scaling_factor=0.625,
            osd_method="osd_0",
        )

        sampler = circuit.compile_detector_sampler(seed=seed)
        syndromes, true_obs = sampler.sample(shots=shots, separate_observables=True)

        q_preds = q_dec.decode_batch(syndromes.astype(np.uint8))

        matches = 0
        mismatches = 0
        ldpc_syn_satisfied = 0
        for i in range(shots):
            s = syndromes[i].astype(np.uint8)
            corr_ldpc = l_dec.decode(s)
            l_pred = (L @ corr_ldpc) % 2
            if np.array_equal(q_preds[i], l_pred):
                matches += 1
            else:
                mismatches += 1
                if np.array_equal((H @ corr_ldpc) % 2, s):
                    ldpc_syn_satisfied += 1

        pct = (matches / shots) * 100.0
        print(f"  * {name}: {matches}/{shots} exact matches ({pct:.2f}%)")
        if mismatches > 0:
            print(f"    - Note: {mismatches} mismatch(es) (ldpc correction satisfies syndrome: {ldpc_syn_satisfied})")
            print("      Mismatches are expected on non-converged-BP shots where OSD-0 ordering is sensitive to floating-point precision.")

        total_shots += shots
        total_matches += matches

    overall_pct = (total_matches / total_shots) * 100.0
    print("-----------------------------------------------------------------")
    print(f"  Total Agreement: {total_matches}/{total_shots} shots ({overall_pct:.2f}%)")
    print("=================================================================")

    # Assert >99.9% agreement (accounting for rare non-converged BP shots where OSD-0 ordering varies with float precision)
    assert overall_pct >= 99.9, f"Agreement {overall_pct:.2f}% below expected 99.9%"
    return True


def test_cross_check_ldpc():
    if not HAS_LDPC:
        if pytest:
            pytest.skip("Optional 'ldpc' package not installed")
        else:
            return
    assert run_cross_check(shots=3000, seed=0)


if __name__ == "__main__":
    success = run_cross_check(shots=3000, seed=0)
    sys.exit(0 if success else 1)
