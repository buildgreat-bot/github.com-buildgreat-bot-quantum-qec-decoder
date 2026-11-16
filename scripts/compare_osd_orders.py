"""Compare decoding error rates across OSD orders.

Decodes 3,000 shots each from circuits/X_circuit.stim and circuits/Z_circuit.stim
comparing:
  1. qldpc-rs (BP + OSD-0)
  2. C++ ldpc package (BP + OSD-0)
  3. C++ ldpc package (BP + OSD-CS order 7)

Skips cleanly if the optional `ldpc` package is not installed.
"""

import os
import sys
import time
import numpy as np
import stim

try:
    from ldpc import BpOsdDecoder
    HAS_LDPC = True
except ImportError:
    HAS_LDPC = False

from qldpc_rs import QldpcDecoder


def extract_dem(circuit: stim.Circuit):
    """Extract binary H, L matrices and priors from a Stim circuit."""
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


def evaluate_circuit(name: str, circuit_path: str, shots: int = 3000, seed: int = 0):
    circuit = stim.Circuit.from_file(circuit_path)
    H, L, priors = extract_dem(circuit)
    priors_f32 = priors.astype(np.float32)

    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, y_true = sampler.sample(shots, separate_observables=True)
    dets_u8 = dets.astype(np.uint8)

    print(f"\n[{name}] Evaluating {shots} shots (seed={seed}):")
    base_bit = float(np.mean(y_true != 0))
    base_block = float(np.mean(np.any(y_true != 0, axis=1)))
    print(f"  * Baseline (all-zeros): Bit LER = {base_bit:.4f} ({base_bit:.2%}), BLER = {base_block:.4f} ({base_block:.2%})")

    # 1. qldpc-rs (Normalized Min-Sum BP + OSD-0)
    dec_rs = QldpcDecoder(H, L, priors_f32, max_bp_iters=25, bp_alpha=0.625)
    t0 = time.perf_counter()
    pred_rs = dec_rs.decode_batch(dets_u8)
    dt_rs = time.perf_counter() - t0
    ler_rs = float(np.mean(pred_rs != y_true))
    bler_rs = float(np.mean(np.any(pred_rs != y_true, axis=1)))
    print(f"  * qldpc-rs (OSD-0):       Bit LER = {ler_rs:.4f} ({ler_rs:.2%}), BLER = {bler_rs:.4f} ({bler_rs:.2%}) [{dt_rs:.3f} s]")

    if not HAS_LDPC:
        print("  * [!] C++ 'ldpc' package not installed; skipping ldpc OSD-0 and OSD-CS comparisons.")
        return ler_rs, bler_rs, None, None, None, None

    # 2. ldpc OSD-0
    priors_list = list(priors)
    dec_b0 = BpOsdDecoder(
        H,
        error_channel=priors_list,
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        max_iter=25,
        osd_method="osd_0",
    )
    t0 = time.perf_counter()
    pred_b0 = np.array([L @ dec_b0.decode(s) % 2 for s in dets_u8], dtype=np.uint8)
    dt_b0 = time.perf_counter() - t0
    ler_b0 = float(np.mean(pred_b0 != y_true))
    bler_b0 = float(np.mean(np.any(pred_b0 != y_true, axis=1)))
    print(f"  * ldpc (OSD-0):           Bit LER = {ler_b0:.4f} ({ler_b0:.2%}), BLER = {bler_b0:.4f} ({bler_b0:.2%}) [{dt_b0:.3f} s]")

    # 3. ldpc OSD-CS order 7
    dec_b7 = BpOsdDecoder(
        H,
        error_channel=priors_list,
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        max_iter=25,
        osd_method="osd_cs",
        osd_order=7,
    )
    t0 = time.perf_counter()
    pred_b7 = np.array([L @ dec_b7.decode(s) % 2 for s in dets_u8], dtype=np.uint8)
    dt_b7 = time.perf_counter() - t0
    ler_b7 = float(np.mean(pred_b7 != y_true))
    bler_b7 = float(np.mean(np.any(pred_b7 != y_true, axis=1)))
    print(f"  * ldpc (OSD-CS order 7):  Bit LER = {ler_b7:.4f} ({ler_b7:.2%}), BLER = {bler_b7:.4f} ({bler_b7:.2%}) [{dt_b7:.3f} s]")

    return ler_rs, bler_rs, ler_b0, bler_b0, ler_b7, bler_b7


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    circ_dir = os.path.join(base_dir, "circuits")

    x_path = os.path.join(circ_dir, "X_circuit.stim")
    z_path = os.path.join(circ_dir, "Z_circuit.stim")

    if not os.path.exists(x_path) or not os.path.exists(z_path):
        print("Circuits directory not found.")
        sys.exit(1)

    print("=================================================================")
    print("      OSD Order Comparison Benchmark (qldpc-rs vs C++ ldpc)    ")
    print("=================================================================")

    evaluate_circuit("X-Circuit", x_path, shots=3000, seed=0)
    evaluate_circuit("Z-Circuit", z_path, shots=3000, seed=0)

    print("\n=================================================================")
    print("Benchmark complete.")
    print("=================================================================")


if __name__ == "__main__":
    main()
