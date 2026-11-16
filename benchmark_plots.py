"""Generate publication-grade benchmark plots for qldpc-rs.

Self-contained: requires only numpy, stim, matplotlib, and qldpc_rs.
All data points are generated dynamically from the underlying quantum circuits.
"""

import os
import sys
import time
import numpy as np
import stim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qldpc_rs import QldpcDecoder

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Set styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.edgecolor"] = "#30363d"
plt.rcParams["axes.linewidth"] = 1.2

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
circ_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuits")
os.makedirs(fig_dir, exist_ok=True)

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

def apply_si1000_noise(circuit: stim.Circuit, p: float) -> stim.Circuit:
    """Applies SI1000 circuit-level noise model (Gidney et al. 2021) to a noiseless circuit."""
    num_qubits = circuit.num_qubits
    noisy_circuit = stim.Circuit()
    touched = set()
    measured_or_reset = set()
    in_use = set(range(num_qubits))

    def add_idle():
        idle = sorted((set(range(num_qubits)) - touched) & in_use)
        if touched and idle:
            noisy_circuit.append("DEPOLARIZE1", idle, p / 10.0)
        waiting = sorted((set(range(num_qubits)) - measured_or_reset) & in_use)
        if measured_or_reset and waiting:
            noisy_circuit.append("DEPOLARIZE1", waiting, 2.0 * p)

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            noisy_body = apply_si1000_noise(instruction.body_copy(), p)
            noisy_circuit.append(stim.CircuitRepeatBlock(instruction.repeat_count, noisy_body))
            continue
        qubits = [t.value for t in instruction.targets_copy() if t.is_qubit_target]
        name = instruction.name
        if name == "TICK":
            add_idle()
            noisy_circuit.append(instruction)
            touched.clear()
            measured_or_reset.clear()
            continue
        elif name == "CZ":
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE2", qubits, p)
        elif name in ("X", "Y", "Z", "H"):
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", qubits, p / 10.0)
        elif name == "R":
            noisy_circuit.append(instruction)
            noisy_circuit.append("X_ERROR", qubits, 2.0 * p)
            measured_or_reset.update(qubits)
        elif name == "M":
            noisy_circuit.append("X_ERROR", qubits, 5.0 * p)
            noisy_circuit.append(instruction)
            measured_or_reset.update(qubits)
        elif name in ("QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"):
            noisy_circuit.append(instruction)
            continue
        touched.update(qubits)
    return noisy_circuit

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate publication benchmark plots for qldpc-rs")
    parser.add_argument("--no-save", action="store_true", help="Run benchmark without overwriting figures on disk")
    args = parser.parse_args()

    print("[+] Loading circuits from circuits/...")
    cx = stim.Circuit.from_file(os.path.join(circ_dir, "X_circuit.stim"))
    cz = stim.Circuit.from_file(os.path.join(circ_dir, "Z_circuit.stim"))

    Hx, Lx, px = extract_dem_info(cx)
    Hz, Lz, pz = extract_dem_info(cz)

    dec_x = QldpcDecoder(Hx, Lx, px, max_bp_iters=25, bp_alpha=0.625)
    dec_z = QldpcDecoder(Hz, Lz, pz, max_bp_iters=25, bp_alpha=0.625)

    # -------------------------------------------------------------------------
    # 1. Plot 1: Empirical Error Rates (Bit LER & Block BLER)
    # -------------------------------------------------------------------------
    print("\n[+] Dynamically sampling 5,000 shots for Plot 1...")
    shots_eval = 5000
    sampler_x = cx.compile_detector_sampler(seed=101)
    sampler_z = cz.compile_detector_sampler(seed=202)

    Dx, y_true_x = sampler_x.sample(shots_eval, separate_observables=True)
    Dz, y_true_z = sampler_z.sample(shots_eval, separate_observables=True)

    y_pred_x = dec_x.decode_batch(Dx.astype(np.uint8))
    y_pred_z = dec_z.decode_batch(Dz.astype(np.uint8))

    # Bit LER
    ler_base_x = float(np.mean(y_true_x != 0))
    ler_base_z = float(np.mean(y_true_z != 0))
    ler_base = (ler_base_x + ler_base_z) / 2.0

    ler_rust_x = float(np.mean(y_true_x != y_pred_x))
    ler_rust_z = float(np.mean(y_true_z != y_pred_z))
    ler_rust = (ler_rust_x + ler_rust_z) / 2.0

    # Block BLER
    bler_base_x = float(np.mean(np.any(y_true_x != 0, axis=1)))
    bler_base_z = float(np.mean(np.any(y_true_z != 0, axis=1)))
    bler_base = (bler_base_x + bler_base_z) / 2.0

    bler_rust_x = float(np.mean(np.any(y_true_x != y_pred_x, axis=1)))
    bler_rust_z = float(np.mean(np.any(y_true_z != y_pred_z, axis=1)))
    bler_rust = (bler_rust_x + bler_rust_z) / 2.0

    red_bit = (ler_base - ler_rust) / ler_base * 100.0
    red_bler = (bler_base - bler_rust) / bler_base * 100.0

    print(f"    - Baseline Bit LER:  {ler_base:.4f} -> qldpc-rs: {ler_rust:.4f} ({red_bit:.1f}% reduction)")
    print(f"    - Baseline Block BLER: {bler_base:.4f} -> qldpc-rs: {bler_rust:.4f} ({red_bler:.1f}% reduction)")

    labels = ["All-Zeros\n(Bit BER)", "qldpc-rs\n(Bit BER)", "All-Zeros\n(Block BLER)", "qldpc-rs\n(Block BLER)"]
    vals = [ler_base, ler_rust, bler_base, bler_rust]
    colors = ["#da3633", "#238636", "#bc4c00", "#1f6feb"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    bars = ax.bar(labels, vals, color=colors, width=0.52, edgecolor="#0d1117", linewidth=1.5)

    for bar, val in zip(bars, vals):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.012,
            f"{val*100:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.annotate(
        f"-{red_bit:.1f}% Bit Error",
        xy=(1, ler_rust),
        xytext=(0.4, 0.16),
        arrowprops=dict(facecolor="#238636", shrink=0.08, width=2, headwidth=7),
        fontweight="bold",
        color="#238636",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#238636", lw=1.2),
    )

    ax.annotate(
        f"-{red_bler:.1f}% Block Error",
        xy=(3, bler_rust),
        xytext=(2.3, 0.38),
        arrowprops=dict(facecolor="#1f6feb", shrink=0.08, width=2, headwidth=7),
        fontweight="bold",
        color="#1f6feb",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#1f6feb", lw=1.2),
    )

    ax.set_ylabel("Error Rate (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Decoding Accuracy on [[18,4,4]] qLDPC Code under SI1000 Noise (p=0.002)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    p1_path = os.path.join(fig_dir, "ler_benchmark.png")
    if not args.no_save:
        plt.savefig(p1_path)
        print(f"[+] Saved: {p1_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Plot 2: Sub-threshold error suppression vs physical noise p
    # -------------------------------------------------------------------------
    print("\n[+] Dynamically sweeping physical noise p for Plot 2...")
    cx_clean = cx.without_noise()
    p_values = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004]
    shots_p = 1500

    base_curve = []
    rust_curve = []

    for p_val in p_values:
        noisy_c = apply_si1000_noise(cx_clean, p=p_val)
        H_p, L_p, priors_p = extract_dem_info(noisy_c)
        dec_p = QldpcDecoder(H_p, L_p, priors_p, max_bp_iters=25, bp_alpha=0.625)

        s_p = noisy_c.compile_detector_sampler(seed=int(p_val * 1e6))
        D_p, y_p = s_p.sample(shots_p, separate_observables=True)

        pred_p = dec_p.decode_batch(D_p.astype(np.uint8))
        base_err = float(np.mean(y_p != 0))
        rust_err = float(np.mean(y_p != pred_p))

        base_curve.append(base_err)
        rust_curve.append(rust_err)
        print(f"    p = {p_val:.4f} -> Baseline LER: {base_err:.4f}, qldpc-rs LER: {rust_err:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(p_values, base_curve, "o--", color="#da3633", label="All-Zeros Baseline", linewidth=2, markersize=7)
    ax.plot(p_values, rust_curve, "s-", color="#238636", label="qldpc-rs (Rust BP-OSD)", linewidth=2.5, markersize=8)

    ax.axvline(x=0.002, color="#58a6ff", linestyle=":", linewidth=1.5, label="Nominal Setting (p = 0.002)")
    idx_comp = p_values.index(0.002)
    ax.scatter([0.002], [rust_curve[idx_comp]], color="#58a6ff", s=110, zorder=5)

    ax.set_xlabel("Physical Error Probability (p)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Per-Observable Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax.set_title("Sub-Threshold Error Suppression under SI1000 Circuit Noise", fontsize=13, fontweight="bold", pad=14)
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="#f6f8fa", edgecolor="#d0d7de")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    p2_path = os.path.join(fig_dir, "ler_vs_noise.png")
    if not args.no_save:
        plt.savefig(p2_path)
        print(f"[+] Saved: {p2_path}")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Plot 3: Multi-Threaded Batch Throughput and Scaling
    # -------------------------------------------------------------------------
    print("\n[+] Dynamically measuring throughput across batch sizes for Plot 3...")
    batch_sizes = [500, 1000, 2500, 5000, 10000]
    sampler = cx.compile_detector_sampler(seed=999)
    D_all, _ = sampler.sample(10000, separate_observables=True)
    D_u8 = D_all.astype(np.uint8)

    # Warmup
    _ = dec_x.decode_batch(D_u8[:100])

    throughputs = []
    latencies = []

    for bs in batch_sizes:
        batch_slice = D_u8[:bs]
        t0 = time.perf_counter()
        _ = dec_x.decode_batch(batch_slice)
        dt = time.perf_counter() - t0

        tps = bs / dt
        lat_us = (dt / bs) * 1e6
        throughputs.append(tps)
        latencies.append(lat_us)
        print(f"    Batch {bs:5d}: {dt:.3f} s ({tps:.0f} shots/s, {lat_us:.1f} us/shot)")

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    c1 = "#1f6feb"
    ax1.set_xlabel("Batch Size (Shots)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Decoding Throughput (Shots / sec)", color=c1, fontsize=12, fontweight="bold")
    ax1.plot(batch_sizes, throughputs, "o-", color=c1, linewidth=2.5, markersize=8)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    c2 = "#8957e5"
    ax2.set_ylabel("Amortized Latency per Shot (µs)", color=c2, fontsize=12, fontweight="bold")
    ax2.plot(batch_sizes, latencies, "s--", color=c2, linewidth=2, markersize=7)
    ax2.tick_params(axis="y", labelcolor=c2)

    plt.title("qldpc-rs Multi-Threaded Rayon Scaling", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    p3_path = os.path.join(fig_dir, "latency_scaling.png")
    if not args.no_save:
        plt.savefig(p3_path)
        print(f"[+] Saved: {p3_path}")
    plt.close()

    if not args.no_save:
        print("\n[+] All benchmark plots generated dynamically and saved to figures/!")
    else:
        print("\n[+] Benchmark completed in dry-run mode (--no-save). Figures left untouched.")

if __name__ == "__main__":
    main()
