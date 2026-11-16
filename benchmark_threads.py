"""Benchmark and plot multi-core scaling of qldpc-rs across thread counts.

Sweeps RAYON_NUM_THREADS = 1, 2, 4, 8, 12... to measure parallel speedup.
Outputs figures/thread_scaling.png with honest single-axis scaling comparison.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def benchmark_threads():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(repo_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    max_cpus = os.cpu_count() or 1
    # Candidate thread counts up to available CPUs
    candidates = [1, 2, 4, 8, 12, 16, 24, 32]
    threads_to_test = [t for t in candidates if t <= max_cpus]
    if max_cpus not in threads_to_test:
        threads_to_test.append(max_cpus)
        threads_to_test.sort()

    print(f"=================================================================")
    print(f"       qldpc-rs: Rayon Multi-Core Thread Scaling Benchmark       ")
    print(f"=================================================================")
    print(f"[+] Total Detected CPU Cores: {max_cpus}")
    print(f"[+] Thread counts to sweep:   {threads_to_test}")

    shots = 5000
    times = []
    throughputs = []
    latencies = []

    sub_script = f"""
import os, time, numpy as np, stim
from qldpc_rs import QldpcDecoder
from verify_claims import extract_dem_info

cx = stim.Circuit.from_file(os.path.join('{repo_dir.replace(chr(92), "/")}', 'circuits', 'X_circuit.stim'))
H, L, p = extract_dem_info(cx)
dec = QldpcDecoder(H, L, p, max_bp_iters=25, bp_alpha=0.625)

sampler = cx.compile_detector_sampler(seed=42)
D, _ = sampler.sample({shots}, separate_observables=True)
D_u8 = D.astype(np.uint8)

# Warmup
_ = dec.decode_batch(D_u8[:100])

t0 = time.perf_counter()
_ = dec.decode_batch(D_u8)
dt = time.perf_counter() - t0
print(f'RESULT:{{dt:.6f}}')
"""

    for t in threads_to_test:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(t)
        p = subprocess.run(
            [sys.executable, "-c", sub_script],
            env=env,
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        if p.returncode != 0:
            print(f"[-] Error running threads={t}: {p.stderr}")
            continue

        for line in p.stdout.splitlines():
            if line.startswith("RESULT:"):
                dt = float(line.split(":")[1])
                tps = shots / dt
                lat_us = (dt / shots) * 1e6
                times.append(dt)
                throughputs.append(tps)
                latencies.append(lat_us)
                speedup = times[0] / dt
                efficiency = (speedup / t) * 100.0
                print(f"  * Threads {t:2d}: {dt:.3f} s | {tps:6.0f} shots/s | {lat_us:6.1f} µs/shot | Speedup: {speedup:.2f}x ({efficiency:.0f}% efficiency)")

    if not times:
        print("[-] No benchmark results collected.")
        return

    # Plot Thread Scaling with shared scale
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.edgecolor"] = "#30363d"
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    c1 = "#1f6feb"
    c_ideal = "#6e7781"

    # Both measured and ideal are on the same ax1 y-axis
    ideal_throughputs = [throughputs[0] * t for t in threads_to_test]
    line_ideal = ax1.plot(threads_to_test, ideal_throughputs, "--", color=c_ideal, linewidth=1.8, label="Ideal Linear Scaling (100% efficiency)")
    line_meas = ax1.plot(threads_to_test, throughputs, "o-", color=c1, linewidth=2.5, markersize=8, label="Measured Throughput")

    ax1.set_xlabel("Worker Threads (RAYON_NUM_THREADS)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Decoding Throughput (Shots / sec)", color=c1, fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(ideal_throughputs) * 1.08)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Annotate measured speedup and latency
    for t, tps, lat in zip(threads_to_test, throughputs, latencies):
        speedup = tps / throughputs[0]
        ax1.annotate(
            f"{speedup:.1f}x ({lat:.0f}µs)",
            xy=(t, tps),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#0d1117",
        )

    # Right axis: Speedup multiplier perfectly synchronized with left axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Speedup Multiplier", color="#0d1117", fontsize=12, fontweight="bold")
    ax2.set_ylim(ax1.get_ylim()[0] / throughputs[0], ax1.get_ylim()[1] / throughputs[0])

    # Combined legend
    lines = line_meas + line_ideal
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#d0d7de")

    plt.title("qldpc-rs Multi-Threaded Rayon Scaling on [[18,4,4]] Code", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "thread_scaling.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[+] Saved honest thread scaling plot to: {plot_path}")
    print(f"=================================================================\n")

if __name__ == "__main__":
    benchmark_threads()
