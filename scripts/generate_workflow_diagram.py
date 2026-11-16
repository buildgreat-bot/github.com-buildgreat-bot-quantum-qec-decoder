"""Generate publication-grade system workflow diagram for qldpc-rs.

Outputs:
  - figures/system_workflow.png (300 DPI high-resolution)
  - figures/system_workflow.svg (Scalable vector graphic)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def build_workflow_diagram(output_png: str, output_svg: str):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig = plt.figure(figsize=(16, 11), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas
    bg = patches.Rectangle((0, 0), 100, 100, facecolor="#f8fafc", edgecolor="none", zorder=0)
    ax.add_patch(bg)

    # 1. Header Banner
    header = patches.FancyBboxPatch(
        (2.5, 92.0), 95.0, 6.8,
        boxstyle="round,pad=0.0,rounding_size=0.7",
        facecolor="#0b2545", edgecolor="#134074", linewidth=1.8, zorder=1
    )
    ax.add_patch(header)
    ax.text(4.5, 96.2, r"$\mathbf{qldpc\text{-}rs}$: Tiered Decoding Decision Workflow", fontsize=15.5, fontweight="bold", color="#ffffff", va="center", zorder=2)
    ax.text(4.5, 93.8, "Three-Tiered Fast Cascade: Zero-Syndrome Bypass (<1 µs) → Normalized Min-Sum BP (~216 µs) → Bit-Parallel OSD-0 (~1.7 ms)", fontsize=9.2, color="#93c5fd", va="center", zorder=2)

    # Helper function for card boxes
    def draw_box(x, y, w, h, title, subtitle, header_col, bg_col, border_col):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.4, zorder=1
        )
        ax.add_patch(box)
        band = patches.FancyBboxPatch(
            (x, y + h - 2.6), w, 2.6,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=header_col, edgecolor=border_col, linewidth=1.4, zorder=2
        )
        ax.add_patch(band)
        band_sq = patches.Rectangle((x, y + h - 2.6), w, 1.0, facecolor=header_col, edgecolor="none", zorder=2)
        ax.add_patch(band_sq)
        ax.text(x + 1.8, y + h - 1.3, title, fontsize=9.5, fontweight="bold", color="#ffffff", va="center", zorder=3)
        if subtitle:
            ax.text(x + w - 1.8, y + h - 1.3, subtitle, fontsize=8.0, color="#e0f2fe", ha="right", va="center", zorder=3)

    # =========================================================================
    # LEFT COLUMN: WATERFALL CASCADE (x = 2.5 to 64.0, w = 61.5)
    # =========================================================================

    # --- Step 0: Input Syndrome ---
    draw_box(2.5, 83.0, 61.5, 7.2, "Stage 0: Input Parity-Check Syndrome s", "M = 70 bits • Stim DEM detectors", "#0284c7", "#f0f9ff", "#38bdf8")
    ax.text(4.5, 87.2, "• Syndrome Extraction: Binary detector measurements s ∈ {0, 1}^70 from circuit DEM graph", fontsize=8.2, color="#0f172a", va="center", zorder=3)
    ax.text(4.5, 85.0, "• Memory Ingestion: Read via C-contiguous NumPy 1D slice view (zero copy, zero heap allocation)", fontsize=8.0, color="#475569", va="center", zorder=3)

    # --- Decision 1: s == 0? ---
    d1 = patches.RegularPolygon((33.25, 77.0), numVertices=4, radius=3.2, facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=1.4, zorder=3)
    ax.add_patch(d1)
    ax.text(33.25, 77.8, "Syndrome", fontsize=7.8, fontweight="bold", color="#92400e", ha="center", va="center", zorder=4)
    ax.text(33.25, 76.2, "s == 0?", fontsize=7.8, fontweight="bold", color="#92400e", ha="center", va="center", zorder=4)

    # Arrow from Stage 0 down to Decision 1
    ax.annotate("", xy=(33.25, 80.2), xytext=(33.25, 83.0),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.0, mutation_scale=14), zorder=5)

    # Tier 1 Bypass Box (Inline on Right of Decision 1)
    draw_box(44.0, 73.0, 20.0, 8.0, "Tier 1: Zero-Syndrome Fast Path", "< 1 µs (~3%)", "#059669", "#ecfdf5", "#34d399")
    ax.text(45.5, 77.2, "• All syndrome bits s = 0", fontsize=7.6, color="#065f46", va="center", zorder=3)
    ax.text(45.5, 75.3, "• Output: ŷ = [0, 0, 0, 0]", fontsize=7.8, fontweight="bold", color="#047857", va="center", zorder=3)
    ax.text(45.5, 73.6, "• 0 BP/OSD iterations", fontsize=7.4, color="#475569", va="center", zorder=3)

    # Arrow from Decision 1 to Tier 1
    ax.annotate("", xy=(44.0, 77.0), xytext=(36.5, 77.0),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.8, mutation_scale=13), zorder=5)
    ax.text(40.2, 78.1, "Yes", fontsize=7.6, fontweight="bold", color="#059669", ha="center", va="bottom", zorder=5)

    # --- Step 1 / Tier 2: Normalized Min-Sum BP ---
    draw_box(2.5, 42.5, 61.5, 27.5, "Tier 2: Normalized Min-Sum Belief Propagation (BP)", "default 30; benchmark 25 iters", "#2563eb", "#eff6ff", "#60a5fa")
    
    # Arrow from Decision 1 down to Tier 2
    ax.annotate("", xy=(22.0, 70.0), xytext=(30.0, 77.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.8, mutation_scale=13,
                                connectionstyle="arc3,rad=-0.1"), zorder=5)
    ax.text(23.5, 73.0, "No: s ≠ 0", fontsize=7.5, fontweight="bold", color="#1d4ed8", va="center", zorder=5)

    bp_content = [
        ("Sparse Tanner Graph Architecture:", "#1d4ed8", 8.4, True),
        ("  • Dual CSR (check-to-variable) & CSC (variable-to-check) index views: 70 checks × 72 vars, 432 edges", "#0f172a", 7.8, False),
        ("Step 1: Check-to-Variable Message Update (Two-Minima O(d_c)):", "#1d4ed8", 8.4, True),
        ("  • Check Sign Parity: sgn_c = (-1)^{s_c} · ∏_{v ∈ N(c)} sgn(m_{v→c})", "#0f172a", 7.8, False),
        ("  • Two Running Minima: min1 = min |m|,  min2 = second min |m|", "#0f172a", 7.8, False),
        ("  • Normalization Dampening: α = 0.625 (exact 5/8 fraction dampens overconfident check beliefs)", "#1e3a8a", 8.0, True),
        ("  • Check Message: m_{c→v} = sgn · min(α · magnitude, 20.0)", "#1e3a8a", 7.8, True),
        ("Step 2: Variable-to-Check Message Update & Belief Accumulation:", "#1d4ed8", 8.4, True),
        ("  • Posterior LLR: γ_v = λ_v + ∑_{c ∈ N(v)} m_{c→v}  (channel prior λ_v = ln((1-p)/p))", "#0f172a", 7.8, False),
        ("  • Hard Decision Candidate: ê_v = 1 if γ_v < 0 else 0", "#0f172a", 7.8, False),
        ("  • Variable Message: m_{v→c} = clamp(γ_v - m_{c→v}, ±20.0)", "#0f172a", 7.8, False),
        ("Step 3: Early Syndrome Parity Check:", "#1d4ed8", 8.4, True),
        ("  • Early Stopping: If H ê ≡ s (mod 2), valid correction found; terminate immediately", "#0f172a", 7.8, True),
    ]
    cury = 66.8
    for text, col, fsz, bold in bp_content:
        ax.text(4.5, cury, text, fontsize=fsz, color=col, fontweight="bold" if bold else "normal", va="center", zorder=3)
        cury -= 1.82

    # --- Decision 2: H ê == s? (Converged?) ---
    d2 = patches.RegularPolygon((22.0, 35.5), numVertices=4, radius=3.2, facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=1.4, zorder=3)
    ax.add_patch(d2)
    ax.text(22.0, 36.3, "H ê == s?", fontsize=7.8, fontweight="bold", color="#92400e", ha="center", va="center", zorder=4)
    ax.text(22.0, 34.7, "Converged?", fontsize=7.4, color="#92400e", ha="center", va="center", zorder=4)

    # Arrow from Tier 2 down to Decision 2
    ax.annotate("", xy=(22.0, 38.7), xytext=(22.0, 42.5),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.0, mutation_scale=14), zorder=5)

    # BP Early Exit Box (Inline on Right of Decision 2)
    draw_box(33.0, 32.0, 31.0, 7.5, "BP Early Exit (Hard Decisions)", "~216 µs (~42% of shots)", "#2563eb", "#eff6ff", "#60a5fa")
    ax.text(34.8, 36.0, "• Valid correction ê found; satisfies full syndrome", fontsize=7.6, color="#1e3a8a", va="center", zorder=3)
    ax.text(34.8, 34.2, "• Direct Observable Projection: ŷ_k = L_k · ê (mod 2)", fontsize=7.8, fontweight="bold", color="#1d4ed8", va="center", zorder=3)
    ax.text(34.8, 32.6, "• Bypasses remaining BP rounds and avoids OSD-0 entirely", fontsize=7.4, color="#15803d", va="center", zorder=3)

    # Arrow from Decision 2 to BP Early Exit
    ax.annotate("", xy=(33.0, 35.5), xytext=(25.2, 35.5),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.8, mutation_scale=13), zorder=5)
    ax.text(29.0, 36.6, "Yes (42%)", fontsize=7.5, fontweight="bold", color="#1d4ed8", ha="center", va="bottom", zorder=5)

    # --- Step 2 / Tier 3: Bit-Parallel GF(2) OSD-0 Fallback ---
    draw_box(2.5, 13.0, 61.5, 17.0, "Tier 3: Bit-Parallel GF(2) OSD-0 Fallback", "Max iters reached (~54% of shots) • ~1.7 ms", "#e11d48", "#fff1f2", "#fb7185")

    # Arrow from Decision 2 down to Tier 3
    ax.annotate("", xy=(12.0, 30.0), xytext=(19.0, 35.5),
                arrowprops=dict(arrowstyle="-|>", color="#e11d48", lw=1.8, mutation_scale=13,
                                connectionstyle="arc3,rad=-0.1"), zorder=5)
    ax.text(12.5, 32.5, "No: Max iters (~54%)", fontsize=7.4, fontweight="bold", color="#be123c", va="center", zorder=5)

    osd_content = [
        ("1. In-Place Most Reliable Basis (MRB): Sort columns by posterior magnitude |γ_v| in descending order", "#be123c", 8.0, True),
        ("2. 64-Bit Word Packing: M = 70 check rows packed into W = 2 u64 words for bit-parallel boolean logic", "#0f172a", 7.8, False),
        ("3. Hardware Bit-Scan Pivoting: Find pivot row via lead_r = word.trailing_zeros() (1 cycle TZCNT/BSF)", "#9f1239", 8.0, True),
        ("4. Single-Cycle XOR Gaussian Elimination: Eliminate lower rows via row[r] ^= pivot_row in GF(2)", "#0f172a", 7.8, False),
        ("5. Direct Observable Parity Evaluation: ŷ_k = ⊕_{j ∈ MRB} 1_{basis[j] ∈ obs[k]} (zero error-vector alloc)", "#9f1239", 8.0, True),
    ]
    cury = 26.6
    for text, col, fsz, bold in osd_content:
        ax.text(4.5, cury, text, fontsize=fsz, color=col, fontweight="bold" if bold else "normal", va="center", zorder=3)
        cury -= 2.05

    # --- Step 3 / Output: Decoded Observable View ---
    draw_box(2.5, 2.5, 61.5, 8.5, "Decoded Output View: Predicted Observables ŷ", "ŷ ∈ {0, 1}^4 • k = 4 logical observables", "#0f766e", "#f0fdfa", "#2dd4bf")
    ax.text(4.5, 7.6, "• Return Format: 1D NumPy array of uint8 binary flags for k=4 logical observables", fontsize=8.0, color="#0f766e", fontweight="bold", va="center", zorder=3)
    ax.text(4.5, 5.7, "• Zero Allocation Pipeline: Output written directly into pre-allocated NumPy buffer or caller slice", fontsize=7.8, color="#065f46", va="center", zorder=3)
    ax.text(4.5, 3.8, "• Parity Verification: Observable flips XOR-accumulated against physical circuit targets", fontsize=7.8, color="#475569", va="center", zorder=3)

    # Connections into Output View:
    # 1. From OSD-0 down into Output View
    ax.annotate("", xy=(33.25, 11.0), xytext=(33.25, 13.0),
                arrowprops=dict(arrowstyle="-|>", color="#e11d48", lw=2.0, mutation_scale=14), zorder=5)

    # 2. From BP Early Exit down into Output View (curving down along right edge of Tier 3)
    ax.annotate("", xy=(57.0, 11.0), xytext=(55.0, 32.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.8, mutation_scale=13,
                                connectionstyle="arc3,rad=-0.12"), zorder=5)
    ax.text(58.5, 21.0, "ê proj ŷ", fontsize=7.2, fontweight="bold", color="#1d4ed8", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#60a5fa", lw=0.8, alpha=0.95))

    # 3. From Tier 1 Fast Path down into Output View (long dashed bypass)
    ax.annotate("", xy=(61.5, 11.0), xytext=(61.5, 73.0),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.8, mutation_scale=13, linestyle="dashed"), zorder=5)
    ax.text(61.5, 52.0, "Bypass s=0 (< 1 µs)", fontsize=7.2, fontweight="bold", color="#059669", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#34d399", lw=0.8, alpha=0.95))


    # =========================================================================
    # RIGHT COLUMN: CONTEXT, ARCHITECTURE & VERIFIED METRICS (x = 66.5 to 97.5)
    # =========================================================================

    # Box R1: Target Quantum Code & Noise
    draw_box(66.5, 70.0, 31.0, 20.2, "Quantum Code & Noise Model", "[[18, 4, 4]] Bivariate Bicycle", "#475569", "#f8fafc", "#94a3b8")
    code_info = [
        ("Family:", "Bivariate Bicycle (Gross et al., 2021)"),
        ("Physical Data Qubits:", "N = 72 qubits"),
        ("Parity Checks:", "M = 70 detectors (35 X, 35 Z)"),
        ("Target Observables:", "k = 4 logical qubits"),
        ("Code Distance:", "d = 4 (guaranteed full detection)"),
        ("Tanner Graph Sparsity:", "432 non-zero entries (8.57%)"),
        ("Circuit Noise Model:", "SI1000 circuit depolarizing"),
        ("Physical Error Rate:", "p = 0.002 (standard threshold)"),
        ("Input Representation:", "Stim DEM Detector Error Model"),
    ]
    cury = 86.6
    for label, val in code_info:
        ax.text(68.0, cury, f"• {label}", fontsize=7.8, fontweight="bold", color="#1e293b", va="center", zorder=3)
        ax.text(95.5, cury, val, fontsize=7.8, color="#0f766e" if "p = 0.002" in val or "k = 4" in val else "#475569", ha="right", va="center", zorder=3)
        cury -= 1.80

    # Box R2: Hardware & Systems Architecture
    draw_box(66.5, 41.5, 31.0, 26.5, "Hardware & Systems Architecture", "Rayon Multi-Threading • PyO3", "#334155", "#f8fafc", "#94a3b8")
    sys_info = [
        ("Language Core:", "Pure Rust 2021 (zero unsafe)"),
        ("Python Integration:", "PyO3 0.22 C-extension bridge"),
        ("Multi-Core Pool:", "Rayon work-stealing thread pool"),
        ("Batch Parallelism:", "Release GIL during chunked batch"),
        ("Synchronization:", "0 mutexes / locks in decode path"),
        ("Memory Footprint:", "Contiguous, cache-friendly arrays"),
        ("Inner Loop Alloc:", "0 heap bytes over 3,000 shots"),
        ("Batch Allocation:", "1 single output NumPy array view"),
        ("Multi-Core Scaling:", "~5.0–5.6x across 12 threads"),
        ("Batch Throughput:", "~6,800–7,750 shots/s"),
        ("Effective Latency:", "~130–150 µs/shot on i5-13420H"),
        ("Single-Core Latency:", "~700–1,000 µs/shot (~847 µs typ)"),
    ]
    cury = 64.6
    for label, val in sys_info:
        ax.text(68.0, cury, f"• {label}", fontsize=7.8, fontweight="bold", color="#1e293b", va="center", zorder=3)
        ax.text(95.5, cury, val, fontsize=7.8, color="#0f766e" if "shots/s" in val or "0 heap" in val else "#475569", ha="right", va="center", zorder=3)
        cury -= 1.82

    # Box R3: Audited Benchmarks & Empirical Verification
    draw_box(66.5, 2.5, 31.0, 37.0, "Audited Verification & Accuracy", "p = 0.002 • 3,000 shots / circuit", "#0f766e", "#f0fdfa", "#2dd4bf")

    # Metric Row 1: Bit LER
    ax.text(68.0, 36.2, "Per-Observable Bit LER:", fontsize=8.2, fontweight="bold", color="#1e293b", va="center", zorder=3)
    ax.text(68.0, 34.4, "All-Zeros Baseline: ~28.8%", fontsize=7.8, color="#64748b", va="center", zorder=3)
    ax.text(68.0, 32.7, "qldpc-rs Decoded:    ~5.8%", fontsize=8.0, fontweight="bold", color="#0f766e", va="center", zorder=3)
    ler_pill = patches.FancyBboxPatch((88.5, 32.3), 7.5, 3.8, boxstyle="round,pad=0.0,rounding_size=0.4", facecolor="#dcfce7", edgecolor="#22c55e", linewidth=1.0, zorder=3)
    ax.add_patch(ler_pill)
    ax.text(92.25, 34.2, "~79.8%", fontsize=8.5, fontweight="bold", color="#15803d", ha="center", va="center", zorder=4)

    # Metric Row 2: Block BLER
    ax.text(68.0, 29.5, "Block Error Rate (BLER):", fontsize=8.2, fontweight="bold", color="#1e293b", va="center", zorder=3)
    ax.text(68.0, 27.7, "All-Zeros Baseline: ~57.6%", fontsize=7.8, color="#64748b", va="center", zorder=3)
    ax.text(68.0, 26.0, "qldpc-rs Decoded:   ~10.8%", fontsize=8.0, fontweight="bold", color="#0f766e", va="center", zorder=3)
    bler_pill = patches.FancyBboxPatch((88.5, 25.6), 7.5, 3.8, boxstyle="round,pad=0.0,rounding_size=0.4", facecolor="#dcfce7", edgecolor="#22c55e", linewidth=1.0, zorder=3)
    ax.add_patch(bler_pill)
    ax.text(92.25, 27.5, "~81.4%", fontsize=8.5, fontweight="bold", color="#15803d", ha="center", va="center", zorder=4)

    # Metric Row 3: Reference ldpc Agreement & Invariants
    verif_items = [
        ("Reference ldpc Agreement:", "5,998-5,999 / 6,000 (99.97-99.98%)", "#1e3a8a", True),
        ("Discrepancy Cause:", "f32 vs f64 precision on non-converged BP", "#475569", False),
        ("Deterministic Output:", "Bit-for-bit identical across 1, 3, 8 threads", "#0f766e", False),
        ("Early Exit Ratio:", "~42% shots exit at iteration 2–5", "#1d4ed8", False),
        ("Fallback OSD Ratio:", "~54% shots invoke bit-parallel OSD-0", "#be123c", False),
        ("Zero Syndrome Bypass:", "~3% shots bypass BP/OSD entirely", "#059669", False),
        ("Single-Core Benchmark:", "~847 µs typical (author-measured)", "#475569", False),
    ]
    cury = 22.8
    for title, val, col, bold in verif_items:
        ax.text(68.0, cury, f"• {title}", fontsize=7.6, fontweight="bold", color="#1e293b", va="center", zorder=3)
        cury -= 1.6
        ax.text(70.0, cury, val, fontsize=7.4, fontweight="bold" if bold else "normal", color=col, va="center", zorder=3)
        cury -= 1.85

    # Save outputs
    fig.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(output_svg, format="svg", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Generated PNG: {output_png} (300 DPI)")
    print(f"Generated SVG: {output_svg} (Vector)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(base_dir) == "scripts":
        repo_root = os.path.dirname(base_dir)
    else:
        repo_root = base_dir
    figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    png_path = os.path.join(figures_dir, "system_workflow.png")
    svg_path = os.path.join(figures_dir, "system_workflow.svg")

    build_workflow_diagram(png_path, svg_path)
