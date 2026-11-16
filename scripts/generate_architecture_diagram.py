"""Generate publication-grade system architecture diagram for qldpc-rs.

Outputs:
  - figures/system_architecture.png (300 DPI high-resolution)
  - figures/system_architecture.svg (Scalable vector graphic)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def build_architecture_diagram(output_png: str, output_svg: str):
    # Set high quality sans-serif font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    
    # 20 inches wide x 12.2 inches high
    fig = plt.figure(figsize=(20, 12.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas: elegant subtle background
    bg = patches.Rectangle((0, 0), 100, 100, facecolor="#f8fafc", edgecolor="none", zorder=0)
    ax.add_patch(bg)

    # -------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------
    header_box = patches.FancyBboxPatch(
        (1.5, 90.5), 97.0, 8.2,
        boxstyle="round,pad=0.0,rounding_size=0.8",
        facecolor="#0b2545", edgecolor="#134074", linewidth=2.0, zorder=1
    )
    ax.add_patch(header_box)

    ax.text(
        3.0, 96.0,
        r"$\mathbf{qldpc\text{-}rs}$: BP+OSD-0 decoder for qLDPC codes (research/simulation)",
        fontsize=19, fontweight="bold", color="#ffffff", va="center", zorder=2
    )
    ax.text(
        3.0, 93.4,
        "End-to-End Architectural Pipeline: Stim DEM Ingestion  →  PyO3 Bridge  →  Rayon Thread Pool  →  Dual CSR/CSC Min-Sum BP  →  Bit-Parallel GF(2) OSD-0",
        fontsize=10.5, color="#93c5fd", va="center", zorder=2
    )
    
    # Header badge pills on right
    badge1 = patches.FancyBboxPatch(
        (64.5, 92.5), 16.5, 4.4,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor="#1e3a8a", edgecolor="#60a5fa", linewidth=1.2, zorder=2
    )
    ax.add_patch(badge1)
    ax.text(72.75, 94.7, "[[18,4,4]] Bivariate Bicycle", fontsize=9.2, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=3)
    ax.text(72.75, 93.4, r"$M=70,\; N=2470,\; k=4$", fontsize=8.0, color="#bfdbfe", ha="center", va="center", zorder=3)

    badge2 = patches.FancyBboxPatch(
        (82.0, 92.5), 15.5, 4.4,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor="#064e3b", edgecolor="#34d399", linewidth=1.2, zorder=2
    )
    ax.add_patch(badge2)
    ax.text(89.75, 94.7, "Benchmark (representative run)", fontsize=8.5, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=3)
    ax.text(89.75, 93.4, "~6,800–7,750 shots/s (~130–150 µs/shot)", fontsize=7.8, color="#a7f3d0", ha="center", va="center", zorder=3)

    # Helper function for card containers
    def draw_card(x, y, w, h, title, stage_num, header_col, border_col, bg_col):
        # Card outer border & body
        card_body = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            facecolor=bg_col, edgecolor=border_col, linewidth=1.5, zorder=1
        )
        ax.add_patch(card_body)

        # Card header band
        card_header = patches.FancyBboxPatch(
            (x, y + h - 3.4), w, 3.4,
            boxstyle="round,pad=0.0,rounding_size=0.7",
            facecolor=header_col, edgecolor=border_col, linewidth=1.5, zorder=2
        )
        ax.add_patch(card_header)
        
        # Bottom cover for header to make lower corners square inside body
        head_bottom = patches.Rectangle((x, y + h - 3.4), w, 1.2, facecolor=header_col, edgecolor="none", zorder=2)
        ax.add_patch(head_bottom)

        # Stage badge
        badge_circle = patches.Circle((x + 2.0, y + h - 1.7), 1.05, facecolor="#ffffff", edgecolor=border_col, linewidth=1.2, zorder=3)
        ax.add_patch(badge_circle)
        ax.text(x + 2.0, y + h - 1.7, str(stage_num), fontsize=10.5, fontweight="bold", color=header_col, ha="center", va="center", zorder=4)

        # Title text
        ax.text(x + 3.7, y + h - 1.7, title, fontsize=10.5, fontweight="bold", color="#ffffff", va="center", zorder=3)

    # -------------------------------------------------------------
    # STAGE 1: STIM CIRCUIT & DEM INGESTION (Col 1, Top)
    # -------------------------------------------------------------
    draw_card(1.5, 49.0, 22.5, 40.0, "Stim Circuit & DEM Ingestion", "1", "#0284c7", "#38bdf8", "#f0f9ff")
    
    s1_text = [
        (r"$\mathbf{Experimental\;Code:\;[[18,4,4]]}$", "#0369a1", 10.0, True),
        ("• Wang et al., Nature Physics (2026)", "#1e293b", 8.8, False),
        ("• 32 Physical Qubits on Superconducting Chip", "#1e293b", 8.8, False),
        ("  (18 Data Qubits: L0-L8, R0-R8; 14 Ancilla Checks)", "#475569", 8.2, False),
        ("• Memory Circuits: X_circuit.stim / Z_circuit.stim", "#1e293b", 8.8, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{SI1000\;Circuit\text{-}Level\;Noise\;(p=0.002):}$", "#0369a1", 10.0, True),
        ("• 2-Qubit Gates (CZ): Depolarize2(p)", "#1e293b", 8.8, False),
        ("• 1-Qubit Gates (H, X, Y): Depolarize1(p/10)", "#1e293b", 8.8, False),
        ("• Qubit Reset: X_ERROR(2p) | Meas: X_ERROR(5p)", "#1e293b", 8.8, False),
        ("• Resonator Idle during Meas/Reset: Depolarize1(2p)", "#1e293b", 8.8, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Detector\;Error\;Model\;(DEM)\;Extraction:}$", "#0369a1", 10.0, True),
        (r"• Check Matrix: $H \in \mathbb{F}_2^{70 \times 2470}$ (Full GF(2) Rank 70/70)", "#0f172a", 8.8, False),
        (r"• Check Degrees: $\min=20,\;\max=216,\;\mathrm{mean}=152.2$", "#0f172a", 8.8, False),
        (r"• Logical Observables: $L \in \mathbb{F}_2^{4 \times 2470}$", "#0f172a", 8.8, False),
        (r"• Channel Priors: $p_v \in [0.00013, 0.02324]$", "#0f172a", 8.8, False),
        (r"• Prior LLRs: $\lambda_v = \ln\left(\frac{1 - p_v}{p_v}\right)$", "#0369a1", 9.2, True),
    ]
    cur_y = 84.5
    for text_line, col, fsz, bold in s1_text:
        if text_line == "":
            cur_y -= 1.0
            continue
        ax.text(2.6, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 2: PyO3 PYTHON API LAYER (Col 1, Bottom)
    # -------------------------------------------------------------
    draw_card(1.5, 7.5, 22.5, 39.5, "PyO3 Zero-Copy Safe Bridge", "2", "#4338ca", "#818cf8", "#eef2ff")

    s2_text = [
        (r"$\mathbf{Python\;Class:\;QldpcDecoder(H, L, p)}$", "#3730a3", 10.0, True),
        ("• High-level Python bindings via PyO3 0.22", "#1e293b", 8.8, False),
        ("• Zero overhead C-ABI / Python 3.10+ wheels", "#1e293b", 8.8, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Defensive\;Validation\;Matrix:}$", "#3730a3", 10.0, True),
        ("• C-Contiguity Enforcement:", "#1e293b", 8.8, False),
        ("  is_standard_layout() prevents Fortran order bugs", "#475569", 8.2, False),
        ("• Binary Values: H, L, syn ∈ {0, 1} strictly asserted", "#1e293b", 8.8, False),
        ("• Dimension Alignment: H.cols == L.cols == len(p)", "#1e293b", 8.8, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Batch\;Execution:\;decode\_batch(syndromes)}$", "#3730a3", 10.0, True),
        (r"• Input: 2D NumPy array $(S \times M)$ uint8", "#0f172a", 8.8, False),
        (r"• Output Buffer Pre-allocation: $vec![0\mathrm{u}8;\; S \times k]$", "#0f172a", 8.8, False),
        (r"• GIL Release: py.allow_threads(|| ...)", "#166534", 9.2, True),
        ("  Frees Python GIL for multi-threaded scaling", "#166534", 8.4, False),
        ("• Zero memory copying across FFI boundary", "#0f172a", 8.8, False),
        ("• Zero-overhead reshape into NumPy (S, k) view", "#0f172a", 8.8, False),
    ]
    cur_y = 42.0
    for text_line, col, fsz, bold in s2_text:
        if text_line == "":
            cur_y -= 1.0
            continue
        ax.text(2.6, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 3: RAYON MULTI-CORE THREAD POOL (Col 2, Top)
    # -------------------------------------------------------------
    draw_card(26.0, 49.0, 22.5, 40.0, "Rayon Multi-Core Engine", "3", "#7c3aed", "#a78bfa", "#f5f3ff")

    s3_text = [
        (r"$\mathbf{Parallel\;Batch\;Chunking:}$", "#6d28d9", 10.0, True),
        ("• results.par_chunks_exact_mut(k).enumerate()", "#1e293b", 8.8, False),
        ("• Dynamic work-stealing across CPU cores", "#1e293b", 8.8, False),
        ("• Near-linear at 2 threads (1.9x), ~3x at 4 threads", "#1e293b", 8.8, False),
        ("• ~5.0–5.6x across 12 threads (~6,800–7,750 shots/s)", "#1e293b", 8.8, False),
        ("  ~130–150 µs/shot (author-measured on i5-13420H)", "#475569", 8.2, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Thread\text{-}Local\;Workspace\;Storage:}$", "#6d28d9", 10.0, True),
        ("thread_local! static THREAD_WS: RefCell<Option<...>>", "#3730a3", 8.4, True),
        ("• No locks in the decode path (thread-local workspaces)", "#166534", 8.6, True),
        ("• Contiguous, cache-friendly flat arrays", "#1e293b", 8.8, False),
        ("• Exactly 0 heap allocs in steady-state loop", "#166534", 8.8, True),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Dynamic\;Dimension\;Guard:}$", "#6d28d9", 10.0, True),
        ("• Validates (M, N, E) against Tanner graph keys", "#0f172a", 8.8, False),
        ("• Reuses scratch buffers when dimensions match", "#0f172a", 8.8, False),
        ("• Automatically resizes on basis switch (X ↔ Z)", "#0f172a", 8.8, False),
        ("• ws.reset() performs fast in-place memset", "#0f172a", 8.8, False),
        ("  vec.fill(0.0) / vec.clear() with 0 malloc calls", "#475569", 8.2, False),
    ]
    cur_y = 84.5
    for text_line, col, fsz, bold in s3_text:
        if text_line == "":
            cur_y -= 1.0
            continue
        ax.text(27.1, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 4: FAST BYPASS & TANNER GRAPH (Col 2, Bottom)
    # -------------------------------------------------------------
    draw_card(26.0, 7.5, 22.5, 39.5, "Fast Bypass & Tanner Topology", "4", "#059669", "#34d399", "#ecfdf5")

    s4_text = [
        (r"$\mathbf{Fast\;Zero\text{-}Syndrome\;Bypass:}$", "#047857", 10.0, True),
        (r"• Check: if syndrome.iter().all(|&s| s == 0)", "#065f46", 8.6, True),
        (r"• Condition: $s = \mathbf{0} \Rightarrow \hat{y} = \mathbf{0}$", "#065f46", 9.0, True),
        ("• Bypasses BP and OSD entirely", "#1e293b", 8.8, False),
        ("• Latency: < 1 µs (~3% of physical shots)", "#166534", 9.0, True),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Dual\;CSR/CSC\;Tanner\;Graph\;Topology:}$", "#047857", 10.0, True),
        ("• Check-to-Variable CSR (Check Update):", "#1e293b", 8.8, False),
        ("  check_offsets, check_neighbors, check_edge_indices", "#475569", 8.2, False),
        ("• Variable-to-Check CSC (Variable Update):", "#1e293b", 8.8, False),
        ("  var_offsets, var_neighbors, var_edge_indices", "#475569", 8.2, False),
        (r"• Unified Edge Handle: $e \in [0, E)$ links CSR & CSC", "#0f172a", 8.8, True),
        ("• Flat Message Arrays: msg_c_to_v[e], msg_v_to_c[e]", "#0f172a", 8.8, False),
        ("• Contiguous, cache-friendly flat arrays", "#166534", 8.8, True),
    ]
    cur_y = 42.0
    for text_line, col, fsz, bold in s4_text:
        if text_line == "":
            cur_y -= 1.0
            continue
        ax.text(27.1, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 5: NORMALIZED MIN-SUM BP ENGINE (Col 3, Top)
    # -------------------------------------------------------------
    draw_card(50.5, 49.0, 22.5, 40.0, "Normalized Min-Sum BP Engine", "5", "#2563eb", "#60a5fa", "#eff6ff")

    s5_text = [
        (r"$\mathbf{Iterative\;Message\;Passing\;(default\;30;\;benchmarks\;use\;25):}$", "#1d4ed8", 9.5, True),
        (r"$\mathbf{1.\;Check\text{-}to\text{-}Variable\;Update\;(O(d_c)):}$", "#1d4ed8", 9.0, True),
        (r"• Check Sign: $\mathrm{sgn}_c = (-1)^{s_c} \prod_{v} \mathrm{sgn}(m_{v \to c})$", "#0f172a", 8.6, False),
        (r"• 2-Minima Tracking: finds $\min_1$ and $\min_2$", "#0f172a", 8.6, False),
        (r"• Normalization: $\alpha = 0.625 = \frac{5}{8}$ (exact fractional)", "#1e3a8a", 9.0, True),
        (r"• $m_{c \to v} = \mathrm{sgn} \cdot \min(\alpha \cdot \mathrm{mag},\; 20.0)$", "#1e3a8a", 8.8, True),
        ("  Tames overconfidence on dense checks (dc ≤ 216)", "#475569", 8.0, False),
        ("", "#ffffff", 3.0, False),
        (r"$\mathbf{2.\;Variable\text{-}to\text{-}Check\;Update:}$", "#1d4ed8", 9.0, True),
        (r"• Belief: $\gamma_v = \lambda_v + \sum_{c \in N(v)} m_{c \to v}$", "#0f172a", 8.8, False),
        (r"• Hard Decision: $\hat{e}_v = \mathbf{1}_{\{\gamma_v < 0\}}$", "#0f172a", 8.8, False),
        (r"• Variable Message: $m_{v \to c} = \mathrm{clamp}(\gamma_v - m_{c \to v}, \pm 20.0)$", "#0f172a", 8.6, False),
        ("", "#ffffff", 3.0, False),
        (r"$\mathbf{3.\;Syndrome\;Verification\;Check:}$", "#1d4ed8", 9.0, True),
        (r"• Parity: $H \hat{e} = s \ (\mathrm{mod}\ 2)$", "#1e3a8a", 9.0, True),
        ("• Converged? Early exit (~42% shots, ~216 µs)", "#166534", 8.8, True),
        ("• Unresolved? Fallback to OSD-0 (~54% shots)", "#b91c1c", 8.8, True),
    ]
    cur_y = 84.5
    for text_line, col, fsz, bold in s5_text:
        if text_line == "":
            cur_y -= 0.8
            continue
        ax.text(51.6, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 6: BIT-PARALLEL GF(2) OSD-0 (Col 3, Bottom)
    # -------------------------------------------------------------
    draw_card(50.5, 7.5, 22.5, 39.5, "Bit-Parallel GF(2) OSD-0", "6", "#e11d48", "#fb7185", "#fff1f2")

    s6_text = [
        (r"$\mathbf{Most\;Reliable\;Basis\;(MRB)\;Sorting:}$", "#be123c", 10.0, True),
        (r"• Sort columns by $|\gamma_v|$ (ascending LLR)", "#0f172a", 8.8, False),
        ("• Zero allocation: in-place sort_unstable_by", "#166534", 8.6, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{64\text{-}Bit\;Word\;Bitmask\;Packing:}$", "#be123c", 10.0, True),
        (r"• $M = 70 \Rightarrow W = \lceil 70 / 64 \rceil = 2$ words (u64)", "#9f1239", 9.0, True),
        ("• Compact columns: osd_col_vec = [u64; 2]", "#0f172a", 8.8, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{CPU\;Hardware\;Bit\text{-}Scan\;Pivoting:}$", "#be123c", 10.0, True),
        ("• lead_r = word.trailing_zeros() (1 CPU cycle)", "#9f1239", 9.0, True),
        ("  Direct hardware mapping: x86 TZCNT/BSF, ARM CLZ", "#475569", 8.2, False),
        ("", "#ffffff", 4.0, False),
        (r"$\mathbf{Single\text{-}Cycle\;XOR\;Gaussian\;Elimination:}$", "#be123c", 10.0, True),
        ("• Row elimination via bitwise XOR: pivot ^= row", "#0f172a", 8.8, False),
        ("• Complexity O(M²/64) vs O(M³) scalar", "#166534", 8.8, True),
        ("  up to 64x fewer word operations (theoretical)", "#166534", 8.2, False),
        (r"$\mathbf{Direct\;Observable\;Projection:}$", "#be123c", 10.0, True),
        (r"• $\hat{y}_k = \bigoplus_{j} \mathbf{1}_{\{\mathrm{basis}[j] \in \mathrm{obs}[k]\}}$", "#9f1239", 9.0, True),
        ("• Zero allocation of full length-N error vector", "#166534", 8.6, True),
    ]
    cur_y = 42.0
    for text_line, col, fsz, bold in s6_text:
        if text_line == "":
            cur_y -= 1.0
            continue
        ax.text(51.6, cur_y, text_line, fontsize=fsz, color=col, va="center", zorder=3)
        cur_y -= 1.85

    # -------------------------------------------------------------
    # STAGE 7: OUTPUT DELIVERY & BENCHMARKS (Col 4, Full Height)
    # -------------------------------------------------------------
    draw_card(75.0, 7.5, 23.5, 81.5, "Decoded Output & Benchmarks", "7", "#0f766e", "#2dd4bf", "#f0fdfa")

    # Inner Metrics Card (Top part of Stage 7)
    m_box = patches.FancyBboxPatch(
        (76.2, 53.0), 21.1, 31.5,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor="#ffffff", edgecolor="#14b8a6", linewidth=1.2, zorder=2
    )
    ax.add_patch(m_box)

    ax.text(77.2, 82.3, "Verified Empirical Benchmarks (representative run)", fontsize=9.2, fontweight="bold", color="#0f766e", va="center", zorder=3)
    ax.text(77.2, 80.5, "[[18,4,4]] Code @ SI1000 (p=0.002, 5 rounds)", fontsize=8.2, color="#64748b", va="center", zorder=3)

    # Metric Row 1: Bit LER
    ax.text(77.2, 77.2, "Per-Observable Bit LER:", fontsize=8.8, fontweight="bold", color="#1e293b", va="center", zorder=3)
    ax.text(77.2, 75.3, "All-Zeros Baseline:  28.58%", fontsize=8.5, color="#64748b", va="center", zorder=3)
    ax.text(77.2, 73.5, "qldpc-rs Decoded:     5.79%", fontsize=8.8, fontweight="bold", color="#0f766e", va="center", zorder=3)
    
    bler_pill = patches.FancyBboxPatch((89.5, 73.0), 7.2, 3.8, boxstyle="round,pad=0.0,rounding_size=0.4", facecolor="#dcfce7", edgecolor="#22c55e", linewidth=1.0, zorder=3)
    ax.add_patch(bler_pill)
    ax.text(93.1, 74.9, "-79.75%", fontsize=9.2, fontweight="bold", color="#15803d", ha="center", va="center", zorder=4)

    # Metric Row 2: Block BLER
    ax.text(77.2, 70.0, "Block Error Rate (BLER):", fontsize=8.8, fontweight="bold", color="#1e293b", va="center", zorder=3)
    ax.text(77.2, 68.1, "All-Zeros Baseline:  57.59%", fontsize=8.5, color="#64748b", va="center", zorder=3)
    ax.text(77.2, 66.3, "qldpc-rs Decoded:    10.54%", fontsize=8.8, fontweight="bold", color="#0f766e", va="center", zorder=3)

    bler2_pill = patches.FancyBboxPatch((89.5, 65.8), 7.2, 3.8, boxstyle="round,pad=0.0,rounding_size=0.4", facecolor="#dcfce7", edgecolor="#22c55e", linewidth=1.0, zorder=3)
    ax.add_patch(bler2_pill)
    ax.text(93.1, 67.7, "-81.70%", fontsize=9.2, fontweight="bold", color="#15803d", ha="center", va="center", zorder=4)

    # Metric Row 3: Throughput & Latency
    ax.text(77.2, 63.0, "Execution Throughput & Latency:", fontsize=8.6, fontweight="bold", color="#1e293b", va="center", zorder=3)
    ax.text(77.2, 61.2, "• Single-Core:  ~700–1,000 µs/shot (representative: ~847 µs)", fontsize=8.0, color="#1e293b", va="center", zorder=3)
    ax.text(77.2, 59.4, "• Multi-Core:   ~6,800–7,750 shots/s (~130–150 µs/shot)", fontsize=8.0, fontweight="bold", color="#0f766e", va="center", zorder=3)
    ax.text(77.2, 57.6, "• Rayon Speedup: ~5.0–5.6x across 12 threads (author-measured)", fontsize=7.8, color="#1e293b", va="center", zorder=3)
    ax.text(77.2, 55.6, "• ldpc BP+OSD-0 agreement: 5,998-5,999 / 6,000 (99.97-99.98%)", fontsize=7.8, fontweight="bold", color="#1e3a8a", va="center", zorder=3)

    # Inner Invariants Card (Bottom part of Stage 7)
    inv_box = patches.FancyBboxPatch(
        (76.2, 9.2), 21.1, 41.5,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        facecolor="#ffffff", edgecolor="#14b8a6", linewidth=1.2, zorder=2
    )
    ax.add_patch(inv_box)

    ax.text(77.2, 48.0, "Core Architectural Invariants", fontsize=10.5, fontweight="bold", color="#0f766e", va="center", zorder=3)
    
    invariants = [
        ("✔ Zero Heap Allocation in Hot Loop", "#15803d", True),
        ("  Pre-allocated DecoderWorkspace per worker thread", "#475569", False),
        ("✔ Deterministic Output", "#15803d", True),
        ("  Identical results on repeated runs and seeds", "#475569", False),
        ("✔ Defensive Boundary Validation", "#15803d", True),
        ("  Strict C-contiguity and binary bounds check", "#475569", False),
        ("✔ Sub-Microsecond Fast Bypass", "#15803d", True),
        ("  Zero-syndrome shots short-circuit in < 1 µs", "#475569", False),
        ("✔ Hardware Bit-Scan Instructions", "#15803d", True),
        ("  Direct CPU TZCNT/BSF instructions for OSD pivot", "#475569", False),
        ("✔ Single-Cycle XOR Linear Algebra", "#15803d", True),
        ("  Bit-packed u64 rows: up to 64x fewer word operations (theor.)", "#475569", False),
        ("✔ Direct Observable Parity Projection", "#15803d", True),
        ("  No memory materialization of length-N error vector", "#475569", False),
        ("✔ No Locks in the Decode Path", "#15803d", True),
        ("  Zero thread contention via thread_local! storage", "#475569", False),
    ]
    cur_y = 45.2
    for itxt, icol, ibold in invariants:
        ax.text(77.2, cur_y, itxt, fontsize=8.0 if not ibold else 8.4, fontweight="bold" if ibold else "normal", color=icol, va="center", zorder=3)
        cur_y -= 2.2

    # -------------------------------------------------------------
    # PIPELINE CONNECTING ARROWS & DATAFLOW PATHS
    # -------------------------------------------------------------
    arrow_props_main = dict(arrowstyle="-|>", color="#0284c7", lw=2.2, mutation_scale=16)
    arrow_props_branch = dict(arrowstyle="-|>", color="#059669", lw=2.0, mutation_scale=15)
    arrow_props_bp = dict(arrowstyle="-|>", color="#2563eb", lw=2.0, mutation_scale=15)
    arrow_props_osd = dict(arrowstyle="-|>", color="#e11d48", lw=2.0, mutation_scale=15)
    arrow_props_out = dict(arrowstyle="-|>", color="#0f766e", lw=2.2, mutation_scale=16)

    # 1 -> 2: Stim Ingestion to PyO3 API
    ax.annotate("", xy=(12.75, 47.0), xytext=(12.75, 49.0), arrowprops=arrow_props_main, zorder=5)
    ax.text(13.2, 48.0, "H, L, p arrays", fontsize=7.5, color="#0369a1", fontweight="bold", va="center", zorder=5)

    # 2 -> 3: PyO3 API to Rayon Thread Pool
    ax.annotate("", xy=(26.0, 68.0), xytext=(24.0, 28.0),
                arrowprops=dict(arrowstyle="-|>", color="#4338ca", lw=2.2, mutation_scale=16,
                                connectionstyle="angle,angleA=0,angleB=90,rad=4"), zorder=5)
    ax.text(25.0, 48.0, "GIL released\nbatch chunking", fontsize=7.2, color="#3730a3", fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#818cf8", lw=0.8, alpha=0.95))

    # 3 -> 4: Rayon Pool to Fast Bypass & Tanner Graph
    ax.annotate("", xy=(37.25, 47.0), xytext=(37.25, 49.0), arrowprops=arrow_props_branch, zorder=5)
    ax.text(37.7, 48.0, "Syndrome s", fontsize=7.5, color="#047857", fontweight="bold", va="center", zorder=5)

    # 4 -> 5: Non-zero syndrome to Min-Sum BP
    ax.annotate("", xy=(50.5, 68.0), xytext=(48.5, 28.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.2, mutation_scale=16,
                                connectionstyle="angle,angleA=0,angleB=90,rad=4"), zorder=5)
    ax.text(49.5, 48.0, "s != 0\nCSR/CSC Msg", fontsize=7.2, color="#1d4ed8", fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#60a5fa", lw=0.8, alpha=0.95))

    # 4 -> 7: Zero-Syndrome Fast Path Bypass to Output
    # ROUTE UNDERNEATH Card 6 in the open corridor (y = 6.85) to avoid ANY overlap with Card 6 text!
    ax.annotate("", xy=(75.0, 6.85), xytext=(48.5, 6.85),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.2, mutation_scale=16, linestyle="dashed"), zorder=5)
    # Connector down from Card 4 to the corridor
    ax.plot([48.5, 48.5], [8.5, 6.85], color="#059669", lw=2.0, linestyle="dashed", zorder=5)
    ax.text(61.75, 6.85, "Fast Path: s = 0 → ŷ = 0 (< 1 µs, ~3% of physical shots)",
            fontsize=8.0, color="#065f46", fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#34d399", lw=0.9, alpha=0.98))

    # 5 -> 6: BP Max Iters Fallback to OSD-0
    ax.annotate("", xy=(61.75, 47.0), xytext=(61.75, 49.0), arrowprops=arrow_props_osd, zorder=5)
    ax.text(62.2, 48.0, "Max iters reached (~54% of shots)", fontsize=7.2, color="#be123c", fontweight="bold", va="center", zorder=5)

    # 5 -> 7: BP Early Convergence to Output
    ax.annotate("", xy=(75.0, 70.0), xytext=(73.0, 70.0), arrowprops=arrow_props_bp, zorder=5)
    ax.text(74.0, 71.3, "Converged (~42%)", fontsize=7.5, color="#1d4ed8", fontweight="bold", ha="center", va="bottom", zorder=5)

    # 6 -> 7: OSD-0 Solved Parity to Output
    ax.annotate("", xy=(75.0, 32.0), xytext=(73.0, 32.0), arrowprops=arrow_props_out, zorder=5)
    ax.text(74.0, 33.3, r"MRB Solution $\hat{y}$", fontsize=7.5, color="#0f766e", fontweight="bold", ha="center", va="bottom", zorder=5)

    # -------------------------------------------------------------
    # FOOTER BAR WITH SCHOLARLY CITATIONS
    # -------------------------------------------------------------
    footer_box = patches.FancyBboxPatch(
        (1.5, 1.2), 97.0, 5.0,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.2, zorder=1
    )
    ax.add_patch(footer_box)

    ax.text(2.5, 4.4, "Key References & Mathematical Standards:", fontsize=8.2, fontweight="bold", color="#334155", va="center", zorder=2)
    ax.text(2.5, 2.7,
            "[1] Wang et al., Nature Physics (2026), Experimental realization of bivariate bicycle codes on a superconducting quantum processor. doi:10.1038/s41567-025-03157-4\n"
            "[2] C. Gidney, M. Newman, A. Fowler, M. Broughton, \"A Fault-Tolerant Honeycomb Memory\", Quantum 5, 605 (2021), doi:10.22331/q-2021-12-20-605  |  [3] Roffe et al., ldpc C++ library",
            fontsize=7.2, color="#64748b", va="center", zorder=2)

    # Save to PNG (300 DPI) and SVG
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
    repo_figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(repo_figures_dir, exist_ok=True)
    
    png_path = os.path.join(repo_figures_dir, "system_architecture.png")
    svg_path = os.path.join(repo_figures_dir, "system_architecture.svg")
    
    build_architecture_diagram(png_path, svg_path)
