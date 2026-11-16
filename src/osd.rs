//! Scalable, zero-allocation Bit-Parallel Ordered Statistics Decoding (OSD-0) over GF(2).
//!
//! Generalized to arbitrary code lengths (scales from [[18,4,4]] to [[144,12,12]] and beyond).
//! All echelon vectors, combinations, and pivot tables are pre-allocated in `DecoderWorkspace`,
//! eliminating all heap allocations during the decoding loop.

use crate::tanner::{DecoderWorkspace, TannerGraph};

#[inline(always)]
fn get_bit(v: &[u64], idx: usize) -> u8 {
    ((v[idx / 64] >> (idx % 64)) & 1) as u8
}

#[inline(always)]
fn set_bit(v: &mut [u64], idx: usize) {
    v[idx / 64] |= 1u64 << (idx % 64);
}

#[inline(always)]
fn first_nonzero_bit(v: &[u64]) -> Option<usize> {
    for (word_idx, &word) in v.iter().enumerate() {
        if word != 0 {
            return Some(word_idx * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

pub struct Osd0Decoder;

impl Osd0Decoder {
    /// Solves H e = s (mod 2) using zero-allocation bit-parallel OSD-0 over GF(2).
    ///
    /// Writes the 0/1 logical observable predictions directly into `out_observables`.
    /// Performs ZERO heap allocations.
    pub fn decode(
        graph: &TannerGraph,
        syndrome: &[u8],
        ws: &mut DecoderWorkspace,
        out_observables: &mut [u8],
    ) -> bool {
        let m = graph.num_checks;
        let w = graph.words_per_check;

        // Reset pre-allocated OSD workspace state
        ws.osd_basis_cols.clear();
        ws.osd_pivot_assigned.fill(false);
        ws.osd_pivot_vectors.fill(0);
        ws.osd_transform.fill(0);
        ws.osd_syn_comb.fill(0);
        ws.osd_syn_vec.fill(0);

        // 1. Pack the syndrome into pre-allocated ws.osd_syn_vec
        let mut is_zero_syndrome = true;
        for (i, &s) in syndrome.iter().take(m).enumerate() {
            if s == 1 {
                set_bit(&mut ws.osd_syn_vec, i);
                is_zero_syndrome = false;
            }
        }

        if is_zero_syndrome {
            out_observables.fill(0);
            return true;
        }

        // 2. Sort variable indices by posterior LLR ascending (most negative first = highest error probability)
        ws.sorted_indices.clear();
        ws.sorted_indices.extend(0..graph.num_vars);
        ws.sorted_indices.sort_unstable_by(|&a, &b| {
            ws.posterior_llrs[a]
                .partial_cmp(&ws.posterior_llrs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Greedy selection of Most Reliable Basis (MRB)
        for &v in &ws.sorted_indices {
            // Pack column v from graph CSR/CSC
            ws.osd_col_vec.fill(0);
            let start = graph.var_offsets[v];
            let end = graph.var_offsets[v + 1];
            for &c in &graph.var_neighbors[start..end] {
                set_bit(&mut ws.osd_col_vec, c);
            }

            ws.osd_comb_vec.fill(0);
            let basis_idx = ws.osd_basis_cols.len();
            if basis_idx < m {
                set_bit(&mut ws.osd_comb_vec, basis_idx);
            }

            // Reduce against existing echelon pivots
            while let Some(lead_r) = first_nonzero_bit(&ws.osd_col_vec) {
                if lead_r >= m {
                    break;
                }
                if ws.osd_pivot_assigned[lead_r] {
                    let p_start = lead_r * w;
                    for i in 0..w {
                        ws.osd_col_vec[i] ^= ws.osd_pivot_vectors[p_start + i];
                        ws.osd_comb_vec[i] ^= ws.osd_transform[p_start + i];
                    }
                } else {
                    // New linearly independent pivot found!
                    ws.osd_pivot_assigned[lead_r] = true;
                    let p_start = lead_r * w;
                    for i in 0..w {
                        ws.osd_pivot_vectors[p_start + i] = ws.osd_col_vec[i];
                        ws.osd_transform[p_start + i] = ws.osd_comb_vec[i];
                    }
                    ws.osd_pivot_to_basis_idx[lead_r] = basis_idx;
                    ws.osd_basis_cols.push(v);
                    break;
                }
            }

            if ws.osd_basis_cols.len() == m {
                break;
            }
        }

        // 4. Reduce syndrome against the echelon basis
        while let Some(lead_r) = first_nonzero_bit(&ws.osd_syn_vec) {
            if lead_r >= m || !ws.osd_pivot_assigned[lead_r] {
                break;
            }
            let p_start = lead_r * w;
            for i in 0..w {
                ws.osd_syn_vec[i] ^= ws.osd_pivot_vectors[p_start + i];
                ws.osd_syn_comb[i] ^= ws.osd_transform[p_start + i];
            }
        }

        // 5. Map basis error e_B to logical observables O = L e (mod 2)
        out_observables.fill(0);
        for (k, obs_vars) in graph.obs_incident_vars.iter().enumerate() {
            let mut parity = 0u8;
            for (basis_idx, &var_idx) in ws.osd_basis_cols.iter().enumerate() {
                if get_bit(&ws.osd_syn_comb, basis_idx) == 1 {
                    if obs_vars.binary_search(&var_idx).is_ok() || obs_vars.contains(&var_idx) {
                        parity ^= 1;
                    }
                }
            }
            out_observables[k] = parity;
        }

        ws.osd_syn_vec.iter().all(|&word| word == 0)
    }
}
