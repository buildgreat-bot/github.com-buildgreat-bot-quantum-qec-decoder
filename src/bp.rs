//! Normalized Min-Sum Belief Propagation engine for qLDPC codes.

use crate::tanner::{DecoderWorkspace, TannerGraph};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BpStatus {
    Converged,
    MaxIterationsReached,
}

pub struct BpDecoder {
    pub max_iterations: usize,
    pub alpha: f32,        // Min-sum normalization factor (e.g. 0.625)
    pub max_llr_clip: f32, // Saturation bound to prevent overflow
}

impl Default for BpDecoder {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            alpha: 0.625,
            max_llr_clip: 20.0,
        }
    }
}

impl BpDecoder {
    pub fn new(max_iterations: usize, alpha: f32) -> Self {
        Self {
            max_iterations,
            alpha,
            max_llr_clip: 20.0,
        }
    }

    /// Runs Belief Propagation on the given syndrome.
    ///
    /// Modifies `ws.posterior_llrs` and `ws.hard_decisions` in place.
    /// Returns `BpStatus::Converged` if a valid error vector matching the syndrome was found.
    pub fn decode(
        &self,
        graph: &TannerGraph,
        syndrome: &[u8],
        ws: &mut DecoderWorkspace,
    ) -> BpStatus {
        ws.reset();

        // Initialize variable-to-check messages with prior LLRs
        for v in 0..graph.num_vars {
            let prior = graph.prior_llrs[v];
            let start = graph.var_offsets[v];
            let end = graph.var_offsets[v + 1];
            for &edge_idx in &graph.var_edge_indices[start..end] {
                ws.msg_v_to_c[edge_idx] = prior;
            }
        }

        for _iter in 0..self.max_iterations {
            // -------------------------------------------------------------
            // Step 1: Check-to-Variable Message Update (Normalized Min-Sum)
            // -------------------------------------------------------------
            for c in 0..graph.num_checks {
                let s = syndrome[c];
                let start = graph.check_offsets[c];
                let end = graph.check_offsets[c + 1];
                let deg = end - start;

                if deg == 0 {
                    continue;
                }

                let mut min1 = f32::INFINITY;
                let mut min2 = f32::INFINITY;
                let mut min1_idx = 0;
                let mut check_sign: i32 = if s == 1 { -1 } else { 1 };

                // First pass: find min1, min2, and cumulative product of signs
                for (local_idx, &edge_idx) in
                    graph.check_edge_indices[start..end].iter().enumerate()
                {
                    let val = ws.msg_v_to_c[edge_idx];
                    let sign = if val < 0.0 { -1 } else { 1 };
                    check_sign *= sign;

                    let abs_val = val.abs();
                    if abs_val < min1 {
                        min2 = min1;
                        min1 = abs_val;
                        min1_idx = local_idx;
                    } else if abs_val < min2 {
                        min2 = abs_val;
                    }
                }

                // Second pass: compute outgoing check-to-variable messages
                for (local_idx, &edge_idx) in
                    graph.check_edge_indices[start..end].iter().enumerate()
                {
                    let in_val = ws.msg_v_to_c[edge_idx];
                    let in_sign = if in_val < 0.0 { -1 } else { 1 };
                    let out_sign = check_sign * in_sign;

                    let mag = if local_idx == min1_idx { min2 } else { min1 };
                    let normalized_mag = (mag * self.alpha).min(self.max_llr_clip);

                    ws.msg_c_to_v[edge_idx] = (out_sign as f32) * normalized_mag;
                }
            }

            // -------------------------------------------------------------
            // Step 2: Variable-to-Check Message Update & Hard Decisions
            // -------------------------------------------------------------
            for v in 0..graph.num_vars {
                let prior = graph.prior_llrs[v];
                let start = graph.var_offsets[v];
                let end = graph.var_offsets[v + 1];

                // Compute total posterior belief S_v = lambda_v + sum(M_c->v)
                let mut total_belief = prior;
                for &edge_idx in &graph.var_edge_indices[start..end] {
                    total_belief += ws.msg_c_to_v[edge_idx];
                }

                ws.posterior_llrs[v] = total_belief;
                ws.hard_decisions[v] = if total_belief < 0.0 { 1 } else { 0 };

                // Extrinsic message to each check: M_v->c = S_v - M_c->v
                for &edge_idx in &graph.var_edge_indices[start..end] {
                    let extrinsic = total_belief - ws.msg_c_to_v[edge_idx];
                    ws.msg_v_to_c[edge_idx] =
                        extrinsic.clamp(-self.max_llr_clip, self.max_llr_clip);
                }
            }

            // -------------------------------------------------------------
            // Step 3: Syndrome Verification (Early Termination Check)
            // -------------------------------------------------------------
            let mut syndrome_match = true;
            for c in 0..graph.num_checks {
                let start = graph.check_offsets[c];
                let end = graph.check_offsets[c + 1];

                let mut parity = 0u8;
                for &v in &graph.check_neighbors[start..end] {
                    parity ^= ws.hard_decisions[v];
                }
                ws.calculated_syndrome[c] = parity;
                if parity != syndrome[c] {
                    syndrome_match = false;
                }
            }

            if syndrome_match {
                return BpStatus::Converged;
            }
        }

        BpStatus::MaxIterationsReached
    }
}
