//! Top-level hybrid BP-OSD Decoder combining Belief Propagation with OSD-0 fallback.
//!
//! Truly zero-allocation in the hot loop: all operations write into pre-allocated slices.

use crate::bp::{BpDecoder, BpStatus};
use crate::osd::Osd0Decoder;
use crate::tanner::{DecoderWorkspace, TannerGraph};
use rayon::prelude::*;
use std::cell::RefCell;

pub struct QldpcDecoder {
    pub graph: TannerGraph,
    pub bp: BpDecoder,
}

impl QldpcDecoder {
    pub fn new(
        num_checks: usize,
        num_vars: usize,
        h_matrix: &[u8],
        num_obs: usize,
        l_matrix: &[u8],
        priors: &[f32],
        max_bp_iters: usize,
        bp_alpha: f32,
    ) -> Self {
        let graph = TannerGraph::new(num_checks, num_vars, h_matrix, num_obs, l_matrix, priors);
        let bp = BpDecoder::new(max_bp_iters, bp_alpha);
        Self { graph, bp }
    }

    /// Decodes a single syndrome shot into a pre-allocated output buffer `out`.
    ///
    /// Performs ZERO heap allocations.
    pub fn decode_shot(&self, syndrome: &[u8], ws: &mut DecoderWorkspace, out: &mut [u8]) -> bool {
        assert_eq!(out.len(), self.graph.num_obs);
        out.fill(0);

        // 1. Instant Fast Path: zero syndrome
        if syndrome.iter().all(|&s| s == 0) {
            return true;
        }

        // 2. Step 1: Run Normalized Min-Sum Belief Propagation
        let bp_status = self.bp.decode(&self.graph, syndrome, ws);

        if bp_status == BpStatus::Converged {
            // BP found an exact valid error pattern matching the syndrome
            for (k, obs_vars) in self.graph.obs_incident_vars.iter().enumerate() {
                let mut parity = 0u8;
                for &v in obs_vars {
                    parity ^= ws.hard_decisions[v];
                }
                out[k] = parity;
            }
            return true;
        }

        // 3. Step 2: BP did not converge -> OSD-0 Bit-Parallel Fallback
        Osd0Decoder::decode(&self.graph, syndrome, ws, out)
    }

    /// Decodes a batch of shots in parallel across all available CPU threads using Rayon.
    ///
    /// Pre-allocates a single contiguous output buffer and writes directly into it in parallel.
    /// Internal decoding threads perform ZERO heap allocations.
    /// Safely handles different circuit sizes across consecutive calls on the same thread pool.
    pub fn decode_batch(&self, syndromes: &[u8], shots: usize) -> Vec<u8> {
        self.decode_batch_with_status(syndromes, shots).0
    }

    /// Decodes a batch of shots returning both the observable predictions and a success flag per shot.
    pub fn decode_batch_with_status(&self, syndromes: &[u8], shots: usize) -> (Vec<u8>, Vec<bool>) {
        assert_eq!(syndromes.len(), shots * self.graph.num_checks);

        thread_local! {
            static THREAD_WS: RefCell<Option<DecoderWorkspace>> = const { RefCell::new(None) };
        }

        let m = self.graph.num_checks;
        let k = self.graph.num_obs;

        let mut results = vec![0u8; shots * k];
        let mut statuses = vec![false; shots];

        results
            .par_chunks_exact_mut(k)
            .zip(statuses.par_iter_mut())
            .enumerate()
            .for_each(|(shot_idx, (out_shot, out_status))| {
                let syn = &syndromes[shot_idx * m..(shot_idx + 1) * m];
                THREAD_WS.with(|cell| {
                    let mut opt = cell.borrow_mut();
                    let needs_realloc = match opt.as_ref() {
                        None => true,
                        Some(ws) => {
                            ws.posterior_llrs.len() != self.graph.num_vars
                                || ws.msg_c_to_v.len() != self.graph.num_edges
                                || ws.calculated_syndrome.len() != self.graph.num_checks
                        }
                    };
                    if needs_realloc {
                        *opt = Some(self.graph.create_workspace());
                    }
                    let ws = opt.as_mut().unwrap();
                    *out_status = self.decode_shot(syn, ws, out_shot);
                });
            });

        (results, statuses)
    }
}
