//! High-performance cache-aligned Tanner Graph and true zero-allocation workspace.

#[derive(Clone, Debug)]
pub struct TannerGraph {
    pub num_checks: usize,
    pub num_vars: usize,
    pub num_edges: usize,
    pub num_obs: usize,
    pub words_per_check: usize, // (num_checks + 63) / 64

    // CSR structure for Check Nodes (checks -> incident variable nodes)
    pub check_offsets: Vec<usize>,
    pub check_neighbors: Vec<usize>,
    pub check_edge_indices: Vec<usize>,

    // CSC structure for Variable Nodes (variables -> incident check nodes)
    pub var_offsets: Vec<usize>,
    pub var_neighbors: Vec<usize>,
    pub var_edge_indices: Vec<usize>,

    // Prior log-likelihood ratios: lambda_v = ln((1 - p_v) / p_v)
    pub prior_llrs: Vec<f32>,

    // Logical Observable Matrix L (k x n)
    pub obs_incident_vars: Vec<Vec<usize>>,

    // Dense column view of H for fast OSD column extraction
    pub check_matrix_dense: Vec<u8>, // row-major: M * N
}

impl TannerGraph {
    pub fn new(
        num_checks: usize,
        num_vars: usize,
        h_matrix: &[u8],
        num_obs: usize,
        l_matrix: &[u8],
        priors: &[f32],
    ) -> Self {
        assert_eq!(h_matrix.len(), num_checks * num_vars);
        assert_eq!(l_matrix.len(), num_obs * num_vars);
        assert_eq!(priors.len(), num_vars);

        let words_per_check = (num_checks + 63) / 64;

        let mut check_offsets = Vec::with_capacity(num_checks + 1);
        let mut check_neighbors = Vec::new();
        let mut check_edge_indices = Vec::new();

        let mut edge_map = std::collections::HashMap::new();
        let mut edge_counter = 0;

        check_offsets.push(0);
        for c in 0..num_checks {
            for v in 0..num_vars {
                if h_matrix[c * num_vars + v] == 1 {
                    check_neighbors.push(v);
                    check_edge_indices.push(edge_counter);
                    edge_map.insert((c, v), edge_counter);
                    edge_counter += 1;
                }
            }
            check_offsets.push(check_neighbors.len());
        }

        let num_edges = edge_counter;

        let mut var_offsets = Vec::with_capacity(num_vars + 1);
        let mut var_neighbors = Vec::with_capacity(num_edges);
        let mut var_edge_indices = Vec::with_capacity(num_edges);

        var_offsets.push(0);
        for v in 0..num_vars {
            for c in 0..num_checks {
                if h_matrix[c * num_vars + v] == 1 {
                    var_neighbors.push(c);
                    let edge_idx = *edge_map.get(&(c, v)).unwrap();
                    var_edge_indices.push(edge_idx);
                }
            }
            var_offsets.push(var_neighbors.len());
        }

        let prior_llrs: Vec<f32> = priors
            .iter()
            .map(|&p| {
                let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
                ((1.0 - p_clamped) / p_clamped).ln()
            })
            .collect();

        let mut obs_incident_vars = Vec::with_capacity(num_obs);
        for k in 0..num_obs {
            let mut inc = Vec::new();
            for v in 0..num_vars {
                if l_matrix[k * num_vars + v] == 1 {
                    inc.push(v);
                }
            }
            obs_incident_vars.push(inc);
        }

        Self {
            num_checks,
            num_vars,
            num_edges,
            num_obs,
            words_per_check,
            check_offsets,
            check_neighbors,
            check_edge_indices,
            var_offsets,
            var_neighbors,
            var_edge_indices,
            prior_llrs,
            obs_incident_vars,
            check_matrix_dense: h_matrix.to_vec(),
        }
    }

    /// Pre-allocates a completely reusable workspace for zero-heap-allocation decoding.
    pub fn create_workspace(&self) -> DecoderWorkspace {
        let w = self.words_per_check;
        let m = self.num_checks;
        DecoderWorkspace {
            msg_c_to_v: vec![0.0; self.num_edges],
            msg_v_to_c: vec![0.0; self.num_edges],
            posterior_llrs: vec![0.0; self.num_vars],
            hard_decisions: vec![0; self.num_vars],
            calculated_syndrome: vec![0; self.num_checks],
            sorted_indices: (0..self.num_vars).collect(),

            // Pre-allocated OSD structures (scales to arbitrary M without heap alloc)
            osd_basis_cols: Vec::with_capacity(m),
            osd_pivot_vectors: vec![0u64; m * w],
            osd_transform: vec![0u64; m * w],
            osd_pivot_assigned: vec![false; m],
            osd_pivot_to_basis_idx: vec![0usize; m],
            osd_syn_comb: vec![0u64; w],
            osd_syn_vec: vec![0u64; w],
            osd_col_vec: vec![0u64; w],
            osd_comb_vec: vec![0u64; w],
        }
    }
}

/// A zero-allocation workspace holding all temporary state for 1 decoding thread.
pub struct DecoderWorkspace {
    pub msg_c_to_v: Vec<f32>,
    pub msg_v_to_c: Vec<f32>,
    pub posterior_llrs: Vec<f32>,
    pub hard_decisions: Vec<u8>,
    pub calculated_syndrome: Vec<u8>,
    pub sorted_indices: Vec<usize>,

    // Pre-allocated OSD workspace
    pub osd_basis_cols: Vec<usize>,
    pub osd_pivot_vectors: Vec<u64>,
    pub osd_transform: Vec<u64>,
    pub osd_pivot_assigned: Vec<bool>,
    pub osd_pivot_to_basis_idx: Vec<usize>,
    pub osd_syn_comb: Vec<u64>,
    pub osd_syn_vec: Vec<u64>,
    pub osd_col_vec: Vec<u64>,
    pub osd_comb_vec: Vec<u64>,
}

impl DecoderWorkspace {
    /// Resets all buffers for the next shot without reallocating memory.
    pub fn reset(&mut self) {
        self.msg_c_to_v.fill(0.0);
        self.msg_v_to_c.fill(0.0);
        self.posterior_llrs.fill(0.0);
        self.hard_decisions.fill(0);
        self.calculated_syndrome.fill(0);
        self.osd_basis_cols.clear();
        self.osd_pivot_vectors.fill(0);
        self.osd_transform.fill(0);
        self.osd_pivot_assigned.fill(false);
        self.osd_pivot_to_basis_idx.fill(0);
        self.osd_syn_comb.fill(0);
        self.osd_syn_vec.fill(0);
        self.osd_col_vec.fill(0);
        self.osd_comb_vec.fill(0);
    }
}
