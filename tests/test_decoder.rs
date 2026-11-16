use qldpc_rs::decoder::QldpcDecoder;

#[test]
fn test_multidecoder_workspace_reuse() {
    // Graph 1: 3 checks, 5 variables
    let h1 = vec![1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0];
    let l1 = vec![1, 0, 0, 0, 1];
    let p1 = vec![0.01; 5];
    let dec1 = QldpcDecoder::new(3, 5, &h1, 1, &l1, &p1, 10, 0.625);

    // Graph 2: 4 checks, 8 variables (larger than Graph 1)
    let h2 = vec![
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0,
    ];
    let l2 = vec![1, 0, 0, 0, 0, 0, 0, 1];
    let p2 = vec![0.01; 8];
    let dec2 = QldpcDecoder::new(4, 8, &h2, 1, &l2, &p2, 10, 0.625);

    // Run small decoder first, then large decoder (tests thread-local resizing bug fix)
    let syn1 = vec![0, 1, 1];
    let res1 = dec1.decode_batch(&syn1, 1);
    assert_eq!(res1.len(), 1);

    let syn2 = vec![0, 1, 1, 0];
    let res2 = dec2.decode_batch(&syn2, 1);
    assert_eq!(res2.len(), 1);

    // Run small again
    let res1_again = dec1.decode_batch(&syn1, 1);
    assert_eq!(res1_again.len(), 1);
}

#[test]
fn test_zero_syndrome() {
    let h = vec![1, 1, 0, 0, 1, 1];
    let l = vec![1, 0, 1];
    let p = vec![0.01; 3];
    let dec = QldpcDecoder::new(2, 3, &h, 1, &l, &p, 10, 0.625);

    let syn = vec![0, 0];
    let res = dec.decode_batch(&syn, 1);
    assert_eq!(res, vec![0]);
}

#[test]
fn test_5bit_repetition_code_single_faults() {
    // 5-bit repetition code: 4 checks, 5 bits, 1 logical observable (bit 0)
    let h = vec![1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1];
    let l = vec![1, 0, 0, 0, 0];
    let priors = vec![0.05f32; 5];
    let dec = QldpcDecoder::new(4, 5, &h, 1, &l, &priors, 20, 0.625);

    // Every single-fault syndrome must decode to the correct logical observable
    let single_fault_syndromes = vec![
        vec![1, 0, 0, 0], // fault 0 -> flips obs 0
        vec![1, 1, 0, 0], // fault 1 -> no flip
        vec![0, 1, 1, 0], // fault 2 -> no flip
        vec![0, 0, 1, 1], // fault 3 -> no flip
        vec![0, 0, 0, 1], // fault 4 -> no flip
    ];
    let expected_observables = [1u8, 0, 0, 0, 0];

    for (j, (syn, &expected)) in single_fault_syndromes
        .iter()
        .zip(expected_observables.iter())
        .enumerate()
    {
        let res = dec.decode_batch(syn, 1);
        assert_eq!(
            res[0], expected,
            "Single fault at bit {} failed: expected {}, got {}",
            j, expected, res[0]
        );
    }

    // Also test all 5 shots in a single parallel batch with return status
    let mut flat_batch = Vec::with_capacity(20);
    for syn in &single_fault_syndromes {
        flat_batch.extend_from_slice(syn);
    }
    let (batch_res, statuses) = dec.decode_batch_with_status(&flat_batch, 5);
    assert_eq!(batch_res, expected_observables);
    assert!(statuses.iter().all(|&s| s));
}

#[test]
fn test_forced_osd0_fallback() {
    use qldpc_rs::bp::BpStatus;

    // 5-bit repetition code: 4 checks, 5 bits, 1 logical observable (bit 0)
    let h = vec![1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1];
    let l = vec![1, 0, 0, 0, 0];
    let priors = vec![0.05f32; 5];

    // max_bp_iters = 1 guarantees that BP cannot converge, forcing execution of OSD-0
    let dec = QldpcDecoder::new(4, 5, &h, 1, &l, &priors, 1, 0.625);

    let syn_fault0 = vec![1, 0, 0, 0];
    let mut ws = dec.graph.create_workspace();
    let bp_status = dec.bp.decode(&dec.graph, &syn_fault0, &mut ws);
    assert_eq!(
        bp_status,
        BpStatus::MaxIterationsReached,
        "BP unexpectedly converged with max_bp_iters = 1"
    );

    // decode_batch_with_status must invoke OSD-0 and return the correct observable flip
    let (res, status) = dec.decode_batch_with_status(&syn_fault0, 1);
    assert_eq!(
        res,
        vec![1],
        "OSD-0 fallback failed to predict observable 1 on non-converged BP"
    );
    assert_eq!(
        status,
        vec![true],
        "OSD-0 fallback failed to satisfy syndrome on non-converged BP"
    );
}

#[test]
fn test_forced_osd0_multiword() {
    use qldpc_rs::bp::BpStatus;

    // 66-bit code with M = 65 checks (> 64, forcing 2 u64 words per check bit-vector)
    let m = 65;
    let n = 66;
    let mut h = vec![0u8; m * n];
    for i in 0..m {
        h[i * n + i] = 1;
        h[i * n + (i + 1)] = 1;
    }
    let mut l = vec![0u8; n];
    l[0] = 1; // Observable is flip on bit 0
    let priors = vec![0.05f32; n];

    // max_bp_iters = 1 guarantees BP cannot converge, forcing OSD-0 across both u64 words
    let dec = QldpcDecoder::new(m, n, &h, 1, &l, &priors, 1, 0.625);

    // Test fault on bit 0: syndrome has check 0 = 1 (resides in word 0)
    let mut syn0 = vec![0u8; m];
    syn0[0] = 1;
    let mut ws = dec.graph.create_workspace();
    assert_eq!(
        dec.bp.decode(&dec.graph, &syn0, &mut ws),
        BpStatus::MaxIterationsReached
    );
    let (res0, stat0) = dec.decode_batch_with_status(&syn0, 1);
    assert_eq!(res0, vec![1]);
    assert_eq!(stat0, vec![true]);

    // Test fault on bit 65: syndrome has check 64 = 1 (resides in word 1, index > 63)
    // Directly guards against OSD ignoring words beyond the first u64
    let mut syn65 = vec![0u8; m];
    syn65[64] = 1;
    let (res65, stat65) = dec.decode_batch_with_status(&syn65, 1);
    assert_eq!(res65, vec![0]);
    assert_eq!(stat65, vec![true]);
}
