"""Unit tests for qldpc_rs Python API and validation checks."""

import pytest
import numpy as np
from qldpc_rs import QldpcDecoder

def test_h_matrix_binary_validation():
    """H matrix with non-binary values (> 1) must raise ValueError."""
    h_bad = np.array([[1, 2], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)

    with pytest.raises(ValueError, match="Invalid H matrix entry: 2"):
        QldpcDecoder(h_bad, l, p)

def test_l_matrix_binary_validation():
    """L matrix with non-binary values (> 1) must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l_bad = np.array([[1, 2]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)

    with pytest.raises(ValueError, match="Invalid L matrix entry: 2"):
        QldpcDecoder(h, l_bad, p)

def test_k_zero_raises_value_error():
    """L matrix with 0 rows (k == 0) must raise ValueError instead of aborting."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l_zero = np.zeros((0, 2), dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="l_matrix must have at least 1 row"):
        QldpcDecoder(h, l_zero, p)

def test_h_zero_columns_raises_value_error():
    """H matrix with 0 columns (N == 0) must raise ValueError."""
    h_zero_col = np.zeros((2, 0), dtype=np.uint8)
    l = np.zeros((1, 0), dtype=np.uint8)
    p = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="h_matrix must have at least 1 column"):
        QldpcDecoder(h_zero_col, l, p)

def test_h_zero_rows_raises_value_error():
    """H matrix with 0 rows (M == 0) must raise ValueError."""
    h_zero_row = np.zeros((0, 2), dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="h_matrix must have at least 1 row"):
        QldpcDecoder(h_zero_row, l, p)

@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, 0.0, -0.05, 1.0, 1.05])
def test_priors_validation_raises_value_error(bad_val):
    """Priors that are NaN, inf, <= 0 or >= 1 must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, bad_val], dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid prior probability"):
        QldpcDecoder(h, l, p)

@pytest.mark.parametrize("bad_alpha", [np.nan, np.inf, -0.5, 0.0, 1.05])
def test_bp_alpha_validation_raises_value_error(bad_alpha):
    """bp_alpha that is NaN, inf, <= 0 or > 1 must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid bp_alpha"):
        QldpcDecoder(h, l, p, bp_alpha=bad_alpha)

@pytest.mark.parametrize("bad_iters", [0, 10_001, 50_000])
def test_max_bp_iters_validation_raises_value_error(bad_iters):
    """max_bp_iters < 1 or > 10_000 must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid max_bp_iters"):
        QldpcDecoder(h, l, p, max_bp_iters=bad_iters)

def test_syndrome_binary_validation():
    """Syndrome containing non-binary values (> 1) must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    dec = QldpcDecoder(h, l, p)

    bad_syn = np.array([[0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid syndrome value: 2"):
        dec.decode_batch(bad_syn)

def test_c_contiguous_requirement():
    """Non-C-contiguous (Fortran-order / transposed) array must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    dec = QldpcDecoder(h, l, p)

    # Create Fortran-order array
    f_arr = np.asfortranarray(np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert not f_arr.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        dec.decode_batch(f_arr)

def test_syndrome_shape_validation():
    """Syndrome with wrong number of detector columns must raise ValueError."""
    h = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    l = np.array([[1, 0]], dtype=np.uint8)
    p = np.array([0.1, 0.1], dtype=np.float32)
    dec = QldpcDecoder(h, l, p)

    bad_shape_syn = np.array([[0, 1, 1]], dtype=np.uint8)  # 3 columns instead of 2
    with pytest.raises(ValueError, match="Dimension mismatch"):
        dec.decode_batch(bad_shape_syn)

def test_zero_syndrome_short_circuit():
    """Zero syndrome must immediately return all-zero observables without running BP/OSD."""
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    l = np.array([[1, 0, 1]], dtype=np.uint8)
    p = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    dec = QldpcDecoder(h, l, p)

    syn = np.zeros((5, 2), dtype=np.uint8)
    pred = dec.decode_batch(syn)
    assert np.array_equal(pred, np.zeros((5, 1), dtype=np.uint8))

def test_decode_batch_return_status():
    """decode_batch with return_status=True returns tuple (y_pred, status)."""
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    l = np.array([[1, 0, 1]], dtype=np.uint8)
    p = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    dec = QldpcDecoder(h, l, p)

    syn = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.uint8)
    y_pred, status = dec.decode_batch(syn, return_status=True)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(status, np.ndarray)
    assert y_pred.shape == (3, 1)
    assert status.shape == (3,)
    assert status.dtype == bool
    assert all(status)

def test_workspace_resize_between_decoders():
    """Reusing threads across decoders of different sizes must not panic or crash."""
    # Decoder 1: smaller graph (3 checks, 5 vars)
    h1 = np.zeros((3, 5), dtype=np.uint8)
    h1[0, 0:2] = 1; h1[1, 1:3] = 1; h1[2, 2:4] = 1
    l1 = np.array([[1, 0, 0, 0, 1]], dtype=np.uint8)
    p1 = np.full(5, 0.05, dtype=np.float32)
    dec1 = QldpcDecoder(h1, l1, p1)

    # Decoder 2: larger graph (6 checks, 12 vars)
    h2 = np.zeros((6, 12), dtype=np.uint8)
    for i in range(6):
        h2[i, i:i+2] = 1
    l2 = np.zeros((2, 12), dtype=np.uint8)
    l2[0, 0] = 1; l2[1, 11] = 1
    p2 = np.full(12, 0.05, dtype=np.float32)
    dec2 = QldpcDecoder(h2, l2, p2)

    syn1 = np.ones((10, 3), dtype=np.uint8)
    syn2 = np.ones((10, 6), dtype=np.uint8)

    # Call smaller first, then larger, then smaller again
    out1 = dec1.decode_batch(syn1)
    assert out1.shape == (10, 1)

    out2 = dec2.decode_batch(syn2)
    assert out2.shape == (10, 2)

    out1_again = dec1.decode_batch(syn1)
    assert np.array_equal(out1, out1_again)
