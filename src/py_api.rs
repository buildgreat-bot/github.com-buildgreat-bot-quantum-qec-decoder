//! PyO3 Python bindings for qldpc-rs with strict layout, shape, and value validation.

use crate::decoder::QldpcDecoder;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass(name = "QldpcDecoder")]
pub struct PyQldpcDecoder {
    inner: QldpcDecoder,
}

#[pymethods]
impl PyQldpcDecoder {
    #[new]
    #[pyo3(signature = (h_matrix, l_matrix, priors, max_bp_iters=30, bp_alpha=0.625))]
    pub fn new(
        h_matrix: PyReadonlyArray2<u8>,
        l_matrix: PyReadonlyArray2<u8>,
        priors: PyReadonlyArray1<f32>,
        max_bp_iters: usize,
        bp_alpha: f32,
    ) -> PyResult<Self> {
        // Enforce C-contiguous memory layout
        let arr_h = h_matrix.as_array();
        if !arr_h.is_standard_layout() {
            return Err(PyValueError::new_err(
                "h_matrix must be C-contiguous (standard row-major layout)",
            ));
        }

        let arr_l = l_matrix.as_array();
        if !arr_l.is_standard_layout() {
            return Err(PyValueError::new_err(
                "l_matrix must be C-contiguous (standard row-major layout)",
            ));
        }

        let shape_h = h_matrix.shape();
        let num_checks = shape_h[0];
        let num_vars = shape_h[1];

        let shape_l = l_matrix.shape();
        let num_obs = shape_l[0];
        let l_vars = shape_l[1];

        if num_checks == 0 {
            return Err(PyValueError::new_err(
                "h_matrix must have at least 1 row (checks > 0)",
            ));
        }

        if num_vars == 0 {
            return Err(PyValueError::new_err(
                "h_matrix must have at least 1 column (variables > 0)",
            ));
        }

        if num_obs == 0 {
            return Err(PyValueError::new_err(
                "l_matrix must have at least 1 row (observables k > 0)",
            ));
        }

        if num_vars != l_vars {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: H has {} columns, but L has {} columns",
                num_vars, l_vars
            )));
        }

        if priors.shape()[0] != num_vars {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: priors has {} elements, but code has {} variables",
                priors.shape()[0],
                num_vars
            )));
        }

        // Validate max_bp_iters
        if !(1..=10_000).contains(&max_bp_iters) {
            return Err(PyValueError::new_err(format!(
                "Invalid max_bp_iters: {}. max_bp_iters must be between 1 and 10,000",
                max_bp_iters
            )));
        }

        // Validate bp_alpha
        if !bp_alpha.is_finite() || bp_alpha <= 0.0 || bp_alpha > 1.0 {
            return Err(PyValueError::new_err(format!(
                "Invalid bp_alpha: {}. bp_alpha must be finite and in the interval (0, 1]",
                bp_alpha
            )));
        }

        let h_slice = h_matrix.as_slice()?;
        let l_slice = l_matrix.as_slice()?;
        let p_slice = priors.as_slice()?;

        // Validate priors are strictly in (0, 1) and finite
        for &p in p_slice {
            if !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(PyValueError::new_err(format!(
                    "Invalid prior probability: {}. Priors must be finite and in the open interval (0, 1)",
                    p
                )));
            }
        }

        // Validate that H and L matrix entries are strictly binary {0, 1}
        for &val in h_slice {
            if val > 1 {
                return Err(PyValueError::new_err(format!(
                    "Invalid H matrix entry: {}. Parity check matrix entries must be binary (0 or 1)",
                    val
                )));
            }
        }
        for &val in l_slice {
            if val > 1 {
                return Err(PyValueError::new_err(format!(
                    "Invalid L matrix entry: {}. Observable matrix entries must be binary (0 or 1)",
                    val
                )));
            }
        }

        let inner = QldpcDecoder::new(
            num_checks,
            num_vars,
            h_slice,
            num_obs,
            l_slice,
            p_slice,
            max_bp_iters,
            bp_alpha,
        );

        Ok(Self { inner })
    }

    /// Decodes a 2D numpy array of syndromes of shape (shots, num_checks) -> (shots, num_obs).
    ///
    /// If `return_status=True`, returns a tuple `(y_pred, status)` where `status` is a 1D boolean
    /// array indicating per-shot decoding success (BP convergence or exact OSD-0 solution).
    /// Validates C-contiguity, dimensions, and binary values {0, 1}.
    /// Releases the Python GIL during decoding to allow multi-threaded Rayon execution.
    #[pyo3(signature = (syndromes, return_status=false))]
    pub fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        syndromes: PyReadonlyArray2<u8>,
        return_status: bool,
    ) -> PyResult<PyObject> {
        let arr_syn = syndromes.as_array();
        if !arr_syn.is_standard_layout() {
            return Err(PyValueError::new_err(
                "Syndrome array must be C-contiguous (standard row-major layout, e.g. np.ascontiguousarray)",
            ));
        }

        let shape = syndromes.shape();
        let shots = shape[0];
        let num_checks = shape[1];

        if num_checks != self.inner.graph.num_checks {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: Expected syndrome array with {} check columns, got {}",
                self.inner.graph.num_checks, num_checks
            )));
        }

        let syn_slice = syndromes.as_slice()?;

        // Validate that syndrome values are binary {0, 1}
        for &val in syn_slice {
            if val > 1 {
                return Err(PyValueError::new_err(format!(
                    "Invalid syndrome value: {}. Syndrome entries must be 0 or 1",
                    val
                )));
            }
        }

        // Release the Python GIL so Rayon can use all CPU cores at 100% speed
        let (flat_results, statuses) =
            py.allow_threads(|| self.inner.decode_batch_with_status(syn_slice, shots));

        let num_obs = self.inner.graph.num_obs;
        let py_flat = PyArray1::from_vec(py, flat_results);
        let py_2d = py_flat.reshape([shots, num_obs])?;

        if return_status {
            let py_status = PyArray1::from_vec(py, statuses);
            let tuple = PyTuple::new(py, &[py_2d.into_any(), py_status.into_any()])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(py_2d.into_any().unbind())
        }
    }
}
