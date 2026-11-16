#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::manual_div_ceil
)]

pub mod bp;
pub mod decoder;
pub mod osd;
pub mod tanner;

#[cfg(feature = "python")]
pub mod py_api;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn qldpc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_api::PyQldpcDecoder>()?;
    Ok(())
}
