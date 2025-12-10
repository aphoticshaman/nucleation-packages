//! CausalDampener: Energy-based hypothesis verifier for ARC tasks
//!
//! Implements minimum energy hypothesis selection:
//! E(H) = α·K(H) + β·R(H)
//!
//! Where:
//! - K(H) = description length (AST complexity)
//! - R(H) = invariance penalty (symmetry violation rate)
//!
//! # Example
//!
//! ```python
//! from causal_dampener import CausalDampener, Grid, TrainingPair
//!
//! dampener = CausalDampener(alpha=1.0, beta=10.0, threshold=5.0)
//! energy = dampener.compute_energy(ast_json, training_pairs, executor)
//! ```

pub mod complexity;
pub mod grid;
pub mod invariance;
pub mod transforms;

use pyo3::prelude::*;
use pyo3::types::PyList;

pub use complexity::{compute_description_length, ComplexityScore};
pub use grid::Grid;
pub use invariance::{compute_invariance_penalty, HypothesisExecutor, InvarianceResult, TrainingPair};
pub use transforms::D4Transform;

/// Main verifier struct exposed to Python
#[pyclass]
#[derive(Debug, Clone)]
pub struct CausalDampener {
    /// Weight for complexity term K(H)
    #[pyo3(get, set)]
    pub alpha: f64,

    /// Weight for invariance term R(H)
    #[pyo3(get, set)]
    pub beta: f64,

    /// Energy threshold for filtering
    #[pyo3(get, set)]
    pub threshold: f64,
}

#[pymethods]
impl CausalDampener {
    /// Create a new CausalDampener
    ///
    /// # Arguments
    /// * `alpha` - Weight for complexity K(H), typically 0.5-2.0
    /// * `beta` - Weight for invariance R(H), typically 5.0-20.0
    /// * `threshold` - Energy cutoff for filtering hypotheses
    #[new]
    #[pyo3(signature = (alpha=1.0, beta=10.0, threshold=5.0))]
    pub fn new(alpha: f64, beta: f64, threshold: f64) -> Self {
        CausalDampener {
            alpha,
            beta,
            threshold,
        }
    }

    /// Compute complexity K(H) from AST JSON
    pub fn compute_complexity(&self, ast_json: &str) -> PyResult<f64> {
        compute_description_length(ast_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute full complexity score
    pub fn complexity_score(&self, ast_json: &str) -> PyResult<PyComplexityScore> {
        let score = ComplexityScore::from_ast(ast_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyComplexityScore {
            description_length: score.description_length,
            token_count: score.token_count,
            ast_depth: score.ast_depth,
            cyclomatic: score.cyclomatic,
        })
    }

    /// Compute energy E(H) = α·K(H) + β·R(H)
    ///
    /// Note: This version only computes K(H). For full E(H) with R(H),
    /// use compute_energy_with_executor which requires a Python callback.
    pub fn compute_energy_complexity_only(&self, ast_json: &str) -> PyResult<f64> {
        let k = self.compute_complexity(ast_json)?;
        Ok(self.alpha * k)
    }

    /// Check if hypothesis passes energy threshold (complexity only)
    pub fn passes_threshold_complexity_only(&self, ast_json: &str) -> PyResult<bool> {
        let e = self.compute_energy_complexity_only(ast_json)?;
        Ok(e < self.threshold)
    }

    /// Filter hypotheses by complexity only, return (ast_json, energy) pairs
    pub fn filter_by_complexity(&self, hypotheses: Vec<String>) -> PyResult<Vec<(String, f64)>> {
        let mut results = Vec::new();

        for ast_json in hypotheses {
            match self.compute_energy_complexity_only(&ast_json) {
                Ok(e) if e < self.threshold => {
                    results.push((ast_json, e));
                }
                _ => {}
            }
        }

        // Sort by energy (lowest first)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Ok(results)
    }

    /// Get best hypothesis by complexity (lowest energy)
    pub fn best_by_complexity(&self, hypotheses: Vec<String>) -> PyResult<Option<(String, f64)>> {
        let filtered = self.filter_by_complexity(hypotheses)?;
        Ok(filtered.into_iter().next())
    }
}

/// Python-exposed complexity score
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyComplexityScore {
    #[pyo3(get)]
    pub description_length: f64,
    #[pyo3(get)]
    pub token_count: usize,
    #[pyo3(get)]
    pub ast_depth: usize,
    #[pyo3(get)]
    pub cyclomatic: Option<usize>,
}

#[pymethods]
impl PyComplexityScore {
    /// Get K(H) value
    pub fn k(&self) -> f64 {
        self.description_length
    }

    fn __repr__(&self) -> String {
        format!(
            "ComplexityScore(k={:.2}, tokens={}, depth={}, cyclomatic={:?})",
            self.description_length, self.token_count, self.ast_depth, self.cyclomatic
        )
    }
}

/// Python-exposed grid
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyGrid {
    inner: Grid,
}

#[pymethods]
impl PyGrid {
    #[new]
    pub fn new(data: Vec<Vec<i32>>) -> Self {
        PyGrid {
            inner: Grid::new(data),
        }
    }

    /// Get grid dimensions (height, width)
    pub fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    /// Get cell value
    pub fn get(&self, row: usize, col: usize) -> Option<i32> {
        self.inner.get(row, col)
    }

    /// Get underlying data as nested lists
    pub fn data(&self) -> Vec<Vec<i32>> {
        self.inner.data().clone()
    }

    /// Get unique colors
    pub fn unique_colors(&self) -> Vec<i32> {
        self.inner.unique_colors()
    }

    /// Check equality
    pub fn equals(&self, other: &PyGrid) -> bool {
        self.inner.equals(&other.inner)
    }

    /// Apply D4 transform
    pub fn transform(&self, transform_name: &str) -> PyResult<PyGrid> {
        let t = match transform_name {
            "identity" => D4Transform::Identity,
            "rotate_90" => D4Transform::Rotate90,
            "rotate_180" => D4Transform::Rotate180,
            "rotate_270" => D4Transform::Rotate270,
            "flip_horizontal" => D4Transform::FlipHorizontal,
            "flip_vertical" => D4Transform::FlipVertical,
            "transpose" => D4Transform::FlipDiagonal,
            "anti_transpose" => D4Transform::FlipAntiDiagonal,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown transform: {}", transform_name),
                ))
            }
        };

        Ok(PyGrid {
            inner: t.apply(&self.inner),
        })
    }

    /// Swap two colors
    pub fn swap_colors(&self, a: i32, b: i32) -> PyGrid {
        PyGrid {
            inner: transforms::swap_colors(&self.inner, a, b),
        }
    }

    fn __repr__(&self) -> String {
        let (h, w) = self.inner.dims();
        format!("Grid({}x{})", h, w)
    }
}

/// Initialize the Python module
#[pymodule]
fn causal_dampener(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CausalDampener>()?;
    m.add_class::<PyComplexityScore>()?;
    m.add_class::<PyGrid>()?;

    // Add version info
    m.add("__version__", "0.1.0")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dampener_creation() {
        let dampener = CausalDampener::new(1.0, 10.0, 5.0);
        assert_eq!(dampener.alpha, 1.0);
        assert_eq!(dampener.beta, 10.0);
        assert_eq!(dampener.threshold, 5.0);
    }

    #[test]
    fn test_complexity_computation() {
        let dampener = CausalDampener::new(1.0, 10.0, 100.0);
        let simple_ast = r#"{"_type": "Module", "body": []}"#;
        let k = dampener.compute_complexity(simple_ast).unwrap();
        assert!(k > 0.0);
    }

    #[test]
    fn test_filter_by_complexity() {
        let dampener = CausalDampener::new(1.0, 10.0, 10.0);

        let hypotheses = vec![
            r#"{"a": 1}"#.to_string(),
            r#"{"a": {"b": {"c": {"d": {"e": 1}}}}}"#.to_string(),
        ];

        let filtered = dampener.filter_by_complexity(hypotheses).unwrap();

        // Simpler one should come first
        assert!(!filtered.is_empty());
        assert!(filtered[0].1 < 10.0);
    }
}
