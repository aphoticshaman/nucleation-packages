//! Invariance penalty computation
//!
//! R(H) = (1 / |G| · N) Σ_{g ∈ G} Σ_{i=1}^N 𝟙(H(g(xᵢ)) ≠ g(yᵢ))
//!
//! Tests whether hypothesis respects symmetry group G (D4 + color permutations)

use crate::grid::Grid;
use crate::transforms::{D4Transform, generate_color_swaps, swap_colors};

/// Training pair for invariance testing
#[derive(Debug, Clone)]
pub struct TrainingPair {
    pub input: Grid,
    pub output: Grid,
}

/// Result of invariance testing
#[derive(Debug, Clone)]
pub struct InvarianceResult {
    /// R(H): fraction of symmetry tests that failed
    pub penalty: f64,
    /// Total tests performed
    pub total_tests: usize,
    /// Tests that failed
    pub failures: usize,
    /// Breakdown by transform type
    pub d4_failures: usize,
    pub color_failures: usize,
}

impl InvarianceResult {
    /// Get the R(H) value
    pub fn r(&self) -> f64 {
        self.penalty
    }
}

/// Executor trait for running hypotheses
///
/// Implementations should execute the hypothesis program on an input grid
/// and return the output grid.
pub trait HypothesisExecutor {
    fn execute(&self, input: &Grid) -> Result<Grid, ExecutionError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Hypothesis execution failed: {0}")]
    Failed(String),
    #[error("Hypothesis timed out")]
    Timeout,
}

/// Compute invariance penalty R(H)
///
/// # Arguments
/// * `executor` - Executes the hypothesis on grids
/// * `training_pairs` - List of (input, output) pairs
/// * `test_d4` - Whether to test D4 symmetries
/// * `test_colors` - Whether to test color permutations
pub fn compute_invariance_penalty<E: HypothesisExecutor>(
    executor: &E,
    training_pairs: &[TrainingPair],
    test_d4: bool,
    test_colors: bool,
) -> InvarianceResult {
    let mut total_tests = 0;
    let mut failures = 0;
    let mut d4_failures = 0;
    let mut color_failures = 0;

    // D4 symmetry tests
    if test_d4 {
        for transform in D4Transform::non_identity() {
            for pair in training_pairs {
                total_tests += 1;

                // Transform input
                let transformed_input = transform.apply(&pair.input);

                // Run hypothesis on transformed input
                match executor.execute(&transformed_input) {
                    Ok(actual_output) => {
                        // Expected: transform applied to original output
                        let expected_output = transform.apply(&pair.output);

                        if !actual_output.equals(&expected_output) {
                            failures += 1;
                            d4_failures += 1;
                        }
                    }
                    Err(_) => {
                        // Execution failure counts as invariance failure
                        failures += 1;
                        d4_failures += 1;
                    }
                }
            }
        }
    }

    // Color permutation tests
    if test_colors {
        // Gather all unique colors across all training pairs
        let mut all_colors = std::collections::HashSet::new();
        for pair in training_pairs {
            for c in pair.input.unique_colors() {
                all_colors.insert(c);
            }
            for c in pair.output.unique_colors() {
                all_colors.insert(c);
            }
        }

        // Generate swaps from unique colors
        let colors: Vec<i32> = all_colors.into_iter().collect();
        let swaps: Vec<(i32, i32)> = colors
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| {
                colors[i + 1..].iter().map(move |&b| (a, b))
            })
            .collect();

        // Test color swaps
        for (color_a, color_b) in swaps {
            for pair in training_pairs {
                total_tests += 1;

                // Swap colors in input
                let swapped_input = swap_colors(&pair.input, color_a, color_b);

                // Run hypothesis on color-swapped input
                match executor.execute(&swapped_input) {
                    Ok(actual_output) => {
                        // Expected: color swap applied to original output
                        let expected_output = swap_colors(&pair.output, color_a, color_b);

                        if !actual_output.equals(&expected_output) {
                            failures += 1;
                            color_failures += 1;
                        }
                    }
                    Err(_) => {
                        failures += 1;
                        color_failures += 1;
                    }
                }
            }
        }
    }

    let penalty = if total_tests > 0 {
        failures as f64 / total_tests as f64
    } else {
        0.0
    };

    InvarianceResult {
        penalty,
        total_tests,
        failures,
        d4_failures,
        color_failures,
    }
}

/// Quick invariance check: test only a subset of transforms
pub fn quick_invariance_check<E: HypothesisExecutor>(
    executor: &E,
    training_pairs: &[TrainingPair],
) -> f64 {
    // Only test rotate_180 and one color swap for speed
    let quick_transforms = [D4Transform::Rotate180, D4Transform::FlipHorizontal];

    let mut total = 0;
    let mut failures = 0;

    for transform in quick_transforms {
        for pair in training_pairs {
            total += 1;
            let transformed_input = transform.apply(&pair.input);

            if let Ok(actual) = executor.execute(&transformed_input) {
                let expected = transform.apply(&pair.output);
                if !actual.equals(&expected) {
                    failures += 1;
                }
            } else {
                failures += 1;
            }
        }
    }

    if total > 0 {
        failures as f64 / total as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock executor that implements identity transform
    struct IdentityExecutor;

    impl HypothesisExecutor for IdentityExecutor {
        fn execute(&self, input: &Grid) -> Result<Grid, ExecutionError> {
            Ok(input.clone())
        }
    }

    // Mock executor that returns constant grid
    struct ConstantExecutor(Grid);

    impl HypothesisExecutor for ConstantExecutor {
        fn execute(&self, _input: &Grid) -> Result<Grid, ExecutionError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn test_identity_is_equivariant() {
        let executor = IdentityExecutor;
        let pairs = vec![TrainingPair {
            input: Grid::new(vec![vec![1, 2], vec![3, 4]]),
            output: Grid::new(vec![vec![1, 2], vec![3, 4]]),
        }];

        let result = compute_invariance_penalty(&executor, &pairs, true, false);

        // Identity transform should be perfectly equivariant
        assert_eq!(result.penalty, 0.0);
        assert_eq!(result.failures, 0);
    }

    #[test]
    fn test_constant_violates_equivariance() {
        let constant_output = Grid::new(vec![vec![0, 0], vec![0, 0]]);
        let executor = ConstantExecutor(constant_output);

        let pairs = vec![TrainingPair {
            input: Grid::new(vec![vec![1, 2], vec![3, 4]]),
            output: Grid::new(vec![vec![5, 6], vec![7, 8]]),
        }];

        let result = compute_invariance_penalty(&executor, &pairs, true, false);

        // Constant output won't match transformed expected output
        assert!(result.penalty > 0.0);
        assert!(result.failures > 0);
    }

    #[test]
    fn test_quick_check() {
        let executor = IdentityExecutor;
        let pairs = vec![TrainingPair {
            input: Grid::new(vec![vec![1, 2], vec![3, 4]]),
            output: Grid::new(vec![vec![1, 2], vec![3, 4]]),
        }];

        let r = quick_invariance_check(&executor, &pairs);
        assert_eq!(r, 0.0);
    }
}
