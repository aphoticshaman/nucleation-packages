//! Complexity measurement for hypotheses
//!
//! K(H) = description length proxy using AST analysis
//! Formula: tokens × (1 + 0.1 × log(1 + depth))

use serde_json::Value;

/// Compute description length K(H) from AST JSON
///
/// Returns: tokens × (1 + 0.1 × log(1 + depth))
pub fn compute_description_length(ast_json: &str) -> Result<f64, ComplexityError> {
    let ast: Value = serde_json::from_str(ast_json)
        .map_err(|e| ComplexityError::ParseError(e.to_string()))?;

    let tokens = count_ast_nodes(&ast);
    let depth = compute_ast_depth(&ast);

    // K(H) = tokens × (1 + 0.1 × log(1 + depth))
    let k = (tokens as f64) * (1.0 + 0.1 * ((1.0 + depth as f64).ln()));

    Ok(k)
}

/// Count total nodes in AST
pub fn count_ast_nodes(value: &Value) -> usize {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => 1,
        Value::Array(arr) => 1 + arr.iter().map(count_ast_nodes).sum::<usize>(),
        Value::Object(obj) => 1 + obj.values().map(count_ast_nodes).sum::<usize>(),
    }
}

/// Compute maximum depth of AST
pub fn compute_ast_depth(value: &Value) -> usize {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => 1,
        Value::Array(arr) => {
            1 + arr.iter().map(compute_ast_depth).max().unwrap_or(0)
        }
        Value::Object(obj) => {
            1 + obj.values().map(compute_ast_depth).max().unwrap_or(0)
        }
    }
}

/// Compute cyclomatic complexity (for validation)
///
/// Counts decision points: if, for, while, and, or, try, except
pub fn compute_cyclomatic_complexity(ast_json: &str) -> Result<usize, ComplexityError> {
    let ast: Value = serde_json::from_str(ast_json)
        .map_err(|e| ComplexityError::ParseError(e.to_string()))?;

    Ok(count_decision_points(&ast) + 1)
}

/// Count decision points in AST
fn count_decision_points(value: &Value) -> usize {
    let decision_node_types = [
        "If", "For", "While", "And", "Or", "Try", "ExceptHandler",
        "IfExp", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    ];

    let mut count = 0;

    if let Value::Object(obj) = value {
        // Check if this node is a decision point
        if let Some(Value::String(node_type)) = obj.get("_type") {
            if decision_node_types.contains(&node_type.as_str()) {
                count += 1;
            }
        }

        // Recurse into children
        for v in obj.values() {
            count += count_decision_points(v);
        }
    } else if let Value::Array(arr) = value {
        for item in arr {
            count += count_decision_points(item);
        }
    }

    count
}

/// Alternative complexity: simple token count from source code
pub fn count_source_tokens(source: &str) -> usize {
    // Simple tokenization: split on whitespace and punctuation
    source
        .split(|c: char| c.is_whitespace() || "()[]{}:,;".contains(c))
        .filter(|s| !s.is_empty())
        .count()
}

/// Compute combined complexity score
#[derive(Debug, Clone)]
pub struct ComplexityScore {
    pub description_length: f64,
    pub token_count: usize,
    pub ast_depth: usize,
    pub cyclomatic: Option<usize>,
}

impl ComplexityScore {
    pub fn from_ast(ast_json: &str) -> Result<Self, ComplexityError> {
        let ast: Value = serde_json::from_str(ast_json)
            .map_err(|e| ComplexityError::ParseError(e.to_string()))?;

        let token_count = count_ast_nodes(&ast);
        let ast_depth = compute_ast_depth(&ast);
        let description_length = (token_count as f64) * (1.0 + 0.1 * ((1.0 + ast_depth as f64).ln()));
        let cyclomatic = Some(count_decision_points(&ast) + 1);

        Ok(ComplexityScore {
            description_length,
            token_count,
            ast_depth,
            cyclomatic,
        })
    }

    /// Get the primary K(H) value
    pub fn k(&self) -> f64 {
        self.description_length
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ComplexityError {
    #[error("Failed to parse AST JSON: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_nodes() {
        let json = r#"{"a": 1, "b": [1, 2, 3]}"#;
        let value: Value = serde_json::from_str(json).unwrap();
        // Object(2) + Number(1) + Array(1) + Number(3) = 7
        assert_eq!(count_ast_nodes(&value), 7);
    }

    #[test]
    fn test_compute_depth() {
        let json = r#"{"a": {"b": {"c": 1}}}"#;
        let value: Value = serde_json::from_str(json).unwrap();
        // Depth: root -> a -> b -> c -> 1 = 5
        assert_eq!(compute_ast_depth(&value), 5);
    }

    #[test]
    fn test_description_length() {
        let simple = r#"{"value": 42}"#;
        let complex = r#"{"a": {"b": {"c": {"d": 1}}}}"#;

        let k_simple = compute_description_length(simple).unwrap();
        let k_complex = compute_description_length(complex).unwrap();

        // Complex should have higher K
        assert!(k_complex > k_simple);
    }

    #[test]
    fn test_source_tokens() {
        let source = "def foo(x): return x + 1";
        let tokens = count_source_tokens(source);
        // def, foo, x, return, x, +, 1 = 7
        assert!(tokens >= 5);
    }

    #[test]
    fn test_complexity_score() {
        let ast = r#"{"_type": "Module", "body": [{"_type": "If", "test": {"_type": "Name"}}]}"#;
        let score = ComplexityScore::from_ast(ast).unwrap();

        assert!(score.k() > 0.0);
        assert_eq!(score.cyclomatic, Some(2)); // 1 base + 1 If
    }
}
