//! LatticeForge Core - High-performance intelligence fusion engine
//!
//! This crate provides Rust implementations of core algorithms:
//! - CIC Framework (Compression-Integration-Coherence)
//! - Gauge-Theoretic Value Clustering (GTVC)
//! - Q-matrix operations for regime transitions
//! - Geodesic integration on Riemannian manifolds
//! - Persistent homology computation
//! - McKean-Vlasov particle dynamics
//! - Graph Laplacian anomaly detection
//! - Dempster-Shafer belief fusion
//! - Transfer entropy causal graphs
//! - Lyapunov exponent regime stability
//! - Cascade probability via belief propagation

#![allow(clippy::needless_range_loop)]

// Core CIC/GTVC modules
pub mod cic;
pub mod clustering;

// Physics-based modules
pub mod q_matrix;
pub mod geodesics;
pub mod persistence;
pub mod particles;
pub mod regime;
pub mod geospatial;

// Anomaly and fusion modules (replacing Python)
pub mod graph_laplacian;
pub mod dempster_shafer;
pub mod transfer_entropy;

// Novel algorithms for intelligence analysis
pub mod lyapunov;
pub mod cascade;
pub mod simulation;

// Content-addressed storage
pub mod briefings;

// Historical pattern matching
pub mod patterns;

// Epistemic uncertainty tracking
pub mod epistemic;

#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience
pub use cic::*;
pub use clustering::*;
pub use q_matrix::*;
pub use geodesics::*;
pub use persistence::*;
pub use particles::*;
pub use regime::*;
pub use geospatial::*;
pub use graph_laplacian::*;
pub use dempster_shafer::*;
pub use transfer_entropy::*;
pub use lyapunov::*;
pub use cascade::*;
pub use briefings::*;
pub use patterns::*;
pub use epistemic::*;
