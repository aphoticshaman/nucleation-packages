//! LatticeForge Core - High-performance intelligence fusion engine
//!
//! This crate provides Rust implementations of core algorithms:
//! - CIC Framework (Compression-Integration-Coherence)
//! - Gauge-Theoretic Value Clustering (GTVC)
//! - Q-matrix operations for regime transitions
//! - Geodesic integration on Riemannian manifolds
//! - Persistent homology computation
//! - McKean-Vlasov particle dynamics

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
