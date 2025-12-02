//! LatticeForge Core - High-performance Great Attractor dynamics
//!
//! This crate provides Rust implementations of core algorithms:
//! - Q-matrix operations for regime transitions
//! - Geodesic integration on Riemannian manifolds
//! - Persistent homology computation
//! - McKean-Vlasov particle dynamics

#![allow(clippy::needless_range_loop)]

pub mod q_matrix;
pub mod geodesics;
pub mod persistence;
pub mod particles;
pub mod regime;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use q_matrix::*;
pub use geodesics::*;
pub use persistence::*;
pub use particles::*;
pub use regime::*;
