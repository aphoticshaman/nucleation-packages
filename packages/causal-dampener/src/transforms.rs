//! Grid transformations for invariance testing
//!
//! Implements D4 symmetry group (rotations + reflections) and color permutations.
//! G_ARC ≈ D4 × S_colors

use crate::grid::Grid;

/// D4 symmetry group element
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D4Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    FlipDiagonal,      // transpose
    FlipAntiDiagonal,  // anti-transpose
}

impl D4Transform {
    /// All D4 group elements
    pub fn all() -> &'static [D4Transform] {
        &[
            D4Transform::Identity,
            D4Transform::Rotate90,
            D4Transform::Rotate180,
            D4Transform::Rotate270,
            D4Transform::FlipHorizontal,
            D4Transform::FlipVertical,
            D4Transform::FlipDiagonal,
            D4Transform::FlipAntiDiagonal,
        ]
    }

    /// Non-identity elements (for testing)
    pub fn non_identity() -> &'static [D4Transform] {
        &[
            D4Transform::Rotate90,
            D4Transform::Rotate180,
            D4Transform::Rotate270,
            D4Transform::FlipHorizontal,
            D4Transform::FlipVertical,
            D4Transform::FlipDiagonal,
            D4Transform::FlipAntiDiagonal,
        ]
    }

    /// Apply transformation to grid
    pub fn apply(&self, grid: &Grid) -> Grid {
        match self {
            D4Transform::Identity => grid.clone(),
            D4Transform::Rotate90 => rotate_90(grid),
            D4Transform::Rotate180 => rotate_180(grid),
            D4Transform::Rotate270 => rotate_270(grid),
            D4Transform::FlipHorizontal => flip_horizontal(grid),
            D4Transform::FlipVertical => flip_vertical(grid),
            D4Transform::FlipDiagonal => transpose(grid),
            D4Transform::FlipAntiDiagonal => anti_transpose(grid),
        }
    }
}

/// Rotate grid 90 degrees clockwise
pub fn rotate_90(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; h]; w];

    for row in 0..h {
        for col in 0..w {
            result[col][h - 1 - row] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Rotate grid 180 degrees
pub fn rotate_180(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; w]; h];

    for row in 0..h {
        for col in 0..w {
            result[h - 1 - row][w - 1 - col] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Rotate grid 270 degrees clockwise (= 90 degrees counter-clockwise)
pub fn rotate_270(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; h]; w];

    for row in 0..h {
        for col in 0..w {
            result[w - 1 - col][row] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Flip grid horizontally (left-right mirror)
pub fn flip_horizontal(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; w]; h];

    for row in 0..h {
        for col in 0..w {
            result[row][w - 1 - col] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Flip grid vertically (top-bottom mirror)
pub fn flip_vertical(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; w]; h];

    for row in 0..h {
        for col in 0..w {
            result[h - 1 - row][col] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Transpose grid (swap rows and columns)
pub fn transpose(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; h]; w];

    for row in 0..h {
        for col in 0..w {
            result[col][row] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Anti-transpose (flip along anti-diagonal)
pub fn anti_transpose(grid: &Grid) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; h]; w];

    for row in 0..h {
        for col in 0..w {
            result[w - 1 - col][h - 1 - row] = grid.get(row, col).unwrap_or(0);
        }
    }

    Grid::new(result)
}

/// Color permutation: swap two colors throughout the grid
pub fn swap_colors(grid: &Grid, color_a: i32, color_b: i32) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; w]; h];

    for row in 0..h {
        for col in 0..w {
            let val = grid.get(row, col).unwrap_or(0);
            result[row][col] = if val == color_a {
                color_b
            } else if val == color_b {
                color_a
            } else {
                val
            };
        }
    }

    Grid::new(result)
}

/// Apply color remapping to grid
pub fn remap_colors(grid: &Grid, mapping: &[i32; 10]) -> Grid {
    let (h, w) = grid.dims();
    let mut result = vec![vec![0; w]; h];

    for row in 0..h {
        for col in 0..w {
            let val = grid.get(row, col).unwrap_or(0);
            if val >= 0 && val < 10 {
                result[row][col] = mapping[val as usize];
            } else {
                result[row][col] = val;
            }
        }
    }

    Grid::new(result)
}

/// Generate color swap permutations for colors present in grid
pub fn generate_color_swaps(grid: &Grid) -> Vec<(i32, i32)> {
    let colors = grid.unique_colors();
    let mut swaps = Vec::new();

    for i in 0..colors.len() {
        for j in (i + 1)..colors.len() {
            swaps.push((colors[i], colors[j]));
        }
    }

    swaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> Grid {
        Grid::new(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ])
    }

    #[test]
    fn test_rotate_90() {
        let grid = test_grid();
        let rotated = rotate_90(&grid);
        assert_eq!(rotated.dims(), (3, 2));
        assert_eq!(rotated.get(0, 0), Some(4));
        assert_eq!(rotated.get(0, 1), Some(1));
    }

    #[test]
    fn test_rotate_180() {
        let grid = test_grid();
        let rotated = rotate_180(&grid);
        assert_eq!(rotated.dims(), (2, 3));
        assert_eq!(rotated.get(0, 0), Some(6));
        assert_eq!(rotated.get(1, 2), Some(1));
    }

    #[test]
    fn test_flip_horizontal() {
        let grid = test_grid();
        let flipped = flip_horizontal(&grid);
        assert_eq!(flipped.get(0, 0), Some(3));
        assert_eq!(flipped.get(0, 2), Some(1));
    }

    #[test]
    fn test_flip_vertical() {
        let grid = test_grid();
        let flipped = flip_vertical(&grid);
        assert_eq!(flipped.get(0, 0), Some(4));
        assert_eq!(flipped.get(1, 0), Some(1));
    }

    #[test]
    fn test_transpose() {
        let grid = test_grid();
        let transposed = transpose(&grid);
        assert_eq!(transposed.dims(), (3, 2));
        assert_eq!(transposed.get(0, 0), Some(1));
        assert_eq!(transposed.get(0, 1), Some(4));
        assert_eq!(transposed.get(2, 0), Some(3));
    }

    #[test]
    fn test_swap_colors() {
        let grid = Grid::new(vec![vec![1, 2, 1], vec![2, 1, 2]]);
        let swapped = swap_colors(&grid, 1, 2);
        assert_eq!(swapped.get(0, 0), Some(2));
        assert_eq!(swapped.get(0, 1), Some(1));
    }

    #[test]
    fn test_d4_closure() {
        // Verify D4 group closure: applying any transform twice in sequence
        // produces another D4 element
        let grid = test_grid();

        // 4 rotations by 90 degrees = identity
        let mut g = grid.clone();
        for _ in 0..4 {
            g = rotate_90(&g);
        }
        assert!(g.equals(&grid));

        // Double flip = identity
        let g = flip_horizontal(&flip_horizontal(&grid));
        assert!(g.equals(&grid));
    }
}
