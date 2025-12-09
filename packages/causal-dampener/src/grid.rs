//! Grid data structure for ARC tasks
//!
//! ARC grids are 2D arrays of integers 0-9 representing colors.
//! Variable size up to 30x30.

use serde::{Deserialize, Serialize};
use std::fmt;

/// ARC grid representation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid {
    data: Vec<Vec<i32>>,
    height: usize,
    width: usize,
}

impl Grid {
    /// Create a new grid from 2D vector
    pub fn new(data: Vec<Vec<i32>>) -> Self {
        let height = data.len();
        let width = if height > 0 { data[0].len() } else { 0 };

        // Validate rectangular
        debug_assert!(data.iter().all(|row| row.len() == width));

        Grid { data, height, width }
    }

    /// Create empty grid of given dimensions
    pub fn empty(height: usize, width: usize) -> Self {
        Grid {
            data: vec![vec![0; width]; height],
            height,
            width,
        }
    }

    /// Get grid dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    /// Get cell value (bounds checked)
    pub fn get(&self, row: usize, col: usize) -> Option<i32> {
        self.data.get(row).and_then(|r| r.get(col).copied())
    }

    /// Set cell value (bounds checked)
    pub fn set(&mut self, row: usize, col: usize, value: i32) -> bool {
        if row < self.height && col < self.width {
            self.data[row][col] = value;
            true
        } else {
            false
        }
    }

    /// Get underlying data
    pub fn data(&self) -> &Vec<Vec<i32>> {
        &self.data
    }

    /// Convert to owned data
    pub fn into_data(self) -> Vec<Vec<i32>> {
        self.data
    }

    /// Count occurrences of each color (0-9)
    pub fn color_histogram(&self) -> [usize; 10] {
        let mut hist = [0usize; 10];
        for row in &self.data {
            for &cell in row {
                if cell >= 0 && cell < 10 {
                    hist[cell as usize] += 1;
                }
            }
        }
        hist
    }

    /// Get unique colors in grid
    pub fn unique_colors(&self) -> Vec<i32> {
        let hist = self.color_histogram();
        hist.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(color, _)| color as i32)
            .collect()
    }

    /// Check if grids are equal
    pub fn equals(&self, other: &Grid) -> bool {
        self == other
    }

    /// Compute hash for quick comparison
    pub fn content_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.data.hash(&mut hasher);
        hasher.finish()
    }
}

impl fmt::Display for Grid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for (i, &cell) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", cell)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl From<Vec<Vec<i32>>> for Grid {
    fn from(data: Vec<Vec<i32>>) -> Self {
        Grid::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let data = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
        ];
        let grid = Grid::new(data);
        assert_eq!(grid.dims(), (2, 3));
    }

    #[test]
    fn test_get_set() {
        let mut grid = Grid::empty(3, 3);
        assert_eq!(grid.get(1, 1), Some(0));
        grid.set(1, 1, 5);
        assert_eq!(grid.get(1, 1), Some(5));
    }

    #[test]
    fn test_color_histogram() {
        let grid = Grid::new(vec![
            vec![0, 1, 1],
            vec![2, 2, 2],
        ]);
        let hist = grid.color_histogram();
        assert_eq!(hist[0], 1);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 3);
    }

    #[test]
    fn test_equality() {
        let g1 = Grid::new(vec![vec![1, 2], vec![3, 4]]);
        let g2 = Grid::new(vec![vec![1, 2], vec![3, 4]]);
        let g3 = Grid::new(vec![vec![1, 2], vec![3, 5]]);

        assert!(g1.equals(&g2));
        assert!(!g1.equals(&g3));
    }
}
