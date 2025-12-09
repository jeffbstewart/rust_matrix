use crate::{Coordinate, Matrix, MatrixAddress, MatrixRowIterator};

/// Row is a quality-of-life assistant to ease processing matrices
/// in a row-major fashion.
pub struct Row<'a, T, I>
where
    I: Coordinate,
{
    matrix: &'a dyn Matrix<'a, T, I>,
    row: I,
}

impl <'a, T, I> Row<'a, T, I>
where
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>, row: I) -> Self {
        Row{
            matrix,
            row,
        }
    }

    /// row returns the row number this Row represents,  0-based.
    pub fn row(&self) -> I {
        self.row
    }

    /// iter returns a bidirectional iterator over row.
    pub fn iter(&self) -> MatrixRowIterator<'a, T, I> {
        MatrixRowIterator::new(self.matrix, self.row)
    }

    /// get retrieves a specified column's cell entry from this row.
    pub fn get(&self, column: I) -> Option<&'a T> {
        self.matrix.get(MatrixAddress{row: self.row, column})
    }
}