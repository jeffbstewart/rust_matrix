use crate::{Coordinate, Matrix, MatrixAddress, MatrixColumnIterator};

/// Column is a quality-of-life assistant to ease processing matrices
/// in a column-major fashion.
pub struct Column<'a, T, I>
where
    I: Coordinate,
{
    matrix: &'a dyn Matrix<'a, T, I>,
    column: I,
}

impl <'a, T, I> Column<'a, T, I>
where
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>, column: I) -> Self {
        Column{
            matrix,
            column,
        }
    }

    /// row returns the column number this Column represents,  0-based.
    pub fn column(&self) -> I {
        self.column
    }

    /// iter returns a bidirectional iterator over row.
    pub fn iter(&self) -> MatrixColumnIterator<'a, T, I> {
        MatrixColumnIterator::new(self.matrix, self.column)
    }

    /// get retrieves a specified row's cell entry from this column.
    pub fn get(&self, row: I) -> Option<&'a T> {
        self.matrix.get(MatrixAddress{column: self.column, row})
    }
}