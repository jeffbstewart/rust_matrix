use crate::{Coordinate, Matrix};
use crate::error::Error;
use crate::dense_matrix::DenseMatrix;
use crate::transpose::TransposedMatrix;

pub fn new_transposed_matrix<'a: 'b, 'b, T, I>(underlay: &'b mut dyn Matrix<'b, T, I>) -> TransposedMatrix<'b, T, I>
where
    I: Coordinate,
{
    TransposedMatrix{
        underlay,
    }
}

/// new_matrix creates a matrix from a vector of values in row-major order.
/// The length of data must be a multiple of rows, and that multiple will become the
/// column_count.
pub fn new_matrix<'a, T, I>(rows: I, data: Vec<T>) -> crate::error::Result<DenseMatrix<T, I>>
where
    T: 'a,
    I: Coordinate,
{
    let zero = I::unit() - I::unit();
    if rows < zero {
        return Err(Error::new("negative row count not supported".to_string()));
    }
    let row_usize: usize = match rows.try_into() {
        Ok(v) => v,
        Err(_) => return Err(Error::new("row count cannot be coerced to usize".to_string())),
    };
    let len = data.len();
    if len == 0 && rows == zero {
        return Ok(DenseMatrix::new(zero, zero, data));
    }
    if len == 0 {
        return Err(Error::new("missing row data".to_string()));
    }
    if len % row_usize != 0 {
        return Err(Error::new(format!("data length {} is not a multiple of rows ({})", len, row_usize)))
    }
    let columns_usize = len / row_usize;
    let columns: I = match columns_usize.try_into() {
        Ok(v) => v,
        Err(_) => return Err(Error::new("cannot convert columns back to I".to_string())),
    };
    Ok(DenseMatrix::new(columns, rows, data))
}

/// new_default_matrix creates a matrix of type T where all cells contain T::default()
/// (typically a zero value).
pub fn new_default_matrix<'a, T, I>(columns: I, rows: I) -> crate::error::Result<DenseMatrix<T, I>>
where
    T: Default,
    I: Coordinate,
{
    let len = match rows.checked_multiply(columns) {
        Some(v) => v,
        None => return Err(Error::new("matrix dimensions exceed chosen index size".to_string())),
    };
    let mut data: Vec<T> = Vec::with_capacity(len);
    for _ in 0..len {
        data.push(T::default());
    }
    new_matrix(rows, data)
}
