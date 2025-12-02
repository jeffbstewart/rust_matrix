// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use crate::error::{Error, Result};
use crate::iter::{MatrixForwardIndexedIterator, MatrixForwardIterator, MatrixRowIterator};
use crate::matrix_address::MatrixAddress;
use crate::traits::{Coordinate, Tensor};
use std::ops::{Index, IndexMut, Range};
use crate::{MatrixColumnIterator, MatrixColumnsIterator, MatrixRowsIterator};

/// Matrix is a rectangular store of type T, providing a variety of
/// useful iterator patterns.
#[derive(Debug)]
pub struct Matrix<T, I>
where
    I: Coordinate,
{
    columns: I,
    rows: I,
    data: Vec<T>,
}

impl<'a, T, I> Matrix<T, I>
where
    I: Coordinate,
{
    /// row_count returns the number of horizontal rows stored in the Matrix.
    pub fn row_count(&self) -> I {
        self.rows
    }

    /// column_count returns the number of vertical columns stored in the Matrix.
    pub fn column_count(&self) -> I {
        self.columns
    }

    fn index_address(&self, address: MatrixAddress<I>) -> usize {
        match (address.row * self.columns + address.column).try_into() {
            Ok(v) => v,
            Err(_) => panic!("address overflows usize.  This should be unreachable."),
        }
    }

    pub fn addresses(&self) -> MatrixForwardIterator<I> {
        MatrixForwardIterator::new(MatrixAddress {
            column: self.columns,
            row: self.rows,
        })
    }

    pub fn indexed_iter(&self) -> MatrixForwardIndexedIterator<'_, T, I> {
        MatrixForwardIndexedIterator::new(self)
    }


    pub fn row(&'a self, row_num: I) -> Option<Row<'a, T, I>> {
        if row_num < I::unit() - I::unit() || row_num >= self.rows {
            None
        } else {
            Some(Row::new(self, row_num))
        }
    }

    pub fn column(&'a self, column_num: I) -> Option<Column<'a, T, I>> {
        if column_num < I::unit() - I::unit() || column_num >= self.columns {
            None
        } else {
            Some(Column::new(self, column_num))
        }
    }

    pub fn rows(&'a self) -> MatrixRowsIterator<'a, T, I> {
        MatrixRowsIterator::new(self)
    }

    pub fn columns(&'a self) -> MatrixColumnsIterator<'a, T, I> {
        MatrixColumnsIterator::new(self)
    }
}

impl<'a, T, I> Matrix<T, I>
where
    I: Coordinate,
{
    pub fn new<F>(columns: I, rows: I, factory_fn: F) -> Result<Self>
    where
        F: Fn(MatrixAddress<I>) -> T,
    {
        let zero = I::unit() - I::unit();
        if columns < zero || rows < zero {
            return Err(Error::new(
                "negative Matrix dimensions are not supported".to_string(),
            ));
        }
        if (columns == zero || rows == zero) && (columns != zero || rows != zero) {
            return Err(Error::new(
                "zero x non-zero Matrix dimensions are not supported".to_string(),
            ));
        }
        let capacity: usize = match rows.checked_multiply(columns) {
            Some(v) => v,
            None => {
                return Err(Error::new(format!(
                    "rows {} * columns {} overflows vector max capacity",
                    rows, columns
                )));
            },
        };
        let data = match std::panic::catch_unwind(||
            Vec::<T>::with_capacity(capacity)
        ) {
            Ok(v) => v,
            Err(_) => {
                return Err(Error::new(
                    "requested Matrix exceeds allocatable vector size".to_string()
                ));
            },
        };
        let mut matrix = Matrix {
            columns,
            rows,
            data,
        };
        matrix
            .addresses()
            .for_each(|a| matrix.data.push(factory_fn(a)));
        Ok(matrix)
    }
}

impl<'a, T, I> Matrix<T, I>
where
    T: Default,
    I: Coordinate,
{
    pub fn new_default(columns: I, rows: I) -> Result<Matrix<T, I>> {
        Self::new(columns, rows, |_| T::default())
    }
}

impl<'a, T: 'a, I> Tensor<'a, T, I, MatrixAddress<I>, 2> for Matrix<T, I>
where
    I: Coordinate,
{
    fn range(&self) -> Range<MatrixAddress<I>> {
        // iteration is row-major, so the last address is the first column of the
        // row after the last row.
        Range {
            start: MatrixAddress {
                column: I::default(),
                row: I::default(),
            },
            end: MatrixAddress {
                column: self.columns,
                row: self.rows,
            },
        }
    }

    fn get(&self, address: MatrixAddress<I>) -> Option<&T> {
        if !self.contains(address) {
            None
        } else {
            let addr = self.index_address(address);
            self.data.get(addr)
        }
    }

    fn get_mut(&mut self, address: MatrixAddress<I>) -> Option<&mut T> {
        if !self.contains(address) {
            None
        } else {
            let addr = self.index_address(address);
            self.data.get_mut(addr)
        }
    }
}

impl<'a, T, I> Index<MatrixAddress<I>> for Matrix<T, I>
where
    I: Coordinate,
{
    type Output = T;

    fn index(&self, index: MatrixAddress<I>) -> &Self::Output {
        match self.get(index) {
            None => panic!("out of range index via Index trait"),
            Some(v) => v,
        }
    }
}

impl<T, I> IndexMut<MatrixAddress<I>> for Matrix<T, I>
where
    I: Coordinate,
{
    fn index_mut(&mut self, index: MatrixAddress<I>) -> &mut T {
        match self.get_mut(index) {
            None => panic!("out of range index via IndexMut trait"),
            Some(v) => v,
        }
    }
}

/// Row is a quality-of-life assistant to ease processing matrices
/// in a row-major fashion.
pub struct Row<'a, T, I>
where
    I: Coordinate,
{
    matrix: &'a Matrix<T, I>,
    row: I,
}

impl <'a, T, I> Row<'a, T, I>
where
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a Matrix<T, I>, row: I) -> Self {
        Row{
            matrix,
            row,
        }
    }

    /// row returns the row number this Row represents,  0-based.
    pub fn row(&self) -> I {
        self.row
    }

    /// iter returns a bi-directional iterator over row.
    pub fn iter(&self) -> MatrixRowIterator<'a, T, I> {
        MatrixRowIterator::new(self.matrix, self.row)
    }

    /// get retrieves a specified column's cell entry from this row.
    pub fn get(&self, column: I) -> Option<&'a T> {
        self.matrix.get(MatrixAddress{row: self.row, column})
    }
}

/// Column is a quality-of-life assistant to ease processing matrices
/// in a column-major fashion.
pub struct Column<'a, T, I>
where
    I: Coordinate,
{
    matrix: &'a Matrix<T, I>,
    column: I,
}

impl <'a, T, I> Column<'a, T, I>
where
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a Matrix<T, I>, column: I) -> Self {
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

pub struct FormatOptions {
    pub column_delimiter: String,
    pub row_delimiter: String,
}

pub struct MatrixParseOptions<T> {
    pub opts: FormatOptions,
    pub element_parser: Box<dyn Fn(&str) -> T>,
}

impl<T> MatrixParseOptions<T> {
    pub fn parse<I>(&self, input: &str) -> Result<Matrix<T, I>>
    where
        I: Coordinate,
    {
        let values: Vec<Vec<&str>> = input
            .split(self.opts.row_delimiter.as_str())
            .map(|row| {
                row.split(self.opts.column_delimiter.as_str())
                    .filter(|string| !string.is_empty())
                    .collect()
            })
            .filter(|row: &Vec<&str>| !row.is_empty())
            .collect();
        let columns: usize = values.first().unwrap().len();
        if values.iter().skip(1).any(|row| row.len() != columns) {
            return Err(Error::new("Row lengths are mismatched".to_string()));
        }
        let rows: I = match values.len().try_into() {
            Ok(v) => v,
            Err(_) => {
                return Err(Error::new(
                    "text input row count overflows index type".to_string(),
                ));
            }
        };
        let ep = &self.element_parser;
        let cols: I = match columns.try_into() {
            Ok(v) => v,
            Err(_) => {
                return Err(Error::new(
                    "text input column count overflows index type".to_string(),
                ));
            }
        };
        Matrix::new(cols, rows, |address: MatrixAddress<I>| {
            let row: usize = match address.row.try_into() {
                Ok(v) => v,
                Err(_) => panic!("row {} cannot convert to usize.", address.row),
            };
            let column: usize = match address.column.try_into() {
                Ok(v) => v,
                Err(_) => panic!("column {} cannot convert to usize.", address.column),
            };
            ep(values[row][column])
        })
    }
}

pub struct MatrixDisplayOptions<T> {
    pub opts: FormatOptions,
    pub element_formatter: Box<dyn Fn(&T) -> String>,
}

impl<T> MatrixDisplayOptions<T> {
    pub fn to_string<I: Coordinate>(&self, matrix: &Matrix<T, I>) -> String {
        let ef = &self.element_formatter;
        matrix
            .indexed_iter()
            .map(|(addr, value)| {
                format!(
                    "{}{}",
                    ef(value),
                    if addr.column == (matrix.columns - I::unit()) {
                        if addr.row != (matrix.rows - I::unit()) {
                            &self.opts.row_delimiter
                        } else {
                            ""
                        }
                    } else {
                        &self.opts.column_delimiter
                    }
                )
            })
            .fold("".to_string(), |a: String, b: String| a + &b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ascii_formatting_options() -> FormatOptions {
        FormatOptions {
            row_delimiter: "\n".to_string(),
            column_delimiter: "".to_string(),
        }
    }

    fn u8addr(row: u8, column: u8) -> MatrixAddress<u8> {
        MatrixAddress { row, column }
    }

    #[test]
    fn parse_matrix() {
        let opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let matrix = opts.parse::<u8>("ABC\nDEF\nGHI").unwrap();
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.columns, 3);
        assert_eq!(matrix[u8addr(0, 0)], "A");
        assert_eq!(matrix[u8addr(0, 1)], "B");
        assert_eq!(matrix[u8addr(0, 2)], "C");
        assert_eq!(matrix[u8addr(1, 0)], "D");
        assert_eq!(matrix[u8addr(1, 1)], "E");
        assert_eq!(matrix[u8addr(1, 2)], "F");
        assert_eq!(matrix[u8addr(2, 0)], "G");
        assert_eq!(matrix[u8addr(2, 1)], "H");
        assert_eq!(matrix[u8addr(2, 2)], "I");
    }

    #[test]
    fn format_matrix() {
        let parse_opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let format_opts = MatrixDisplayOptions::<String> {
            opts: ascii_formatting_options(),
            element_formatter: Box::new(|x| x.to_string()),
        };
        let matrix = parse_opts.parse::<u16>("ABC\nDEF\nGHI").unwrap();
        let got = format_opts.to_string(&matrix);
        assert_eq!(got, "ABC\nDEF\nGHI");
    }

    #[test]
    fn fancy_format_matrix() {
        let parse_opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let format_opts = MatrixDisplayOptions::<String> {
            opts: FormatOptions {
                row_delimiter: "&&".to_string(),
                column_delimiter: "|".to_string(),
            },
            element_formatter: Box::new(|x| format!("{}_", x)),
        };
        let matrix = parse_opts.parse::<u16>("ABC\nDEF\nGHI").unwrap();
        let got = format_opts.to_string(&matrix);
        assert_eq!(got, "A_|B_|C_&&D_|E_|F_&&G_|H_|I_");
    }

    #[test]
    fn parse_mismatched_lengths() {
        let parse_opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let got = parse_opts.parse::<u16>("ABC\nD\nEFG");
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(err, Error::new("Row lengths are mismatched".to_string()));
    }

    #[test]
    fn parse_too_many_rows() {
        let parse_opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let input = "A\n".repeat(128);
        let got = parse_opts.parse::<i8>(input.as_str());
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("text input row count overflows index type".to_string())
        );
    }

    #[test]
    fn parse_too_many_columns() {
        let parse_opts = MatrixParseOptions {
            opts: ascii_formatting_options(),
            element_parser: Box::new(|x| x.to_string()),
        };
        let input = "A".repeat(128);
        let got = parse_opts.parse::<i8>(input.as_str());
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("text input column count overflows index type".to_string())
        );
    }

    #[test]
    fn negative_row_count() {
        let got = Matrix::new(1, -1, Box::new(|x| x));
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("negative Matrix dimensions are not supported".to_string())
        )
    }

    #[test]
    fn negative_column_count() {
        let got = Matrix::new(-1, 0, Box::new(|x| x));
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("negative Matrix dimensions are not supported".to_string())
        )
    }

    #[test]
    fn empty_matrix() {
        let got = Matrix::new(0, 0, Box::new(|x| x)).unwrap();
        assert_eq!(got.rows, 0);
        assert_eq!(got.columns, 0);
    }

    #[test]
    fn empty_column_non_empty_row_matrix() {
        let got = Matrix::new(0, 1, Box::new(|x| x));
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("zero x non-zero Matrix dimensions are not supported".to_string())
        );
    }

    #[test]
    fn empty_row_non_empty_column_matrix() {
        let got = Matrix::new(0, 1, Box::new(|x| x));
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("zero x non-zero Matrix dimensions are not supported".to_string())
        );
    }

    #[test]
    fn dimensions_exceed_memory() {
        let got: Result<Matrix<u32, u32>> = Matrix::new_default(u32::MAX, u32::MAX);
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(err, Error::new("requested Matrix exceeds allocatable vector size".to_string()));
    }

    #[test]
    fn new_default_matrix() {
        let matrix: Matrix<u8, u8> = Matrix::new_default(1, 1).unwrap();
        assert_eq!(matrix.rows, 1);
        assert_eq!(matrix.columns, 1);
        assert_eq!(matrix[u8addr(0, 0)], 0);
    }

    #[test]
    fn row_column_access() {
        let matrix: Matrix<u8, u8> = Matrix::new_default(1, 1).unwrap();
        let row = matrix.row(0).unwrap();
        assert_eq!(row.row(), 0u8);
        let contents: Vec<&u8> = row.iter().collect();
        assert_eq!(contents, vec![&0u8]);
        let value = row.get(0).unwrap();
        assert_eq!(*value, 0u8);
        let missing = row.get(1);
        assert_eq!(missing, None);
    }
}
