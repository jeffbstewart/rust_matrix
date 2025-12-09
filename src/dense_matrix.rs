// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use crate::iter::{MatrixForwardIndexedIterator, MatrixForwardIterator};
use crate::matrix_address::MatrixAddress;
use crate::traits::{Coordinate, Tensor};
use std::ops::{Index, IndexMut, Range};
use crate::{Matrix, MatrixColumnsIterator, MatrixRowsIterator, MatrixValueIterator};
use crate::column::Column;
use crate::row::Row;

/// DenseMatrix pre-allocates storage for every storage cell.
#[derive(Debug)]
pub struct DenseMatrix<T, I>
where
    I: Coordinate,
{
    columns: I,
    rows: I,
    pub(crate) data: Vec<T>,
}

impl <'a, T, I> DenseMatrix<T, I>
where
    I: Coordinate,
{
    pub(crate) fn new(columns: I, rows: I, data: Vec<T>) -> Self {
        Self { columns, rows, data }
    }

    fn index_address(&self, address: MatrixAddress<I>) -> usize {
        match (address.row * self.columns + address.column).try_into() {
            Ok(v) => v,
            Err(_) => panic!("address overflows usize.  This should be unreachable."),
        }
    }
}

impl<'a, T: 'a, I> Matrix<'a, T, I> for DenseMatrix<T, I>
where
    T: 'static,
    I: Coordinate,
{
    fn row_count(&self) -> I {
        self.rows
    }

    fn column_count(&self) -> I {
        self.columns
    }

    fn addresses(&self) -> MatrixForwardIterator<I> {
        MatrixForwardIterator::new(MatrixAddress {
            column: self.columns,
            row: self.rows,
        })
    }
    
    fn iter(&'a self) -> MatrixValueIterator<'a, T, I> {
        MatrixValueIterator::new(self)
    }

    fn indexed_iter(&self) -> MatrixForwardIndexedIterator<'_, T, I> {
        MatrixForwardIndexedIterator::new(self)
    }


    fn row(&'a self, row_num: I) -> Option<Row<'a, T, I>> {
        if row_num < I::unit() - I::unit() || row_num >= self.rows {
            None
        } else {
            Some(Row::new(self, row_num))
        }
    }

    fn column(&'a self, column_num: I) -> Option<Column<'a, T, I>> {
        if column_num < I::unit() - I::unit() || column_num >= self.columns {
            None
        } else {
            Some(Column::new(self, column_num))
        }
    }

    fn rows(&'a self) -> MatrixRowsIterator<'a, T, I> {
        MatrixRowsIterator::new(self)
    }

    fn columns(&'a self) -> MatrixColumnsIterator<'a, T, I> {
        MatrixColumnsIterator::new(self)
    }
}

impl<'a, T: 'a, I> Tensor<T, I, MatrixAddress<I>, 2> for DenseMatrix<T, I>
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

impl<'a, T, I> Index<MatrixAddress<I>> for DenseMatrix<T, I>
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

impl<T, I> IndexMut<MatrixAddress<I>> for DenseMatrix<T, I>
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

impl<T, I> Clone for DenseMatrix<T, I>
where
    T: Clone,
    I: Coordinate,
{
    fn clone(&self) -> Self {
        DenseMatrix{
            columns: self.columns,
            rows: self.rows,
            data: self.data.clone(),
        }
    }
}

impl<T, I> PartialEq for DenseMatrix<T, I>
where
    T: PartialEq,
    I: Coordinate,
{
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows {
            return false;
        }
        if self.columns != other.columns {
            return false;
        }
        self.data.eq(&other.data)
    }
}

impl <T, I> Eq for DenseMatrix<T, I>
where
    T: Eq,
    I: Coordinate,
{}

#[cfg(test)]
mod tests {
    use std::panic;
    use crate::error::Error;
    use crate::factories::*;
    use crate::format::FormatOptions;
    use crate::traits::MatrixMap;
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
        let opts = ascii_formatting_options();
        let matrix = Box::new(opts.parse_matrix(
            "ABC\nDEF\nGHI",
    |x| x.to_string())
            .unwrap());
        assert_eq!(matrix.row_count(), 3);
        assert_eq!(matrix.column_count(), 3);
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
        let opts = ascii_formatting_options();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF\nGHI", |x| x.to_string()).unwrap();
        let got = opts.format(&matrix, |x| x.to_string());
        assert_eq!(got, "ABC\nDEF\nGHI");
    }

    #[test]
    fn fancy_format_matrix() {
        let opts = ascii_formatting_options();
        let matrix = opts.parse_matrix::<String, u16>("ABC\nDEF\nGHI", |x| x.to_string()).unwrap();
        let opts2 = FormatOptions{
            column_delimiter: "|".to_string(),
            row_delimiter: "&&".to_string(),
        };
        let got = opts2.format(&matrix, |x| format!("{}_", x));
        assert_eq!(got, "A_|B_|C_&&D_|E_|F_&&G_|H_|I_");
    }

    #[test]
    fn parse_without_terminal_line_termination() {
        let opts = ascii_formatting_options();
        let got = opts.parse_matrix::<String, u16>("ABC\nEFG", |x| x.to_string()).unwrap();
        assert_eq!(got.row_count(), 2);
        assert_eq!(got.column_count(), 3);
        let row0 = got.row(0).unwrap();
        let row0v: Vec<String> = row0.iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(row0v, vec!["A", "B", "C"]);
        let row1v: Vec<String> = got.row(1).unwrap().iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(row1v, vec!["E", "F", "G"]);
    }

    #[test]
    fn parse_with_terminal_line_termination() {
        let opts = ascii_formatting_options();
        let got = opts.parse_matrix::<String, u16>("ABC\nEFG\n", |x| x.to_string()).unwrap();
        assert_eq!(got.row_count(), 2);
        assert_eq!(got.column_count(), 3);
        let row0 = got.row(0).unwrap();
        let row0v: Vec<String> = row0.iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(row0v, vec!["A", "B", "C"]);
        let row1v: Vec<String> = got.row(1).unwrap().iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(row1v, vec!["E", "F", "G"]);
    }


    #[test]
    fn parse_mismatched_lengths() {
        let opts = ascii_formatting_options();
        let got = opts.parse_matrix::<String, u16>("ABC\nD\nEFG", |x| x.to_string());
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(err, Error::new("Row lengths are mismatched".to_string()));
    }

    #[test]
    fn parse_too_many_rows() {
        let opts = ascii_formatting_options();
        let input = "A\n".repeat(128);
        let got = opts.parse_matrix::<String, i8>(input.as_str(), |x| x.to_string());
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("text input row count overflows index type".to_string())
        );
    }

    #[test]
    fn parse_too_many_columns() {
        let opts = ascii_formatting_options();
        let input = "A".repeat(128);
        let got = opts.parse_matrix::<String, i8>(input.as_str(), |x| x.to_string());
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("cannot convert columns back to I".to_string())
        );
    }

    #[test]
    fn negative_row_count() {
        let got = new_matrix(-1, vec![23, 5, 2]);
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("negative row count not supported".to_string())
        )
    }

    #[test]
    fn uneven_data_vector_size() {
        let got = new_matrix(2, vec![23, 5, 2]);
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("data length 3 is not a multiple of rows (2)".to_string())
        )
    }

    #[test]
    fn empty_matrix() {
        let data: Vec<u8> = Vec::new();
        let got = new_matrix(0, data).unwrap();
        assert_eq!(got.row_count(), 0);
        assert_eq!(got.column_count(), 0);
    }

    #[test]
    fn empty_column_non_empty_row_matrix() {
        let empty: Vec<u8> = Vec::new();
        let got = new_matrix(1, empty);
        assert!(got.is_err());
        let err = got.err().unwrap();
        assert_eq!(
            err,
            Error::new("missing row data".to_string())
        );
    }

    #[test]
    fn dimensions_exceed_memory() {
        match panic::catch_unwind(|| {
            _ = new_default_matrix::<u32, u32>(u32::MAX, u32::MAX);
            unreachable!("should have panicked(1)");
        }) {
            Ok(_) => unreachable!("should have panicked(2)"),
            Err(_) => {
                // can't tell what the actual error is.  It's not a string.
                // settle for a panic, any panic.
            }
        }
    }

    #[test]
    fn new_default_matrix_test() {
        let matrix = match new_default_matrix::<u8, u8>(1, 1) {
            Ok(g) => Box::new(g),
            Err(e) => panic!("{}", e),
        };
        assert_eq!(matrix.row_count(), 1);
        assert_eq!(matrix.column_count(), 1);
        assert_eq!(matrix[u8addr(0, 0)], 0);
    }

    #[test]
    fn row_column_access() {
        let g = match new_default_matrix::<u8, u8>(1, 1) {
            Ok(res) => res,
            Err(e) => unreachable!("{}", e),
        };
        let row = g.row(0).unwrap();
        assert_eq!(row.row(), 0u8);
        let contents: Vec<&u8> = row.iter().collect();
        assert_eq!(contents, vec![&0u8]);
        let value = row.get(0).unwrap();
        assert_eq!(*value, 0u8);
        let missing = row.get(1);
        assert_eq!(missing, None);
    }

    #[test]
    fn test_map_matrix() {
        let m = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |v| v.to_string())
            .unwrap();
        let mapper = |v: &String| v.parse::<u8>().unwrap();
        let t = Box::new(m.map_matrix(&mapper));
        let row0_values = t.row(0u8).unwrap().iter()
            .map(|v|*v)
            .collect::<Vec<u8>>();
        assert_eq!(row0_values, vec!(1u8, 2u8, 3u8));
    }

    #[test]
    fn test_indexed_map_matrix() {
        let m = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |v| v.to_string())
            .unwrap();
        let mut x = |addr: MatrixAddress<u8>, v: &String| {
            let n: u64 = v.parse().unwrap();
            let coord = 10 * addr.column + addr.row;
            n + coord as u64
        };
        let t = m.map_indexed_matrix(&mut x);
        let row0_values = t.row(0u8).unwrap().iter()
            .map(|v|*v)
            .collect::<Vec<u64>>();
        assert_eq!(row0_values, vec!(1u64, 12u64, 23u64));
        let row1_values = t.row(1u8).unwrap().iter()
            .map(|v| *v)
            .collect::<Vec<u64>>();
        assert_eq!(row1_values, vec!(5u64, 16u64, 27u64));
    }
}
