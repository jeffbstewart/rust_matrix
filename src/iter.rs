// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use crate::{Coordinate, Matrix};
use crate::column::Column;
use crate::matrix_address::MatrixAddress;
use crate::row::Row;

/// MatrixForwardIterator returns the available addresses in a matrix in
/// row-major format starting at the origin, or upper left (0, 0) address.
pub struct MatrixForwardIterator<I>
    where I: Coordinate
{
    end_exclusive: MatrixAddress<I>,
    cursor: Option<MatrixAddress<I>>
}

impl <I> MatrixForwardIterator<I>
    where I: Coordinate {
    pub(crate) fn new(end_exclusive: MatrixAddress<I>) -> Self {
        if end_exclusive == MatrixAddress::default() {
            MatrixForwardIterator{
                end_exclusive,
                cursor: None,
            }
        } else {
            MatrixForwardIterator{
                end_exclusive,
                cursor: Some(MatrixAddress::default()),
            }
            }
    }
}

impl <I> Iterator for MatrixForwardIterator<I>
    where I: Coordinate {
    type Item = MatrixAddress<I>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.cursor;
        let next = self.cursor;
        match next {
            None => {},
            Some(mut v) => {
                v.column = v.column + I::unit();
                if v.column == self.end_exclusive.column {
                    v.row = v.row + I::unit();
                    if v.row == self.end_exclusive.row {
                        self.cursor = None;
                    } else {
                        v.column = I::default();
                        self.cursor = Some(v)
                    }
                } else {
                    self.cursor = Some(v);
                }
            }
        }
        result
    }
}

/// MatrixValueIterator returns the values in a matrix
/// in row-major order, starting at the upper left origin (0, 0).
pub struct MatrixValueIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    matrix: &'a dyn Matrix<'a, T, I>,
    addrs: MatrixForwardIterator<I>,
}

impl <'a, T, I> MatrixValueIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>) -> Self {
        MatrixValueIterator{
            matrix,
            addrs: matrix.addresses(),
        }
    }
}

impl <'a, T, I> Iterator for MatrixValueIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.addrs.next() {
            None => None,
            Some(addr) => Some(self.matrix.get(addr).unwrap()),
        }
    }
}

/// MatrixForwardIndexedIterator returns (address, value) tuples for
/// a matrix in row-major order, starting at the upper left origin (0,0).
pub struct MatrixForwardIndexedIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    matrix: &'a dyn Matrix<'a, T, I>,
    addrs: MatrixForwardIterator<I>,
}

impl <'a, T, I> MatrixForwardIndexedIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate,
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>) -> Self {
        MatrixForwardIndexedIterator{
            matrix,
            addrs: MatrixForwardIterator::new(MatrixAddress{
                row: matrix.row_count(),
                column: matrix.column_count(),
            }),
        }
    }
}

impl <'a, T, I> Iterator for MatrixForwardIndexedIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate,
{
    type Item = (MatrixAddress<I>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.addrs.next() {
            None => None,
            Some(a) => Some((a, &self.matrix[a]))
        }
    }
}

pub struct MatrixRowIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate {
    matrix: &'a dyn Matrix<'a, T, I>,
    row: I,
    column_cursor_forward: I,
    column_cursor_back: I,
    terminated: bool,
}

impl <'a, T, I> MatrixRowIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>, row: I) -> Self {
        MatrixRowIterator{
            matrix,
            row,
            column_cursor_forward: I::unit() - I::unit(),
            column_cursor_back: matrix.column_count() - I::unit(),
            terminated: matrix.column_count() == I::unit() - I::unit(),
        }
    }
}

impl <'a, T, I> Iterator for MatrixRowIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // because some of the coordinate types can be unsigned,
        // and because we can cause termination in reverse iteration,
        // we can't just look for cursor_forward > cursor_back.
        if self.terminated {
            None
        } else {
            let addr = MatrixAddress{
                row: self.row,
                column: self.column_cursor_forward,
            };
            let result = Some(&self.matrix[addr]);
            if self.column_cursor_forward == self.column_cursor_back {
                self.terminated = true;
            }
            self.column_cursor_forward = self.column_cursor_forward + I::unit();
            result
        }
    }
}

impl <'a, T, I> DoubleEndedIterator for MatrixRowIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.terminated {
            None
        } else {
            let addr = MatrixAddress{
                row: self.row,
                column: self.column_cursor_back,
            };
            let result = Some(&self.matrix[addr]);
            if self.column_cursor_back == self.column_cursor_forward {
                self.terminated = true;
            } else {
                self.column_cursor_back = self.column_cursor_back - I::unit();
            }
            result
        }
    }
}

pub struct MatrixRowsIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate {
    matrix: &'a dyn Matrix<'a, T, I>,
    row_cursor_forward: I,
    row_cursor_back: I,
    terminated: bool,
}

impl <'a, T, I> MatrixRowsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>) -> Self {
        MatrixRowsIterator{
            matrix,
            row_cursor_forward: I::unit() - I::unit(),
            row_cursor_back: matrix.row_count() - I::unit(),
            terminated: matrix.row_count() == I::unit() - I::unit(),
        }
    }
}

impl <'a, T, I> Iterator for MatrixRowsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    type Item = Row<'a, T, I>;

    fn next(&mut self) -> Option<Self::Item> {
        // because some of the coordinate types can be unsigned,
        // and because we can cause termination in reverse iteration,
        // we can't just look for cursor_forward > cursor_back.
        if self.terminated {
            None
        } else {
            let row : Row<T, I> = Row::new(self.matrix, self.row_cursor_forward);
            if self.row_cursor_forward == self.row_cursor_back {
                self.terminated = true;
            }
            self.row_cursor_forward = self.row_cursor_forward + I::unit();
            Some(row)
        }
    }
}

impl <'a, T, I> DoubleEndedIterator for MatrixRowsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.terminated {
            None
        } else {
            let row : Row<T, I> = Row::new(self.matrix, self.row_cursor_back);
            if self.row_cursor_forward == self.row_cursor_back {
                self.terminated = true;
            } else {
                self.row_cursor_back = self.row_cursor_back - I::unit();
            }
            Some(row)
        }
    }
}


pub struct MatrixColumnIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate {
    matrix: &'a dyn Matrix<'a, T, I>,
    column: I,
    row_cursor_forward: I,
    row_cursor_back: I,
    terminated: bool,
}

impl <'a, T, I> MatrixColumnIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>, column: I) -> Self {
        MatrixColumnIterator{
            matrix,
            column,
            row_cursor_forward: I::unit() - I::unit(),
            row_cursor_back: matrix.row_count() - I::unit(),
            terminated: matrix.row_count() == I::unit() - I::unit(),
        }
    }
}

impl <'a, T, I> Iterator for MatrixColumnIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        // because some of the coordinate types can be unsigned,
        // and because we can cause termination in reverse iteration,
        // we can't just look for cursor_forward > cursor_back.
        if self.terminated {
            None
        } else {
            let addr = MatrixAddress{
                row: self.row_cursor_forward,
                column: self.column,
            };
            let result = Some(&self.matrix[addr]);
            if self.row_cursor_forward == self.row_cursor_back {
                self.terminated = true;
            }
            self.row_cursor_forward = self.row_cursor_forward + I::unit();
            result
        }
    }
}

impl <'a, T, I> DoubleEndedIterator for MatrixColumnIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.terminated {
            None
        } else {
            let addr = MatrixAddress{
                row: self.row_cursor_back,
                column: self.column,
            };
            let result = Some(&self.matrix[addr]);
            if self.row_cursor_back == self.row_cursor_forward {
                self.terminated = true;
            } else {
                self.row_cursor_back = self.row_cursor_back - I::unit();
            }
            result
        }
    }
}

pub struct MatrixColumnsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    matrix: &'a dyn Matrix<'a, T, I>,
    column_cursor_forward: I,
    column_cursor_back: I,
    terminated: bool,
}

impl <'a, T, I> MatrixColumnsIterator<'a, T, I>
where
    T: 'static,
    I: Coordinate
{
    pub(crate) fn new(matrix: &'a dyn Matrix<'a, T, I>) -> Self {
        MatrixColumnsIterator{
            matrix,
            column_cursor_forward: I::unit() - I::unit(),
            column_cursor_back: matrix.column_count() - I::unit(),
            terminated: matrix.row_count() == I::unit() - I::unit(),
        }
    }
}

impl <'a, T, I> Iterator for MatrixColumnsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    type Item = Column<'a, T, I>;

    fn next(&mut self) -> Option<Self::Item> {
        // because some of the coordinate types can be unsigned,
        // and because we can cause termination in reverse iteration,
        // we can't just look for cursor_forward > cursor_back.
        if self.terminated {
            None
        } else {
            let column : Column<T, I> = Column::new(self.matrix, self.column_cursor_forward);
            if self.column_cursor_forward == self.column_cursor_back {
                self.terminated = true;
            }
            self.column_cursor_forward = self.column_cursor_forward + I::unit();
            Some(column)
        }
    }
}

impl <'a, T, I> DoubleEndedIterator for MatrixColumnsIterator<'a, T, I>
where
    T: 'a,
    I: Coordinate,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.terminated {
            None
        } else {
            let column : Column<T, I> = Column::new(self.matrix, self.column_cursor_back);
            if self.column_cursor_forward == self.column_cursor_back {
                self.terminated = true;
            } else {
                self.column_cursor_back = self.column_cursor_back - I::unit();
            }
            Some(column)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::factories::new_default_matrix;
    use crate::format::FormatOptions;
    use super::*;

    fn u8addr(row: u8, column: u8) -> MatrixAddress<u8> {
        MatrixAddress{row, column}
    }

    #[test]
    fn iterator_as_expected() {
        let end_exclusive = u8addr(3, 2);
        let iter = MatrixForwardIterator::new(end_exclusive);
        let values: Vec<MatrixAddress<u8>> = iter.collect();
        assert_eq!(values, vec![
            u8addr(0, 0), u8addr(0, 1),
            u8addr(1, 0), u8addr(1, 1),
            u8addr(2, 0), u8addr(2, 1),
        ]);
    }

    #[test]
    fn empty_matrix_iterator_is_empty() {
        let end_exclusive = u8addr(0, 0);
        let iter = MatrixForwardIterator::new(end_exclusive);
        let values: Vec<MatrixAddress<u8>> = iter.collect();
        assert!(values.is_empty());
    }

    #[test]
    fn indexed_iterator_as_expected() {
        let opts = FormatOptions{
            row_delimiter: "|".to_string(),
            column_delimiter: ",".to_string(),
        };
        let matrix = opts.parse_matrix(
            "a,bc,d|d,ef,g",
            |x| x.to_string()).unwrap();
        let mut iter = matrix.indexed_iter();
        let (a1, v1) = (&mut iter).next().unwrap();
        assert_eq!(a1, u8addr(0, 0));
        assert_eq!(v1, "a");
        let (a2, v2) = (&mut iter).next().unwrap();
        assert_eq!(a2, u8addr(0, 1));
        assert_eq!(v2, "bc");
        let (a3, v3) = (&mut iter).next().unwrap();
        assert_eq!(a3, u8addr(0, 2));
        assert_eq!(v3, "d");
        let (a4, v4) = (&mut iter).next().unwrap();
        assert_eq!(a4, u8addr(1, 0));
        assert_eq!(v4, "d");
        let (a5, v5) = (&mut iter).next().unwrap();
        assert_eq!(a5, u8addr(1, 1));
        assert_eq!(v5, "ef");
        let (a6, v6) = (&mut iter).next().unwrap();
        assert_eq!(a6, u8addr(1, 2));
        assert_eq!(v6, "g");
        assert!(iter.next().is_none());
    }

    #[test]
    fn empty_indexed_iterator_as_expected() {
        let matrix = new_default_matrix::<u8, u8>(0, 0).unwrap();
        let mut iter = matrix.indexed_iter();
        assert!((&mut iter).next().is_none());
    }

    fn ascii_parse_opts<'a>() -> FormatOptions {
        FormatOptions{
            row_delimiter: "\n".to_string(),
            column_delimiter: "".to_string(),
        }
    }

    #[test]
    fn row_iterator_forward_only() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let row0 = matrix.row(0).unwrap().iter();
        let values: Vec<&String> = row0.collect();
        assert_eq!(values, vec!["A", "B", "C"]);
    }

    #[test]
    fn row_iterator_reverse_only() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let row0 = matrix.row(0).unwrap().iter().rev();
        let values: Vec<&String> = row0.collect();
        assert_eq!(values, vec!["C", "B", "A"]);
    }

    #[test]
    fn row_iterator_forward_passes_reverse() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut row0 = matrix.row(0).unwrap().iter();
        assert_eq!(row0.next(), Some(&"A".to_string()));
        assert_eq!(row0.next_back(), Some(&"C".to_string()));
        assert_eq!(row0.next(), Some(&"B".to_string()));
        assert_eq!(row0.next(), None);
        assert_eq!(row0.next_back(), None);
    }

    #[test]
    fn row_iterator_reverse_passes_forward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut row1 = matrix.row(1).unwrap().iter();
        assert_eq!(row1.next(), Some(&"D".to_string()));
        assert_eq!(row1.next_back(), Some(&"F".to_string()));
        assert_eq!(row1.next_back(), Some(&"E".to_string()));
        assert_eq!(row1.next_back(), None);
        assert_eq!(row1.next(), None);
    }

    #[test]
    fn rows_iterator_forward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut rows = matrix.rows();
        let row1 = rows.next().unwrap();
        let values1: Vec<&String> = row1.iter().collect();
        assert_eq!(values1, vec!["A", "B", "C"]);
        let row2 = rows.next().unwrap();
        let values2: Vec<&String> = row2.iter().collect();
        assert_eq!(values2, vec!["D", "E", "F"]);
        assert!(rows.next().is_none());
    }

    #[test]
    fn rows_iterator_backward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut rows = matrix.rows().rev();
        let row1 = rows.next().unwrap();
        let values1: Vec<&String> = row1.iter().collect();
        assert_eq!(values1, vec!["D", "E", "F"]);
        let row2 = rows.next().unwrap();
        let values2: Vec<&String> = row2.iter().collect();
        assert_eq!(values2, vec!["A", "B", "C"]);
        assert!(rows.next().is_none());
    }


    #[test]
    fn column_iterator_forward_only() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let column0 = matrix.column(0).unwrap().iter();
        let values: Vec<&String> = column0.collect();
        assert_eq!(values, vec!["A", "D"]);
    }

    #[test]
    fn column_iterator_reverse_only() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let column0 = matrix.column(0).unwrap().iter().rev();
        let values: Vec<&String> = column0.collect();
        assert_eq!(values, vec!["D", "A"]);
    }

    #[test]
    fn column_iterator_forward_passes_reverse() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut column0 = matrix.column(0).unwrap().iter();
        assert_eq!(column0.next(), Some(&"A".to_string()));
        assert_eq!(column0.next_back(), Some(&"D".to_string()));
        assert_eq!(column0.next(), None);
        assert_eq!(column0.next_back(), None);
    }

    #[test]
    fn column_iterator_reverse_passes_forward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut column1 = matrix.column(1).unwrap().iter();
        assert_eq!(column1.next(), Some(&"B".to_string()));
        assert_eq!(column1.next_back(), Some(&"E".to_string()));
        assert_eq!(column1.next_back(), None);
        assert_eq!(column1.next(), None);
    }

    #[test]
    fn columns_iterator_forward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut columns = matrix.columns();
        let column1 = columns.next().unwrap();
        let values1: Vec<&String> = column1.iter().collect();
        assert_eq!(values1, vec!["A", "D"]);
        let column2 = columns.next().unwrap();
        let values2: Vec<&String> = column2.iter().collect();
        assert_eq!(values2, vec!["B", "E"]);
        let column3 = columns.next().unwrap();
        let values3: Vec<&String> = column3.iter().collect();
        assert_eq!(values3, vec!["C", "F"]);
        assert!(columns.next().is_none());
    }

    #[test]
    fn columns_iterator_backward() {
        let opts = ascii_parse_opts();
        let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string()).unwrap();
        let mut columns = matrix.columns().rev();
        let column1 = columns.next().unwrap();
        let values1: Vec<&String> = column1.iter().collect();
        assert_eq!(values1, vec!["C", "F"]);
        let column2 = columns.next().unwrap();
        let values2: Vec<&String> = column2.iter().collect();
        assert_eq!(values2, vec!["B", "E"]);
        let column3 = columns.next().unwrap();
        let values3: Vec<&String> = column3.iter().collect();
        assert_eq!(values3, vec!["A", "D"]);
        assert!(columns.next().is_none());
    }
}

