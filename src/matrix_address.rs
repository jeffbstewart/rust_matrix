// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use crate::LogicalDimension::{Column, Row};
use crate::traits::{Address, Coordinate, Dimension};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Index, Sub};

/// MatrixAddress references a cell in a matrix by its row and column.
/// Rows are numbered from zero at the top, and columns are numbered
/// from zero at the left.  Thus (0, 0), the origin, is positioned
/// at the upper-left of the matrix.  Any built in numeric or character
/// type that fits in usize can be used as the index (thus in practice
/// up to i16 / u16).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct MatrixAddress<I>
where
    I: Coordinate,
{
    pub column: I,
    pub row: I,
}

/// LogicalDimension lets you refer to the address dimensions of a matrix
/// conceptually, rather than numerically.
pub enum LogicalDimension {
    Column,
    Row,
}

impl From<LogicalDimension> for Dimension {
    fn from(value: LogicalDimension) -> Self {
        match value {
            Column => 0,
            Row => 1,
        }
    }
}

// This is an alternative way to extract a row or a column from a MatrixAddress.
impl<I> Index<LogicalDimension> for MatrixAddress<I>
where
    I: Coordinate,
{
    type Output = I;

    fn index(&self, dimension: LogicalDimension) -> &Self::Output {
        match dimension {
            Row => &self.row,
            Column => &self.column,
        }
    }
}

impl<I> Index<Dimension> for MatrixAddress<I>
where
    I: Coordinate,
{
    type Output = I;

    fn index(&self, index: Dimension) -> &Self::Output {
        match index {
            0 => &self.column,
            1 => &self.row,
            _ => panic!("invalid dimension"),
        }
    }
}

impl<I> Address<I, 2usize> for MatrixAddress<I> where I: Coordinate {}

impl<I> Display for MatrixAddress<I>
where
    I: Coordinate,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("({},{})", self.column, self.row))
    }
}

impl<I> From<[I; 2]> for MatrixAddress<I>
where
    I: Coordinate,
{
    fn from(value: [I; 2]) -> Self {
        Self {
            column: value[0],
            row: value[1],
        }
    }
}

impl <I> From<MatrixAddress<I>> for [I; 2]
where
    I: Coordinate,{
    fn from(value: MatrixAddress<I>) -> Self {
        [value.column, value.row]
    }
}

impl<I> Add for MatrixAddress<I>
where
    I: Coordinate,
{
    type Output = MatrixAddress<I>;

    fn add(self, rhs: Self) -> Self::Output {
        MatrixAddress {
            // Warning: result can be out of bounds
            column: self.column + rhs.column,
            row: self.row + rhs.row,
        }
    }
}

impl<I> Sub for MatrixAddress<I>
where
    I: Coordinate,
{
    type Output = MatrixAddress<I>;

    fn sub(self, rhs: Self) -> Self::Output {
        // Warning: result can be out of bounds
        MatrixAddress {
            column: self.column - rhs.column,
            row: self.row - rhs.row,
        }
    }
}

impl<I> Default for MatrixAddress<I>
where
    I: Coordinate,
{
    fn default() -> Self {
        MatrixAddress {
            column: I::default(),
            row: I::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u8addr(row: u8, column: u8) -> MatrixAddress<u8> {
        MatrixAddress { row, column }
    }

    #[test]
    fn test_display() {
        let got = u8addr(5, 23).to_string();
        assert_eq!(got, "(23,5)")
    }

    #[test]
    fn test_dimensions() {
        let addr = u8addr(5, 23);
        assert_eq!(addr.row, 5u8);
        assert_eq!(addr.column, 23u8);
        assert_eq!(addr[Row], 5u8);
        assert_eq!(addr[Column], 23u8);
        assert_eq!(addr[0], 23u8);
        assert_eq!(addr[1], 5u8);
        let index: Dimension = Row.into();
        assert_eq!(index, 1usize);
        let index2: Dimension = Column.into();
        assert_eq!(index2, 0usize);
    }

    #[test]
    fn test_into_u8_2() {
        let addr = u8addr(5, 23);
        let want: [u8; 2] = [23, 5];
        let got: [u8; 2] = addr.into();
        assert_eq!(want, got);
    }

    #[test]
    fn test_into_matrix_address() {
        let rep: [u8; 2] = [5, 23];
        let got: MatrixAddress<u8> = rep.into();
        let want = u8addr(23, 5);
        assert_eq!(got, want);
    }

    #[test]
    fn test_invalid_dimension() {
        let rep = u8addr(5, 23);
        match std::panic::catch_unwind(|| rep[2]) {
            Ok(_) => unreachable!("should have panicked"),
            Err(e) => {
                if let Some(s) = e.downcast_ref::<&'static str>() {
                    assert_eq!(s.to_string(), "invalid dimension");
                } else {
                    unreachable!("wrong panic type");
                }
            },
        };
    }

    #[test]
    fn test_add() {
        let a = u8addr(1, 2);
        let b = u8addr(3, 4);
        let c = a + b;
        assert_eq!(c, u8addr(4, 6))
    }

    #[test]
    fn test_sub() {
        let a = u8addr(3, 4);
        let b = u8addr(1, 3);
        let c = a - b;
        assert_eq!(c, u8addr(2, 1));
    }

    #[test]
    fn test_default() {
        let def: MatrixAddress<u8> = MatrixAddress::default();
        assert_eq!(def, u8addr(0, 0));
    }
}
