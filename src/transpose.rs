use std::ops::{Index, IndexMut, Range};
use crate::{Coordinate, Matrix, MatrixAddress, MatrixColumnsIterator, MatrixForwardIndexedIterator, MatrixForwardIterator, MatrixRowsIterator, MatrixValueIterator, Tensor};
use crate::column::Column;
use crate::row::Row;

/// TransposedMatrix builds a transposed view over another Matrix.
/// Because IndexMut is a required trait of Matrix, the matrix we
/// construct the transposed view over must be mutable.
pub struct TransposedMatrix<'a, T, I>
where
    I: Coordinate {
    pub(crate) underlay: &'a mut dyn Matrix<'a, T, I>,
}

impl <'a, T, I> Tensor<T, I, MatrixAddress<I>, 2> for TransposedMatrix<'a, T, I>
where
    T: 'static,
    I: Coordinate,
{
    fn range(&self) -> Range<MatrixAddress<I>> {
        let under = self.underlay.range();
        Range{
            start: under.start,
            end: under.end.transpose(),
        }
    }

    fn get(&self, address: MatrixAddress<I>) -> Option<&T> {
        self.underlay.get(address.transpose())
    }

    fn get_mut(&mut self, address: MatrixAddress<I>) -> Option<&mut T> {
        self.underlay.get_mut(address.transpose())
    }
}

impl<'a, T, I> Index<MatrixAddress<I>> for TransposedMatrix<'a, T, I>
where
    I: Coordinate,
{
    type Output = T;

    fn index(&self, address: MatrixAddress<I>) -> &Self::Output {
        self.underlay.index(address.transpose())
    }
}

impl<'a, T, I> IndexMut<MatrixAddress<I>> for TransposedMatrix<'a, T, I>
where
    I: Coordinate,
{
    fn index_mut(&mut self, index: MatrixAddress<I>) -> &mut Self::Output {
        self.underlay.index_mut(index.transpose())
    }
}

impl <'a, T, I> Matrix<'a, T, I> for TransposedMatrix<'a, T, I>
where
    T: 'static,
    I: Coordinate,
{
    fn row_count(&self) -> I {
        self.underlay.column_count()
    }

    fn column_count(&self) -> I {
        self.underlay.row_count()
    }

    fn iter(&'a self) -> MatrixValueIterator<'a, T, I> {
        MatrixValueIterator::new(self)
    }

    fn addresses(&self) -> MatrixForwardIterator<I> {
        MatrixForwardIterator::new(MatrixAddress{
            row: self.row_count(),
            column: self.column_count(),
        })
    }

    fn indexed_iter(&'a self) -> MatrixForwardIndexedIterator<'a, T, I> {
        MatrixForwardIndexedIterator::new(self)
    }

    fn row(&'a self, row_num: I) -> Option<Row<'a, T, I>> {
        if row_num >= (I::unit() - I::unit()) && row_num < self.row_count() {
            Some(Row::new(self, row_num))
        } else {
            None
        }
    }

    fn column(&'a self, col_num: I) -> Option<Column<'a, T, I>> {
        if col_num >= (I::unit() - I::unit()) && col_num < self.column_count() {
            Some(Column::new(self, col_num))
        } else {
            None
        }
    }

    fn rows(&'a self) -> MatrixRowsIterator<'a, T, I> {
        MatrixRowsIterator::new(self)
    }

    fn columns(&'a self) -> MatrixColumnsIterator<'a, T, I> {
        MatrixColumnsIterator::new(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::format::FormatOptions;
    use crate::new_transposed_matrix;
    use super::*;

    fn u8addr(row: u8, column: u8) -> MatrixAddress<u8> {
        MatrixAddress{
            row, column
        }
    }

    #[test]
    fn transpose_format() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        let got = FormatOptions::default()
            .format(&transposed, |x| x.to_string());
        assert_eq!(got, "14\n25\n36");
    }

    #[test]
    fn transpose_accessors() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        assert_eq!(transposed.row_count(), 3);
        assert_eq!(transposed.column_count(), 2);
    }

    #[test]
    fn transpose_addresses() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        assert_eq!(transposed.addresses().collect::<Vec<MatrixAddress<u8>>>(),
                   vec![
                       u8addr(0, 0),
                       u8addr(0, 1),
                       u8addr(1, 0),
                       u8addr(1, 1),
                       u8addr(2, 0),
                       u8addr(2, 1)
                   ]);
    }

    #[test]
    fn transpose_get() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let mut transposed = new_transposed_matrix(&mut base);
        let addr = u8addr(1, 1);
        assert_eq!(transposed[addr], "5");
        assert_eq!(transposed.get(addr).unwrap(), "5");
        transposed[addr] = "3, sir!".to_string();
        assert_eq!(transposed[addr], "3, sir!");
        assert_eq!(transposed.get(addr).unwrap(), "3, sir!");
        *transposed.get_mut(addr).unwrap() = "V".to_string();
        assert_eq!(transposed[addr], "V");
        assert_eq!(transposed.get(addr).unwrap(), "V");
    }

    #[test]
    fn transpose_indexed_iter() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        let got: Vec<String> = transposed.indexed_iter()
            .map(|(addr, value)|
                format!("a={},v={}", addr, value)).collect();
        assert_eq!(got, vec![
            "a=(row=0,col=0),v=1",
            "a=(row=0,col=1),v=4",
            "a=(row=1,col=0),v=2",
            "a=(row=1,col=1),v=5",
            "a=(row=2,col=0),v=3",
            "a=(row=2,col=1),v=6",
        ]);
    }

    #[test]
    fn transpose_row() {
        let mut base = FormatOptions::default()
        .parse_matrix::< String, u8 > ("123\n456", | x | x.to_string())
        .unwrap();
        let transposed = new_transposed_matrix( & mut base);
        assert!(transposed.row(3).is_none());
        let row = transposed.row(1).unwrap();
        let got: Vec<&String> = row.iter().collect();
        assert_eq!(got, vec!["2", "5"]);
    }

    #[test]
    fn transpose_column() {
        let mut base = FormatOptions::default()
            .parse_matrix::< String, u8 > ("123\n456", | x | x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix( & mut base);
        assert!(transposed.column(2).is_none());
        let column = transposed.column(1).unwrap();
        let got: Vec<&String> = column.iter().collect();
        assert_eq!(got, vec!["4", "5", "6"]);
    }

    #[test]
    fn transpose_rows() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        let mut rows = transposed.rows();
        let row0 = rows.next().unwrap();
        let got: Vec<&String> = row0.iter().collect();
        assert_eq!(got, vec!["1", "4"]);
        let row1 = rows.next().unwrap();
        let got1: Vec<&String> = row1.iter().collect();
        assert_eq!(got1, vec!["2", "5"]);
        let row2 = rows.next().unwrap();
        let got2: Vec<&String> = row2.iter().collect();
        assert_eq!(got2, vec!["3", "6"]);
        assert!(rows.next().is_none());
    }

    #[test]
    fn transpose_columns() {
        let mut base = FormatOptions::default()
            .parse_matrix::<String, u8>("123\n456", |x| x.to_string())
            .unwrap();
        let transposed = new_transposed_matrix(&mut base);
        let mut columns = transposed.columns();
        let col0 = columns.next().unwrap();
        let got0: Vec<&String> = col0.iter().collect();
        assert_eq!(got0, vec!["1", "2", "3"]);
        let col1 = columns.next().unwrap();
        let got1: Vec<&String> = col1.iter().collect();
        assert_eq!(got1, vec!["4", "5", "6"]);
        assert!(columns.next().is_none());
    }

}