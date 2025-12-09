use crate::error::{Error, Result};
use crate::factories::new_matrix;
use crate::{Coordinate, Matrix};
use crate::dense_matrix::DenseMatrix;

/// FormatOptions controls the parsing and string formatting of matrices.
pub struct FormatOptions {
    /// This element, which can be the empty string, will be present between each column,
    /// but not at the start or end of each row.
    pub column_delimiter: String,
    /// This element, which must not be the empty string, will delimit the rows of the matrix.
    pub row_delimiter: String,
}

impl Default for FormatOptions {
    fn default() -> Self {
        FormatOptions{
            column_delimiter: "".to_string(),
            row_delimiter: "\n".to_string(),
        }
    }
}

impl FormatOptions {

    /// parse_matrix takes a text representation of a matrix and a converter function and
    /// returns a DenseMatrix representing the same matrix.
    /// The number of parsed entries in each row must be the same.
    pub fn parse_matrix<T, I>(&self, text_matrix: &str, parse_entry: fn(&str) -> T) -> Result<DenseMatrix<T, I>>
    where
        T: 'static,
        I: Coordinate {
        let values: Vec<Vec<&str>> = text_matrix
            .split(self.row_delimiter.as_str())
            .map(|row| {
                row.split(self.column_delimiter.as_str())
                    .filter(|string| !string.is_empty())
                    .collect()
            })
            .filter(|row: &Vec<&str>| !row.is_empty())
            .collect();
        let columns: usize = match values.first() {
            Some(vec) => vec.len(),
            None => return Err(Error::new("empty input cannot be parsed".to_string()))
        };
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
        let folded_values: Vec<T> = values.into_iter()
            .flatten()
            .map(|v| parse_entry(v))
            .collect();
        new_matrix(
            rows,
            folded_values)
    }

    /// Render a matrix to a string.
    pub fn format<'a, 'b: 'a, T, I>(&'a self, matrix: &'b dyn Matrix<'a, T, I>, format_element: fn(&T) -> String) -> String
    where
        T: 'static,
        I: Coordinate,
    {
        matrix
            .indexed_iter()
            .map(|(addr, value)| {
                format!(
                    "{}{}",
                    format_element(value),
                    if addr.column == (matrix.column_count() - I::unit()) {
                        if addr.row != (matrix.row_count() - I::unit()) {
                            self.row_delimiter.as_str()
                        } else {
                            ""
                        }
                    } else {
                        self.column_delimiter.as_str()
                    }
                )
            })
            .fold("".to_string(), |a: String, b: String| a + &b)
    }
}

#[cfg(test)]
mod tests {
    use crate::format::FormatOptions;

    #[test]
    fn parser_does_not_have_to_outlive_matrix() {
        let _ = {
            let opts = FormatOptions::default();
            let matrix = opts.parse_matrix::<String, u8>("ABC\nDEF", |x| x.to_string())
                .unwrap();
            matrix
        };
    }
}
