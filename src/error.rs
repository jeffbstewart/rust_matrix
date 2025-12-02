// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use std::fmt::{Display, Formatter};

/// Error is a simple type to prevent Result<T, String> in our signatures.
#[derive(Debug, Eq, PartialEq)]
pub struct Error {
    msg: String,
}

impl Error {
    pub(crate) fn new(msg: String) -> Error {
        Error { msg }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.msg.as_str())
    }
}

impl std::error::Error for Error {
}

/// Result is a convenience placeholder for methods that use Error and the error result
/// type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fmt() {
        let e = Error::new("hi there".to_string());
        assert_eq!(format!("{}", e), "hi there");
    }
}