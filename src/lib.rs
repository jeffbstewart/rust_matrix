// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

//! The matrix crate provides a generic multidimensional tensor, and
//! the matrix concrete implementation of that type, along with various
//! quality-of-life methods and accessors for the matrix type.  It was
//! initially developed for use implementing solutions for the annual
//! advent-of-code challenges, and was heavily inspired and adapted from
//! https://github.com/Daedelus1/RustTensors
mod iter;
mod matrix_address;
mod matrix;
mod traits;
mod error;

pub use iter::*;
pub use matrix_address::*;
pub use matrix::*;
pub use traits::*;
