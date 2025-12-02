// Copyright 2025 Jeffrey B. Stewart <jeff@stewart.net>.  All Rights Reserved.

use std::fmt::{Debug, Display};
use std::ops::{Add, Index, IndexMut, Mul, Range, Sub};

/// Dimension is an axis of the storage.  In a vector there's a single Dimension (0)
/// and it's the horizontal position within the vector.  For a matrix, there are two
/// Dimensions, x and y (0 and 1, but whether 0 is x or 1 is an implementation detail).
pub type Dimension = usize;

/// Addresses must be convertible to and from a slice containing one V for each of the
/// (fixed number) of dimensions for that Tensor.
pub trait Address<V, const DIMENSION: usize>:
    Clone
    + Debug
    + From<[V; DIMENSION]>
    // Note: instead of implementing Into<V; DIMENSION> for T, implement
    // FROM<T> for [V; DIMENSION] instead.
    + Into<[V; DIMENSION]>
    + Index<Dimension, Output = V>
{
}

/// CheckedMul is a trait to capture the built-in checked_mul behavior
/// provided for all intrinsic integer types in Rust, casting to usize.
/// This is intended for computing matrix bounds for storage in a
/// contiguous vector, whose upper-bound size is limited by usize.
pub trait CheckedMul where Self: Sized {
    // returns None on overflow.
    fn checked_multiply(&self, rhs: Self) -> Option<usize>;
}

/// Coordinate is the traits required to act as a dimensional index in
/// a Matrix.  All the built-in integral types should satisfy these.
pub trait Coordinate:
    Add<Output = Self>
    + CheckedMul
    + Clone
    + Copy
    + Debug
    + Default
    + Display
    + TryInto<usize>
    + TryFrom<usize>
    + Mul<Output = Self>
    + PartialOrd
    + Sub<Output = Self>
    + Unit
{
}

/// Tensor is a generic multidimensional data store trait.  Think of it as a shared
/// interface for a vector, a matrix, a cube, and a hypercube.
pub trait Tensor<'a,
    T,
    V: Copy + Unit + Add<Output = V> + Sub<Output = V> + PartialOrd,
    A: Address<V, DIMENSION>,
    const DIMENSION: usize,
>: IndexMut<A, Output = T>
{
    /// range provides the bounds of the address space for the Tensor.
    /// The lower (inclusive bound) is the origin, conceptually placed at the left of
    /// a vector, the upper left of a matrix, and so on.
    ///  That lower bound is conventionally zero-based, but does not
    /// have to be.  The upper bound (exclusive) is the right side of the vector,
    /// the lower right of the matrix, etc.  Be aware that while Range provides
    /// iterator functionality, once you move beyond single-dimension Tensors,
    /// that iterator does not provide the correct iteration of available
    /// addresses.
    fn range(&self) -> Range<A>;

    /// contains is true if the given address is within the Tensor's bounds
    /// for all dimensions.
    fn contains(&self, address: A) -> bool {
        let range = self.range();
        (0..DIMENSION).all(|d| {
            let v = address[d];
            v >= range.start[d] && v < range.end[d]
        })
    }

    /// An out-of-range-safe version of the Index trait.
    fn get(&self, address: A) -> Option<&T>;

    /// An out-of-range-safe version of the IndexMut trait.
    fn get_mut(&mut self, address: A) -> Option<&mut T>;
}

/// Unit returns the natural "one" value for a given type.
/// This in turn is used to increment and decrement values within a range to
/// provide an iterator.
pub trait Unit {
    fn unit() -> Self;
}

//noinspection DuplicatedCode
/// blanket implementation of Coordinate for all eligible types.
impl<T> Coordinate for T where
    T: Add<Output = Self>
        + CheckedMul
        + Clone
        + Copy
        + Debug
        + Default
        + Display
        + TryInto<usize>
        + TryFrom<usize>
        + Mul<Output = Self>
        + PartialOrd
        + Sub<Output = Self>
        + Unit
{
}

// I had a blanket implementation of Unit for anything that converted from u8, but that didn't
// work for i8, and adding an i8 explicit implementation complains that something might add a
// From<u8> for i8 in the future.  Unlikely, but let's just enumerate the built ins here.

impl Unit for i8 {
    fn unit() -> Self {
        1
    }
}

impl Unit for u8 {
    fn unit() -> Self {
        1
    }
}

impl Unit for i16 {
    fn unit() -> Self {
        1
    }
}

impl Unit for u16 {
    fn unit() -> Self {
        1
    }
}

impl Unit for i32 {
    fn unit() -> Self {
        1
    }
}

impl Unit for u32 {
    fn unit() -> Self {
        1
    }
}

impl Unit for i64 {
    fn unit() -> Self {
        1
    }
}

impl Unit for u64 {
    fn unit() -> Self {
        1
    }
}

impl Unit for i128 {
    fn unit() -> Self {
        1
    }
}

impl Unit for u128 {
    fn unit() -> Self {
        1
    }
}

impl Unit for char {
    fn unit() -> Self {
        1 as char
    }
}

struct Internals{}

impl Internals {
    fn checked_multiply_unsigned(lhs: u64, rhs: u64) -> Option<usize> {
        let product = lhs.checked_mul(rhs)?;
        let us: usize = match product.try_into() {
            Ok(v) => v,
            Err(_) => return None,
        };
        Some(us)
    }

    fn checked_multiply_signed(lhs: i64, rhs: i64) -> Option<usize> {
        if lhs < 0 || rhs < 0 {
            return None
        }
        let product = lhs.checked_mul(rhs)?;
        let us: usize = match product.try_into() {
            Ok(v) => v,
            Err(_) => return None,
        };
        Some(us)
    }
}

impl CheckedMul for u8 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_unsigned(*self as u64, rhs as u64)
    }
}

impl CheckedMul for u16 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_unsigned(*self as u64, rhs as u64)
    }
}

impl CheckedMul for u32 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_unsigned(*self as u64, rhs as u64)
    }
}

impl CheckedMul for u64 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_unsigned(*self, rhs)
    }
}

impl CheckedMul for i8 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_signed(*self as i64, rhs as i64)
    }
}

impl CheckedMul for i16 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_signed(*self as i64, rhs as i64)
    }
}

impl CheckedMul for i32 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_signed(*self as i64, rhs as i64)
    }
}

impl CheckedMul for i64 {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_signed(*self, rhs)
    }
}

impl CheckedMul for char {
    fn checked_multiply(&self, rhs: Self) -> Option<usize> {
        Internals::checked_multiply_unsigned(*self as u64, rhs as u64)
    }
}







