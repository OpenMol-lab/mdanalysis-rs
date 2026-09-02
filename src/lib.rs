//! A native Rust toolkit for analysing molecular dynamics data.
//!
//! The crate is split into topology and trajectory objects ([`core`]), file
//! formats ([`pdb`]), atom selection ([`selection`]), and numerical routines
//! ([`geometry`]).

pub mod analysis;
pub mod coordinates;
pub mod core;
pub mod geometry;
pub mod mdamath;
pub mod pdb;
pub mod selection;
pub mod transformations;
pub mod units;

pub use coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
pub use coordinates::{read_gro, read_xyz, write_gro, write_xyz};
pub use core::{Atom, AtomGroup, Bond, Frame, Residue, Segment, Topology, Trajectory, Universe};
pub use geometry::{Matrix3, Vec3, center_of_mass, distance, distance_array, rmsd};
pub use mdamath::{angle, box_volume, dihedral, norm, triclinic_box, triclinic_vectors};
pub use pdb::{PdbAtom, PdbError, PdbStructure, read_pdb, write_pdb};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Pdb(PdbError),
    Coordinate(CoordinateError),
    Selection(selection::SelectionError),
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Pdb(error) => write!(f, "PDB error: {error}"),
            Self::Coordinate(error) => write!(f, "coordinate error: {error}"),
            Self::Selection(error) => write!(f, "selection error: {error}"),
            Self::InvalidInput(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<PdbError> for Error {
    fn from(error: PdbError) -> Self {
        Self::Pdb(error)
    }
}

impl From<CoordinateError> for Error {
    fn from(error: CoordinateError) -> Self {
        Self::Coordinate(error)
    }
}

impl From<selection::SelectionError> for Error {
    fn from(error: selection::SelectionError) -> Self {
        Self::Selection(error)
    }
}
