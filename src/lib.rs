//! A native Rust toolkit for analysing molecular dynamics data.
//!
//! The crate is split into topology and trajectory objects ([`core`]), file
//! formats ([`pdb`]), atom selection ([`selection`]), and numerical routines
//! ([`geometry`]).

pub mod analysis;
pub mod analysis_algorithms;
pub mod coordinates;
pub mod core;
pub mod dcd;
pub mod distances;
pub mod formats;
pub mod geometry;
pub mod guesser;
pub mod mdamath;
pub mod pdb;
pub mod psf;
pub mod selection;
pub mod topology_groups;
pub mod transformations;
pub mod units;

pub use analysis::{
    Analysis, CenterOfMassAnalysis, MeanSquareDisplacementAnalysis, RmsdAnalysis, RmsfAnalysis,
};
pub use analysis_algorithms::{kabsch_fit, kabsch_rmsd, rmsd_array};
pub use coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
pub use coordinates::{read_gro, read_xyz, write_gro, write_xyz};
pub use core::{Atom, AtomGroup, Bond, Frame, Residue, Segment, Topology, Trajectory, Universe};
pub use dcd::{DcdEndian, DcdError, DcdFile, DcdHeader, DcdWriteOptions, read_dcd, write_dcd};
pub use distances::{
    DistanceError, PairDistances, calc_angle, calc_angles, calc_bond, calc_bonds, calc_dihedral,
    calc_dihedrals, capped_distance, distance_array as coordinate_distance_array,
    self_capped_distance, self_distance_array as coordinate_self_distance_array,
};
pub use formats::{
    FormatAtom, FormatBond, FormatError, Structure, read_crd, read_mol2, read_pqr, write_mol2,
    write_pqr,
};
pub use geometry::{
    Matrix3, Vec3, center_of_geometry, center_of_mass, distance, distance_array, rmsd,
    self_distance_array, weighted_rmsd,
};
pub use guesser::{Guesser, GuesserError, guess_bonds, guess_element, guess_mass};
pub use mdamath::{angle, box_volume, dihedral, norm, triclinic_box, triclinic_vectors};
pub use pdb::{PdbAtom, PdbBond, PdbCryst1, PdbError, PdbStructure, read_pdb, write_pdb};
pub use psf::{PsfAtom, PsfBond, PsfError, PsfStructure, read_psf, write_psf};
pub use topology_groups::{AngleValue, BondLength, DihedralValue, TopologyGroupExt};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Pdb(PdbError),
    Psf(PsfError),
    Dcd(dcd::DcdError),
    Coordinate(CoordinateError),
    Format(formats::FormatError),
    Distance(DistanceError),
    Guesser(guesser::GuesserError),
    Selection(selection::SelectionError),
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Pdb(error) => write!(f, "PDB error: {error}"),
            Self::Psf(error) => write!(f, "PSF error: {error}"),
            Self::Dcd(error) => write!(f, "DCD error: {error}"),
            Self::Coordinate(error) => write!(f, "coordinate error: {error}"),
            Self::Format(error) => write!(f, "format error: {error}"),
            Self::Distance(error) => write!(f, "distance error: {error}"),
            Self::Guesser(error) => write!(f, "guesser error: {error}"),
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

impl From<PsfError> for Error {
    fn from(error: PsfError) -> Self {
        Self::Psf(error)
    }
}

impl From<dcd::DcdError> for Error {
    fn from(error: dcd::DcdError) -> Self {
        Self::Dcd(error)
    }
}

impl From<CoordinateError> for Error {
    fn from(error: CoordinateError) -> Self {
        Self::Coordinate(error)
    }
}

impl From<formats::FormatError> for Error {
    fn from(error: formats::FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<DistanceError> for Error {
    fn from(error: DistanceError) -> Self {
        Self::Distance(error)
    }
}

impl From<guesser::GuesserError> for Error {
    fn from(error: guesser::GuesserError) -> Self {
        Self::Guesser(error)
    }
}

impl From<selection::SelectionError> for Error {
    fn from(error: selection::SelectionError) -> Self {
        Self::Selection(error)
    }
}
