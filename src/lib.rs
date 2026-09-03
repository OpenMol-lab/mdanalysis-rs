//! A native Rust toolkit for analysing molecular dynamics data.
//!
//! The crate is split into topology and trajectory objects ([`core`]), file
//! formats ([`pdb`]), atom selection ([`selection`]), and numerical routines
//! ([`geometry`]).

pub mod amber;
pub mod analysis;
pub mod analysis_algorithms;
pub mod coordinates;
pub mod core;
pub mod correlations;
pub mod dcd;
pub mod distances;
pub mod dlpoly;
pub mod dms;
pub mod fhiaims;
pub mod formats;
pub mod geometry;
pub mod gms;
pub mod gsd;
pub mod guesser;
pub mod h5md;
pub mod hoomdxml;
mod io_utils;
pub mod itp;
pub mod lammps;
pub mod mdamath;
pub mod mmtf;
pub mod neighbor_search;
pub mod netcdf;
pub mod pdb;
pub mod pdbqt;
pub mod psf;
pub mod selection;
pub mod tng;
pub mod topology_groups;
pub mod tpr;
pub mod transformations;
pub mod trc;
pub mod trj;
pub mod trz;
pub mod txyz;
pub mod units;
pub mod xdr;

pub use amber::{
    AmberAngle, AmberBond, AmberDihedral, AmberError, AmberImproper, AmberTopAtom, AmberTopFile,
    AmberTopology, InpcrdFile, NamdBinFile, PrmtopFile, read_amber_top, read_inpcrd, read_namdbin,
    read_prmtop, read_top, write_inpcrd, write_namdbin,
};
pub use analysis::{
    Analysis, CenterOfMassAnalysis, MeanSquareDisplacementAnalysis, RmsdAnalysis, RmsfAnalysis,
};
pub use analysis_algorithms::{kabsch_fit, kabsch_rmsd, rmsd_array};
pub use coordinates::{CoordinateError, CoordinateFile, CoordinateFrame};
pub use coordinates::{read_gro, read_xyz, write_gro, write_xyz};
pub use core::{Atom, AtomGroup, Bond, Frame, Residue, Segment, Topology, Trajectory, Universe};
pub use correlations::{
    AutocorrelationResult, CorrelationError, autocorrelation, correct_intermittency,
};
pub use dcd::{DcdEndian, DcdError, DcdFile, DcdHeader, DcdWriteOptions, read_dcd, write_dcd};
pub use distances::{
    DistanceError, PairDistances, apply_pbc, calc_angle, calc_angles, calc_bond, calc_bonds,
    calc_dihedral, calc_dihedrals, capped_distance, distance_array as coordinate_distance_array,
    minimize_vectors, minimum_image_triclinic, self_capped_distance,
    self_distance_array as coordinate_self_distance_array, transform_r_to_s, transform_s_to_r,
};
pub use dlpoly::{
    ConfigFile, DlpolyConfig, DlpolyError, DlpolyFrame, DlpolyHistory, HistoryFile, read_config,
    read_history, write_config, write_config_file, write_history, write_history_file,
};
pub use dms::{DmsBond, DmsError, DmsFile, DmsParticle, read_dms};
pub use fhiaims::{
    FhiaimsData, FhiaimsError, FhiaimsFile, FhiaimsStructure, read_fhiaims, write_fhiaims,
    write_fhiaims_file,
};
pub use formats::{
    FormatAtom, FormatBond, FormatError, Structure, read_crd, read_mol2, read_pqr, write_crd,
    write_mol2, write_pqr,
};
pub use geometry::{
    Matrix3, Vec3, center_of_geometry, center_of_mass, distance, distance_array, rmsd,
    self_distance_array, weighted_rmsd,
};
pub use gms::{GmsAtom, GmsData, GmsError, GmsFile, GmsParser, GmsReader, GmsStructure, read_gms};
pub use gsd::{
    GsdAngle, GsdAtom, GsdBond, GsdData, GsdDihedral, GsdError, GsdFile, GsdFrame, GsdImproper,
    GsdParticle, GsdStructure, read_gsd,
};
pub use guesser::{Guesser, GuesserError, guess_bonds, guess_element, guess_mass};
pub use h5md::{H5mdData, H5mdError, H5mdFile, H5mdStructure, read_h5md};
pub use hoomdxml::{
    HoomdXmlAngle, HoomdXmlAtom, HoomdXmlBond, HoomdXmlBox, HoomdXmlData, HoomdXmlDihedral,
    HoomdXmlError, HoomdXmlFile, HoomdXmlImproper, HoomdXmlStructure, read_hoomdxml,
};
pub use itp::{
    ItpAngle, ItpAtom, ItpAtomType, ItpBond, ItpData, ItpDihedral, ItpError, ItpImproper,
    ItpMoleculeCount, ItpMoleculeType, ItpOptions, ItpSettle, ItpStructure, read_itp,
    read_itp_file, read_itp_with_options,
};
pub use lammps::{
    LammpsAtom, LammpsBond, LammpsBox, LammpsCoordinateConvention, LammpsData, LammpsDataFile,
    LammpsDumpData, LammpsDumpFile, LammpsDumpFrame, LammpsDumpOptions, LammpsDumpReader,
    LammpsError, read_lammps_data, read_lammps_dump, read_lammps_dump_with_options,
    write_lammps_data,
};
pub use mdamath::{angle, box_volume, dihedral, norm, triclinic_box, triclinic_vectors};
pub use mmtf::{MmtfBond, MmtfError, MmtfFile, MmtfGroup, read_mmtf};
pub use neighbor_search::{
    AtomNeighborSearch, NeighborPairs, NeighborSearch, NeighborSearchError, PeriodicKDTree,
    SearchLevel,
};
pub use netcdf::{NetcdfData, NetcdfError, NetcdfFile, NetcdfStructure, read_netcdf};
pub use pdb::{PdbAtom, PdbBond, PdbCryst1, PdbError, PdbStructure, read_pdb, write_pdb};
pub use pdbqt::{PdbqtAtom, PdbqtError, PdbqtStructure, read_pdbqt, write_pdbqt};
pub use psf::{PsfAtom, PsfBond, PsfError, PsfStructure, read_psf, write_psf};
pub use tng::{TngData, TngError, TngFile, TngStructure, read_tng};
pub use topology_groups::{AngleValue, BondLength, DihedralValue, TopologyGroupExt};
pub use tpr::{
    TprAtom, TprBond, TprData, TprError, TprFile, TprHeader, TprPrecision, TprSimBox, TprStructure,
    TprTopology, read_tpr,
};
pub use trc::{TrcData, TrcError, TrcFile, TrcStructure, read_trc, read_trc_files};
pub use trj::{MdcrdFile, TrjData, TrjError, TrjFile, TrjStructure, read_trj, read_trj_with_dt};
pub use trz::{TrzError, TrzFile, TrzHeader, TrzWriteOptions, read_trz, write_trz};
pub use txyz::{
    ArcFile, TxyzAtom, TxyzBond, TxyzData, TxyzError, TxyzFile, TxyzStructure, read_arc, read_txyz,
    write_arc, write_txyz,
};
pub use xdr::{
    TrrFile, TrrPrecision, TrrWriteOptions, XdrError, XtcFile, XtcWriteOptions, read_trr, read_xtc,
    write_trr, write_xtc,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Amber(amber::AmberError),
    Xdr(xdr::XdrError),
    Pdb(PdbError),
    Pdbqt(pdbqt::PdbqtError),
    Lammps(lammps::LammpsError),
    Txyz(txyz::TxyzError),
    Trz(trz::TrzError),
    Psf(PsfError),
    Dcd(dcd::DcdError),
    Dms(dms::DmsError),
    Dlpoly(dlpoly::DlpolyError),
    Coordinate(CoordinateError),
    Format(formats::FormatError),
    Fhiaims(fhiaims::FhiaimsError),
    Gms(gms::GmsError),
    Gsd(gsd::GsdError),
    Trc(trc::TrcError),
    Tpr(tpr::TprError),
    Netcdf(netcdf::NetcdfError),
    Trj(trj::TrjError),
    Tng(tng::TngError),
    H5md(h5md::H5mdError),
    Itp(itp::ItpError),
    HoomdXml(hoomdxml::HoomdXmlError),
    Distance(DistanceError),
    Guesser(guesser::GuesserError),
    Selection(selection::SelectionError),
    InvalidInput(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Amber(error) => write!(f, "Amber/NAMD coordinate error: {error}"),
            Self::Xdr(error) => write!(f, "XDR trajectory error: {error}"),
            Self::Pdb(error) => write!(f, "PDB error: {error}"),
            Self::Pdbqt(error) => write!(f, "PDBQT error: {error}"),
            Self::Lammps(error) => write!(f, "LAMMPS DATA error: {error}"),
            Self::Txyz(error) => write!(f, "Tinker XYZ error: {error}"),
            Self::Trz(error) => write!(f, "TRZ trajectory error: {error}"),
            Self::Psf(error) => write!(f, "PSF error: {error}"),
            Self::Dcd(error) => write!(f, "DCD error: {error}"),
            Self::Dms(error) => write!(f, "DMS error: {error}"),
            Self::Dlpoly(error) => write!(f, "DL_POLY error: {error}"),
            Self::Coordinate(error) => write!(f, "coordinate error: {error}"),
            Self::Format(error) => write!(f, "format error: {error}"),
            Self::Fhiaims(error) => write!(f, "FHI-AIMS error: {error}"),
            Self::Gms(error) => write!(f, "GMS error: {error}"),
            Self::Gsd(error) => write!(f, "GSD error: {error}"),
            Self::Trc(error) => write!(f, "TRC trajectory error: {error}"),
            Self::Tpr(error) => write!(f, "TPR topology error: {error}"),
            Self::Netcdf(error) => write!(f, "NetCDF trajectory error: {error}"),
            Self::Trj(error) => write!(f, "Amber TRJ trajectory error: {error}"),
            Self::Tng(error) => write!(f, "TNG trajectory error: {error}"),
            Self::H5md(error) => write!(f, "H5MD trajectory error: {error}"),
            Self::Itp(error) => write!(f, "ITP error: {error}"),
            Self::HoomdXml(error) => write!(f, "HOOMD XML error: {error}"),
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

impl From<amber::AmberError> for Error {
    fn from(error: amber::AmberError) -> Self {
        Self::Amber(error)
    }
}

impl From<xdr::XdrError> for Error {
    fn from(error: xdr::XdrError) -> Self {
        Self::Xdr(error)
    }
}

impl From<PdbError> for Error {
    fn from(error: PdbError) -> Self {
        Self::Pdb(error)
    }
}

impl From<pdbqt::PdbqtError> for Error {
    fn from(error: pdbqt::PdbqtError) -> Self {
        Self::Pdbqt(error)
    }
}

impl From<lammps::LammpsError> for Error {
    fn from(error: lammps::LammpsError) -> Self {
        Self::Lammps(error)
    }
}

impl From<txyz::TxyzError> for Error {
    fn from(error: txyz::TxyzError) -> Self {
        Self::Txyz(error)
    }
}

impl From<trz::TrzError> for Error {
    fn from(error: trz::TrzError) -> Self {
        Self::Trz(error)
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

impl From<dms::DmsError> for Error {
    fn from(error: dms::DmsError) -> Self {
        Self::Dms(error)
    }
}

impl From<dlpoly::DlpolyError> for Error {
    fn from(error: dlpoly::DlpolyError) -> Self {
        Self::Dlpoly(error)
    }
}

impl From<CoordinateError> for Error {
    fn from(error: CoordinateError) -> Self {
        Self::Coordinate(error)
    }
}

impl From<tpr::TprError> for Error {
    fn from(error: tpr::TprError) -> Self {
        Self::Tpr(error)
    }
}

impl From<netcdf::NetcdfError> for Error {
    fn from(error: netcdf::NetcdfError) -> Self {
        Self::Netcdf(error)
    }
}

impl From<trj::TrjError> for Error {
    fn from(error: trj::TrjError) -> Self {
        Self::Trj(error)
    }
}

impl From<tng::TngError> for Error {
    fn from(error: tng::TngError) -> Self {
        Self::Tng(error)
    }
}

impl From<h5md::H5mdError> for Error {
    fn from(error: h5md::H5mdError) -> Self {
        Self::H5md(error)
    }
}

impl From<formats::FormatError> for Error {
    fn from(error: formats::FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<fhiaims::FhiaimsError> for Error {
    fn from(error: fhiaims::FhiaimsError) -> Self {
        Self::Fhiaims(error)
    }
}

impl From<gms::GmsError> for Error {
    fn from(error: gms::GmsError) -> Self {
        Self::Gms(error)
    }
}

impl From<gsd::GsdError> for Error {
    fn from(error: gsd::GsdError) -> Self {
        Self::Gsd(error)
    }
}

impl From<trc::TrcError> for Error {
    fn from(error: trc::TrcError) -> Self {
        Self::Trc(error)
    }
}

impl From<itp::ItpError> for Error {
    fn from(error: itp::ItpError) -> Self {
        Self::Itp(error)
    }
}

impl From<hoomdxml::HoomdXmlError> for Error {
    fn from(error: hoomdxml::HoomdXmlError) -> Self {
        Self::HoomdXml(error)
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
