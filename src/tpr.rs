//! Native support for modern GROMACS portable run input (TPR/TPX) files.
//!
//! TPR is a versioned XDR format whose layout changes with GROMACS releases.
//! Parsing is delegated to [`minitpr`], which supports TPX versions 103 and
//! newer (GROMACS 5.1+).  Older files, including the v58 fixture shipped with
//! MDAnalysis, are rejected with [`TprError::UnsupportedVersion`].

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Header metadata from a parsed TPR file.
pub type TprHeader = minitpr::TprHeader;
/// Precision used by a parsed TPR file.
pub type TprPrecision = minitpr::Precision;
/// Simulation box vectors from a parsed TPR file.
pub type TprSimBox = minitpr::SimBox;
/// One atom record from a parsed TPR topology.
pub type TprAtom = minitpr::Atom;
/// One bond record from a parsed TPR topology.
pub type TprBond = minitpr::Bond;
/// Parsed atom and bond topology.
pub type TprTopology = minitpr::TprTopology;
/// Conventional data alias for [`TprFile`].
pub type TprData = TprFile;
/// Conventional structure alias for [`TprFile`].
pub type TprStructure = TprFile;

/// A parsed modern GROMACS TPR file.
#[derive(Clone, Debug)]
pub struct TprFile {
    pub header: TprHeader,
    pub system_name: String,
    pub simbox: Option<TprSimBox>,
    pub topology: TprTopology,
    /// The topology's initial coordinates as a one-frame coordinate file.
    pub coordinates: CoordinateFile,
}

impl TprFile {
    /// Parse a TPR file from a filesystem path.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, TprError> {
        let parsed = minitpr::TprFile::parse(path.as_ref()).map_err(TprError::from)?;
        Ok(Self::from_minitpr(parsed))
    }

    /// Alias for [`TprFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TprError> {
        Self::read_file(path)
    }

    /// Parse a TPR file from a byte stream.
    ///
    /// `minitpr` exposes a path-based parser.  The bytes are therefore written
    /// to a process-unique temporary file before parsing and removed whether
    /// parsing succeeds or fails.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, TprError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse TPR bytes held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TprError> {
        let path = temporary_path();
        {
            let mut file = File::create(&path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        let parsed = Self::read_file(&path);
        let _ = std::fs::remove_file(&path);
        parsed
    }

    /// Number of atoms in the topology.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.topology.atoms.len()
    }

    /// Number of bonds in the topology.
    #[must_use]
    pub fn n_bonds(&self) -> usize {
        self.topology.bonds.len()
    }

    /// Number of coordinate frames represented by this TPR.
    #[must_use]
    pub const fn n_frames(&self) -> usize {
        1
    }

    /// Return the initial coordinate frame.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    /// Convert the parsed topology and initial coordinates into a universe.
    pub fn to_universe(&self) -> crate::Result<Universe> {
        if self.topology.atoms.is_empty() {
            return Err(crate::Error::InvalidInput(
                "TPR file contains no atoms".to_owned(),
            ));
        }
        let atoms = self
            .topology
            .atoms
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let mut atom = Atom::new(
                    index,
                    source.atom_name.clone(),
                    source.position.unwrap_or([0.0; 3]),
                );
                atom.mass = source.mass;
                atom.charge = source.charge;
                atom.resid = source.residue_number;
                atom.resname = source.residue_name.clone();
                atom.element = source.element.map(|element| element.symbol().to_owned());
                atom.velocity = source.velocity;
                atom.force = source.force;
                atom
            })
            .collect::<Vec<_>>();
        let mut topology = Topology::new(atoms);
        for source in &self.topology.bonds {
            topology.add_bond(Bond::new(source.atom1, source.atom2));
        }
        let source = self
            .coordinates
            .frames
            .first()
            .ok_or_else(|| crate::Error::InvalidInput("TPR file has no frame".to_owned()))?;
        let mut frame = Frame::new(source.positions.clone());
        frame.velocities = source.velocities.clone();
        frame.forces = source
            .positions
            .iter()
            .enumerate()
            .map(|(index, _)| self.topology.atoms.get(index).and_then(|atom| atom.force))
            .collect::<Option<Vec<_>>>();
        frame.dimensions = source.dimensions;
        Ok(Universe {
            topology,
            trajectory: Trajectory::new(vec![frame]),
        })
    }

    fn from_minitpr(parsed: minitpr::TprFile) -> Self {
        let coordinates = coordinate_file(&parsed);
        Self {
            header: parsed.header,
            system_name: parsed.system_name,
            simbox: parsed.simbox,
            topology: parsed.topology,
            coordinates,
        }
    }
}

/// Read a modern GROMACS TPR file from a path.
pub fn read_tpr(path: impl AsRef<Path>) -> Result<TprFile, TprError> {
    TprFile::read_file(path)
}

impl CoordinateFile {
    /// Read a TPR's initial coordinate frame from a byte stream.
    pub fn read_tpr<R: Read>(reader: R) -> Result<Self, TprError> {
        Ok(TprFile::read(reader)?.coordinates)
    }

    /// Parse TPR bytes and return the initial coordinate frame.
    pub fn from_tpr_bytes(bytes: &[u8]) -> Result<Self, TprError> {
        Ok(TprFile::from_bytes(bytes)?.coordinates)
    }

    /// Read a TPR's initial coordinate frame from a path.
    pub fn read_tpr_file(path: impl AsRef<Path>) -> Result<Self, TprError> {
        Ok(TprFile::read_file(path)?.coordinates)
    }
}

impl Universe {
    /// Construct a universe from a modern GROMACS TPR file.
    pub fn from_tpr(path: impl AsRef<Path>) -> crate::Result<Self> {
        TprFile::read_file(path)?.to_universe()
    }

    /// Construct a universe from parsed TPR data.
    pub fn from_tpr_file(file: TprFile) -> crate::Result<Self> {
        file.to_universe()
    }

    /// Construct a universe from TPR bytes.
    pub fn from_tpr_bytes(bytes: &[u8]) -> crate::Result<Self> {
        TprFile::from_bytes(bytes)?.to_universe()
    }
}

fn coordinate_file(parsed: &minitpr::TprFile) -> CoordinateFile {
    let positions = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.position.unwrap_or([0.0; 3]))
        .collect::<Vec<_>>();
    let velocities = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.velocity)
        .collect::<Option<Vec<_>>>();
    let dimensions = parsed
        .simbox
        .as_ref()
        .map(|simbox| triclinic_box(simbox.simbox));
    let mut frame = CoordinateFrame::new(positions);
    frame.velocities = velocities;
    frame.dimensions = dimensions;
    frame.names = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.atom_name.clone())
        .collect();
    frame.residue_names = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.residue_name.clone())
        .collect();
    frame.residue_ids = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.residue_number)
        .collect();
    frame.atom_ids = parsed
        .topology
        .atoms
        .iter()
        .map(|atom| atom.atom_number.max(1) as usize)
        .collect();
    frame.title = parsed.system_name.clone();
    CoordinateFile::new(vec![frame])
}

fn temporary_path() -> PathBuf {
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("mdanalysis-rs-tpr-{}-{id}.tpr", std::process::id()))
}

/// Errors produced while reading a TPR file.
#[derive(Debug)]
pub enum TprError {
    Io(io::Error),
    UnsupportedVersion(i32),
    Parse(String),
}

impl fmt::Display for TprError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "TPR I/O error: {error}"),
            Self::UnsupportedVersion(version) => write!(
                formatter,
                "unsupported TPR version {version}; modern parser supports versions 103 and newer"
            ),
            Self::Parse(message) => write!(formatter, "TPR parse error: {message}"),
        }
    }
}

impl std::error::Error for TprError {}

impl From<io::Error> for TprError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<minitpr::errors::ParseTprError> for TprError {
    fn from(error: minitpr::errors::ParseTprError) -> Self {
        use minitpr::errors::ParseTprError;
        match error {
            ParseTprError::UnsupportedVersion(version) => Self::UnsupportedVersion(version),
            ParseTprError::CouldNotRead(error) => Self::Io(error),
            other => Self::Parse(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/tprs")
            .join(name)
    }

    #[test]
    fn reads_modern_fixture_and_builds_universe() {
        let file = read_tpr(fixture("2lyz_gmx_2024.tpr")).expect("modern TPR should parse");
        assert_eq!(file.header.tpr_version, 133);
        assert_eq!(file.n_atoms(), 2263);
        assert_eq!(file.n_bonds(), 2186);
        assert_eq!(file.n_frames(), 1);
        assert_eq!(file.system_name, "HEN EGG WHITE LYSOZYME");
        let frame = file.frame(0).expect("initial TPR frame");
        assert_eq!(frame.n_atoms(), 2263);
        assert_eq!(frame.names[0], "N");
        assert_eq!(frame.residue_names[0], "LYSH");
        assert_eq!(frame.residue_ids[0], 1);
        assert!(frame.velocities.is_some());
        let dimensions = frame.dimensions.expect("TPR box");
        assert!((dimensions[0] - 7.91).abs() < 1e-5);
        assert!((dimensions[2] - 3.79).abs() < 1e-5);

        let universe = file.to_universe().expect("TPR universe");
        assert_eq!(universe.n_atoms(), 2263);
        assert_eq!(universe.n_frames(), 1);
        assert_eq!(universe.topology.bonds.len(), 2186);
        assert_eq!(universe.topology.atoms[0].name, "N");
        assert_eq!(universe.topology.atoms[0].resname, "LYSH");
        assert_eq!(
            universe.current_frame().unwrap().positions[0],
            frame.positions[0]
        );
    }

    #[test]
    fn bytes_and_coordinate_constructors_match_path_reader() {
        let bytes = std::fs::read(fixture("2lyz_gmx_2024.tpr")).expect("fixture bytes");
        let from_bytes = TprFile::from_bytes(&bytes).expect("TPR bytes should parse");
        let coordinates = CoordinateFile::from_tpr_bytes(&bytes).expect("TPR coordinates");
        assert_eq!(coordinates, from_bytes.coordinates);
        assert_eq!(coordinates.n_frames(), 1);
        assert_eq!(coordinates.n_atoms(), 2263);
    }

    #[test]
    fn rejects_legacy_v58_fixture_explicitly() {
        let error = read_tpr(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../mdanalysis/testsuite/MDAnalysisTests/data/adk_oplsaa.tpr"),
        )
        .expect_err("legacy TPR should be rejected by modern parser");
        assert!(matches!(error, TprError::UnsupportedVersion(58)));
        assert!(error.to_string().contains("103 and newer"));
    }
}
