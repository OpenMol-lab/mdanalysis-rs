//! Amber ASCII trajectory (TRJ/MDCRD) support.
//!
//! Amber coordinate trajectories do not store their atom count, so callers
//! must provide `n_atoms`.  Frames contain fixed-width coordinate values and
//! may be followed by three unit-cell lengths; the latter are detected from
//! the first frame and represented as an orthorhombic box.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Frame, Trajectory, Universe};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// Parsed Amber ASCII trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct TrjFile {
    pub title: String,
    pub n_atoms: usize,
    pub periodic: bool,
    pub dt: f64,
    pub coordinates: CoordinateFile,
}

pub type TrjData = TrjFile;
pub type TrjStructure = TrjFile;
pub type MdcrdFile = TrjFile;

/// Errors produced while reading an Amber ASCII trajectory.
#[derive(Debug)]
pub enum TrjError {
    Io(io::Error),
    InvalidInput(String),
    Parse { line: usize, message: String },
}

impl fmt::Display for TrjError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "Amber TRJ I/O error: {error}"),
            Self::InvalidInput(message) => write!(formatter, "invalid Amber TRJ input: {message}"),
            Self::Parse { line, message } => {
                write!(formatter, "Amber TRJ parse error on line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for TrjError {}

impl From<io::Error> for TrjError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl TrjFile {
    /// Read a trajectory from a path with a one-ps frame interval.
    pub fn read_file(path: impl AsRef<Path>, n_atoms: usize) -> Result<Self, TrjError> {
        Self::read_with_dt(File::open(path)?, n_atoms, 1.0)
    }

    /// Read a trajectory from a path with an explicit frame interval in ps.
    pub fn read_file_with_dt(
        path: impl AsRef<Path>,
        n_atoms: usize,
        dt: f64,
    ) -> Result<Self, TrjError> {
        Self::read_with_dt(File::open(path)?, n_atoms, dt)
    }

    /// Alias for [`TrjFile::read_file`].
    pub fn open(path: impl AsRef<Path>, n_atoms: usize) -> Result<Self, TrjError> {
        Self::read_file(path, n_atoms)
    }

    /// Parse a trajectory from a reader with a one-ps frame interval.
    pub fn read<R: Read>(reader: R, n_atoms: usize) -> Result<Self, TrjError> {
        Self::read_with_dt(reader, n_atoms, 1.0)
    }

    /// Parse a trajectory from a reader with an explicit frame interval.
    pub fn read_with_dt<R: Read>(mut reader: R, n_atoms: usize, dt: f64) -> Result<Self, TrjError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes_with_dt(&bytes, n_atoms, dt)
    }

    /// Parse bytes with a one-ps frame interval. Gzip and bzip2 streams are
    /// transparently decompressed.
    pub fn from_bytes(bytes: &[u8], n_atoms: usize) -> Result<Self, TrjError> {
        Self::from_bytes_with_dt(bytes, n_atoms, 1.0)
    }

    /// Parse bytes with an explicit frame interval in ps.
    pub fn from_bytes_with_dt(bytes: &[u8], n_atoms: usize, dt: f64) -> Result<Self, TrjError> {
        if n_atoms == 0 {
            return Err(TrjError::InvalidInput(
                "n_atoms must be greater than zero".to_owned(),
            ));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(TrjError::InvalidInput(
                "dt must be finite and non-negative".to_owned(),
            ));
        }
        let decoded = if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut output = Vec::new();
            GzDecoder::new(Cursor::new(bytes)).read_to_end(&mut output)?;
            output
        } else if bytes.starts_with(b"BZh") {
            let mut output = Vec::new();
            BzDecoder::new(Cursor::new(bytes)).read_to_end(&mut output)?;
            output
        } else {
            bytes.to_vec()
        };
        let text = std::str::from_utf8(&decoded).map_err(|error| TrjError::Parse {
            line: 0,
            message: format!("trajectory is not valid UTF-8: {error}"),
        })?;
        parse_trj(text, n_atoms, dt)
    }

    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    pub fn to_universe(&self) -> crate::Result<Universe> {
        Universe::from_trj_file(self.clone())
    }
}

/// Read an Amber TRJ/MDCRD file from a path.
pub fn read_trj(path: impl AsRef<Path>, n_atoms: usize) -> Result<TrjFile, TrjError> {
    TrjFile::read_file(path, n_atoms)
}

/// Read an Amber TRJ/MDCRD file with an explicit frame interval.
pub fn read_trj_with_dt(
    path: impl AsRef<Path>,
    n_atoms: usize,
    dt: f64,
) -> Result<TrjFile, TrjError> {
    TrjFile::read_file_with_dt(path, n_atoms, dt)
}

impl CoordinateFile {
    pub fn read_trj<R: Read>(reader: R, n_atoms: usize) -> Result<Self, TrjError> {
        Ok(TrjFile::read(reader, n_atoms)?.coordinates)
    }

    pub fn read_trj_file(path: impl AsRef<Path>, n_atoms: usize) -> Result<Self, TrjError> {
        Ok(TrjFile::read_file(path, n_atoms)?.coordinates)
    }

    pub fn from_trj_bytes(bytes: &[u8], n_atoms: usize) -> Result<Self, TrjError> {
        Ok(TrjFile::from_bytes(bytes, n_atoms)?.coordinates)
    }
}

impl Universe {
    pub fn from_trj(path: impl AsRef<Path>, n_atoms: usize) -> crate::Result<Self> {
        Self::from_trj_file(read_trj(path, n_atoms)?)
    }

    pub fn from_trj_bytes(bytes: &[u8], n_atoms: usize) -> crate::Result<Self> {
        Self::from_trj_file(TrjFile::from_bytes(bytes, n_atoms)?)
    }

    pub fn from_trj_file(file: TrjFile) -> crate::Result<Self> {
        let atoms = file
            .coordinates
            .frames
            .first()
            .map(|frame| frame.positions.clone())
            .ok_or_else(|| crate::Error::InvalidInput("TRJ trajectory has no frames".to_owned()))?
            .into_iter()
            .enumerate()
            .map(|(index, position)| crate::core::Atom::new(index, "X", position))
            .collect();
        let mut universe = Universe {
            topology: crate::core::Topology::new(atoms),
            trajectory: Trajectory::default(),
        };
        attach_trj(&mut universe, file)?;
        Ok(universe)
    }

    pub fn from_psf_and_trj(
        psf_path: impl AsRef<Path>,
        trj_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_psf(psf_path)?;
        let n_atoms = universe.n_atoms();
        let trajectory = read_trj(trj_path, n_atoms)?;
        attach_trj(&mut universe, trajectory)?;
        Ok(universe)
    }

    pub fn from_psf_and_trj_bytes(psf: &str, bytes: &[u8], n_atoms: usize) -> crate::Result<Self> {
        let mut universe = Self::from_psf_str(psf)?;
        if universe.n_atoms() != n_atoms {
            return Err(crate::Error::InvalidInput(format!(
                "PSF contains {} atoms, requested TRJ atom count is {n_atoms}",
                universe.n_atoms()
            )));
        }
        attach_trj(&mut universe, TrjFile::from_bytes(bytes, n_atoms)?)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_trj(
        topology_path: impl AsRef<Path>,
        trj_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop(topology_path)?;
        let n_atoms = universe.n_atoms();
        let trajectory = read_trj(trj_path, n_atoms)?;
        attach_trj(&mut universe, trajectory)?;
        Ok(universe)
    }

    pub fn from_prmtop_and_trj_bytes(
        topology: &str,
        bytes: &[u8],
        n_atoms: usize,
    ) -> crate::Result<Self> {
        let mut universe = Self::from_prmtop_str(topology)?;
        if universe.n_atoms() != n_atoms {
            return Err(crate::Error::InvalidInput(format!(
                "PRMTOP contains {} atoms, requested TRJ atom count is {n_atoms}",
                universe.n_atoms()
            )));
        }
        attach_trj(&mut universe, TrjFile::from_bytes(bytes, n_atoms)?)?;
        Ok(universe)
    }
}

fn attach_trj(universe: &mut Universe, file: TrjFile) -> crate::Result<()> {
    if file.n_atoms != universe.n_atoms() {
        return Err(crate::Error::InvalidInput(format!(
            "TRJ contains {} atoms, topology contains {}",
            file.n_atoms,
            universe.n_atoms()
        )));
    }
    if file.coordinates.frames.is_empty() {
        return Err(crate::Error::InvalidInput(
            "TRJ trajectory has no frames".to_owned(),
        ));
    }
    let frames = file
        .coordinates
        .frames
        .into_iter()
        .map(|coordinate| {
            let mut frame = Frame::new(coordinate.positions);
            frame.dimensions = coordinate.dimensions;
            frame.step = coordinate.step;
            frame.time = coordinate.time;
            frame
        })
        .collect();
    universe.trajectory = Trajectory::new(frames);
    Ok(())
}

fn parse_trj(input: &str, n_atoms: usize, dt: f64) -> Result<TrjFile, TrjError> {
    let mut lines = input.lines();
    let title = lines.next().unwrap_or_default().trim_end().to_owned();
    if title.len() > 80 {
        return Err(TrjError::Parse {
            line: 1,
            message: "header is longer than 80 characters".to_owned(),
        });
    }
    let numbered = lines
        .enumerate()
        .filter_map(|(index, line)| {
            if line.trim().is_empty() {
                None
            } else {
                Some((index + 2, parse_values(line, index + 2)))
            }
        })
        .map(|(line, values)| values.map(|values| (line, values)))
        .collect::<Result<Vec<_>, _>>()?;
    if numbered.is_empty() {
        return Err(TrjError::InvalidInput(
            "trajectory contains no coordinate data".to_owned(),
        ));
    }
    let coordinate_values = n_atoms
        .checked_mul(3)
        .ok_or_else(|| TrjError::InvalidInput("n_atoms overflows usize".to_owned()))?;
    let coordinate_lines = coordinate_values.div_ceil(10);
    if numbered.len() < coordinate_lines {
        return Err(TrjError::InvalidInput(
            "trajectory ends before one frame".to_owned(),
        ));
    }
    let periodic = n_atoms > 1
        && numbered
            .get(coordinate_lines)
            .is_some_and(|(_, values)| values.len() == 3);
    let lines_per_frame = coordinate_lines + usize::from(periodic);
    if numbered.len() % lines_per_frame != 0 {
        return Err(TrjError::InvalidInput(format!(
            "trajectory has {} data lines, not a multiple of {lines_per_frame}",
            numbered.len()
        )));
    }
    let n_frames = numbered.len() / lines_per_frame;
    let mut frames = Vec::with_capacity(n_frames);
    for frame_index in 0..n_frames {
        let start = frame_index * lines_per_frame;
        let mut values = Vec::with_capacity(coordinate_values);
        for (line, line_values) in &numbered[start..start + coordinate_lines] {
            values.extend(line_values.iter().copied());
            if values.len() > coordinate_values {
                return Err(TrjError::Parse {
                    line: *line,
                    message: format!("frame contains more than {coordinate_values} coordinates"),
                });
            }
        }
        if values.len() != coordinate_values {
            return Err(TrjError::InvalidInput(format!(
                "frame {frame_index} contains {} coordinates, expected {coordinate_values}",
                values.len()
            )));
        }
        let positions = values
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        let dimensions = periodic.then(|| {
            let box_values = &numbered[start + coordinate_lines].1;
            [
                box_values[0],
                box_values[1],
                box_values[2],
                90.0,
                90.0,
                90.0,
            ]
        });
        let mut frame = CoordinateFrame::new(positions);
        frame.dimensions = dimensions;
        frame.step = frame_index;
        frame.time = frame_index as f64 * dt;
        frame.title = title.clone();
        frames.push(frame);
    }
    Ok(TrjFile {
        title,
        n_atoms,
        periodic,
        dt,
        coordinates: CoordinateFile::new(frames),
    })
}

fn parse_values(line: &str, line_number: usize) -> Result<Vec<f64>, TrjError> {
    let values = line
        .split_whitespace()
        .map(|value| parse_float(value).ok_or(()))
        .collect::<Result<Vec<_>, _>>();
    if let Ok(values) = values
        && !values.is_empty()
        && values.len() <= 10
    {
        return Ok(values);
    }
    let bytes = line.as_bytes();
    let mut values = Vec::new();
    for chunk in bytes.chunks(8) {
        let value = std::str::from_utf8(chunk)
            .ok()
            .and_then(|text| parse_float(text.trim()))
            .ok_or_else(|| TrjError::Parse {
                line: line_number,
                message: format!("invalid numeric field in {line:?}"),
            })?;
        values.push(value);
    }
    if values.is_empty() || values.len() > 10 {
        return Err(TrjError::Parse {
            line: line_number,
            message: "expected one to ten numeric fields".to_owned(),
        });
    }
    Ok(values)
}

fn parse_float(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .or_else(|| value.replace(['D', 'd'], "E").parse::<f64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/Amber")
            .join(name)
    }

    #[test]
    fn prmtop_constructor_preserves_selection_and_trajectory_geometry() {
        let universe =
            Universe::from_prmtop_and_trj(fixture("ache.prmtop"), fixture("ache.mdcrd")).unwrap();
        assert_eq!(universe.n_atoms(), 252);
        assert_eq!(universe.n_frames(), 11);
        assert_eq!(universe.trajectory.frames[0].dimensions, None);
        let protein = universe.select_atoms("protein").unwrap();
        assert_eq!(protein.len(), 252);
        let total: f64 = universe
            .trajectory
            .frames
            .iter()
            .map(|frame| {
                let positions = protein
                    .atoms
                    .iter()
                    .map(|atom| frame.positions[atom.index])
                    .collect::<Vec<_>>();
                let mut center = [0.0; 3];
                for position in &positions {
                    for axis in 0..3 {
                        center[axis] += position[axis];
                    }
                }
                center
                    .iter_mut()
                    .for_each(|value| *value /= positions.len() as f64);
                center.iter().sum::<f64>()
            })
            .sum();
        assert!((total - 472.2592159509659).abs() < 1.0e-3);
    }

    #[test]
    fn periodic_prmtop_constructor_preserves_box_and_protein_selection() {
        let universe = Universe::from_prmtop_and_trj(
            fixture("capped-ala.prmtop"),
            fixture("capped-ala.mdcrd.bz2"),
        )
        .unwrap();
        assert_eq!(universe.n_atoms(), 5071);
        assert_eq!(universe.n_frames(), 11);
        assert!(universe.trajectory.frames[0].dimensions.is_some());
        assert_eq!(universe.select_atoms("protein").unwrap().len(), 22);
    }

    #[test]
    fn reads_unboxed_and_compressed_trajectory() {
        let plain = read_trj(fixture("ache.mdcrd"), 252).unwrap();
        let compressed = read_trj(fixture("ache.mdcrd.bz2"), 252).unwrap();
        assert_eq!(plain.n_frames(), 11);
        assert!(!plain.periodic);
        assert_eq!(plain.coordinates, compressed.coordinates);
        assert_eq!(
            plain.frame(0).unwrap().positions[0],
            [32.555, 24.652, 14.213]
        );
    }

    #[test]
    fn reads_periodic_trajectory_and_explicit_dt() {
        let file = TrjFile::read_file_with_dt(fixture("capped-ala.mdcrd.bz2"), 5071, 0.5).unwrap();
        assert_eq!(file.n_frames(), 11);
        assert!(file.periodic);
        assert_eq!(
            file.frame(0).unwrap().dimensions.unwrap()[3..],
            [90.0, 90.0, 90.0]
        );
        assert_eq!(file.frame(2).unwrap().time, 1.0);
    }

    #[test]
    fn rejects_missing_atom_count_and_bad_header() {
        assert!(TrjFile::from_bytes(b"title\n 1.0 2.0 3.0\n", 0).is_err());
        let title = format!("{}\n", "x".repeat(81));
        assert!(TrjFile::from_bytes(title.as_bytes(), 1).is_err());
    }

    #[test]
    fn accepts_fortran_d_exponents() {
        let file = TrjFile::from_bytes(b"title\n 1.0D+00 2.0D+00 3.0D+00\n", 1).unwrap();
        assert_eq!(file.frame(0).unwrap().positions[0], [1.0, 2.0, 3.0]);
    }
}
