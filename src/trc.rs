//! GROMOS11 TRC text trajectory support.
//!
//! The reader handles the blocks used by ordinary GROMOS trajectories:
//! `TIMESTEP`, `POSITIONRED`, and `GENBOX`. Other blocks are skipped as a
//! unit, preserving frame boundaries while making their unsupported status
//! explicit through the parsed result's lack of corresponding data.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use crate::core::{Atom, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// A parsed GROMOS11 TRC trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct TrcFile {
    /// Optional title text from the first `TITLE` block.
    pub title: String,
    /// Trajectory frames in source units (nm and ps).
    pub coordinates: CoordinateFile,
}

/// Naming aliases matching the other coordinate modules.
pub type TrcData = TrcFile;
pub type TrcStructure = TrcFile;

impl TrcFile {
    /// Parse a TRC document from a path, transparently decompressing gzip.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, TrcError> {
        let mut file = File::open(path)?;
        Self::read(&mut file)
    }

    /// Alias for [`TrcFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TrcError> {
        Self::read_file(path)
    }

    /// Parse a TRC document from any reader, accepting gzip-compressed bytes.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, TrcError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a TRC document held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrcError> {
        let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut decoder = GzDecoder::new(Cursor::new(bytes));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else if bytes.starts_with(b"BZh") {
            let mut decoder = BzDecoder::new(Cursor::new(bytes));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            bytes.to_vec()
        };
        let text = String::from_utf8(bytes)
            .map_err(|error| invalid(format!("TRC document is not UTF-8: {error}")))?;
        parse_trc(&text)
    }

    /// Number of atoms in each frame.
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.coordinates.n_atoms()
    }

    /// Number of frames in the trajectory.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    /// Read-only access to one coordinate frame by zero-based index.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }

    /// Read and concatenate several TRC files in source order.
    ///
    /// This is the in-memory equivalent of MDAnalysis' ``continuous=True``
    /// multi-file trajectory mode. Every input must contain at least one
    /// frame and all files must have the same atom count. Frame step and time
    /// values are retained exactly as written in each source file.
    pub fn read_files<I, P>(paths: I) -> Result<Self, TrcError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut files = paths.into_iter();
        let first = files
            .next()
            .ok_or_else(|| invalid("no TRC paths were supplied"))?;
        let mut result = Self::read_file(first)?;
        let expected_atoms = result.n_atoms();
        for path in files {
            let next = Self::read_file(path)?;
            if next.n_atoms() != expected_atoms {
                return Err(invalid(format!(
                    "TRC file contains {} atoms; expected {expected_atoms}",
                    next.n_atoms()
                )));
            }
            result.coordinates.frames.extend(next.coordinates.frames);
        }
        Ok(result)
    }

    /// Construct a coordinate-only universe from this trajectory.
    pub fn to_universe(&self) -> crate::Result<Universe> {
        Universe::from_trc_file(self.clone())
    }
}

/// Read a TRC trajectory from a path.
pub fn read_trc(path: impl AsRef<Path>) -> Result<TrcFile, TrcError> {
    TrcFile::read_file(path)
}

/// Read and concatenate several TRC trajectories in source order.
pub fn read_trc_files<I, P>(paths: I) -> Result<TrcFile, TrcError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    TrcFile::read_files(paths)
}

impl CoordinateFile {
    /// Read a TRC trajectory and return its coordinate frames.
    pub fn read_trc<R: Read>(reader: R) -> Result<Self, TrcError> {
        Ok(TrcFile::read(reader)?.coordinates)
    }

    /// Parse TRC bytes and return the coordinate frames.
    pub fn from_trc_bytes(bytes: &[u8]) -> Result<Self, TrcError> {
        Ok(TrcFile::from_bytes(bytes)?.coordinates)
    }
}

impl Universe {
    /// Construct a coordinate-only universe from a TRC trajectory.
    pub fn from_trc(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_trc_file(read_trc(path)?)
    }

    /// Construct a coordinate-only universe from parsed TRC data.
    pub fn from_trc_file(file: TrcFile) -> crate::Result<Self> {
        let first =
            file.coordinates.frames.first().ok_or_else(|| {
                crate::Error::InvalidInput("TRC trajectory has no frames".to_owned())
            })?;
        let atoms = first
            .positions
            .iter()
            .enumerate()
            .map(|(index, &position)| {
                let name = first
                    .names
                    .get(index)
                    .filter(|name| !name.is_empty())
                    .map_or("X", String::as_str);
                let mut atom = Atom::new(index, name, position);
                atom.element = crate::guesser::guess_element(name, None, None).ok();
                atom.mass = atom
                    .element
                    .as_deref()
                    .and_then(|element| crate::guesser::guess_mass(element).ok())
                    .unwrap_or(0.0);
                atom
            })
            .collect::<Vec<_>>();
        let frames = file
            .coordinates
            .frames
            .into_iter()
            .map(|source| {
                let mut frame = Frame::new(source.positions);
                frame.velocities = source.velocities;
                frame.dimensions = source.dimensions;
                frame.step = source.step;
                frame.time = source.time;
                frame
            })
            .collect();
        Ok(Self {
            topology: Topology::new(atoms),
            trajectory: Trajectory::new(frames),
        })
    }

    /// Construct a universe from a PDB topology and one TRC trajectory.
    pub fn from_pdb_and_trc(
        pdb_path: impl AsRef<Path>,
        trc_path: impl AsRef<Path>,
    ) -> crate::Result<Self> {
        let universe = Self::from_pdb(pdb_path)?;
        attach_trc(universe, read_trc(trc_path)?)
    }

    /// Construct a universe from a PDB topology and several concatenated TRC
    /// trajectories.
    pub fn from_pdb_and_trc_files<I, P>(
        pdb_path: impl AsRef<Path>,
        trc_paths: I,
    ) -> crate::Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let universe = Self::from_pdb(pdb_path)?;
        attach_trc(universe, TrcFile::read_files(trc_paths)?)
    }

    /// Construct a universe from PDB text and TRC bytes.
    pub fn from_pdb_and_trc_bytes(pdb: &str, trc: &[u8]) -> crate::Result<Self> {
        let universe = Self::from_pdb_str(pdb)?;
        attach_trc(universe, TrcFile::from_bytes(trc)?)
    }
}

fn attach_trc(mut universe: Universe, file: TrcFile) -> crate::Result<Universe> {
    if universe.n_atoms() != file.n_atoms() {
        return Err(crate::Error::InvalidInput(format!(
            "TRC contains {} atoms, topology contains {}",
            file.n_atoms(),
            universe.n_atoms()
        )));
    }
    let frames = file
        .coordinates
        .frames
        .into_iter()
        .map(|source| {
            let mut frame = Frame::new(source.positions);
            frame.dimensions = source.dimensions;
            frame.step = source.step;
            frame.time = source.time;
            frame
        })
        .collect();
    universe.trajectory = Trajectory::new(frames);
    Ok(universe)
}

/// Errors produced while reading a TRC document.
#[derive(Debug)]
pub enum TrcError {
    Io(io::Error),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for TrcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "TRC I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "TRC parse error on line {line}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid TRC structure: {message}")
            }
        }
    }
}

impl std::error::Error for TrcError {}

impl From<io::Error> for TrcError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Default)]
struct PendingFrame {
    step: usize,
    time: f64,
    positions: Option<Vec<[f64; 3]>>,
    dimensions: Option<[f64; 6]>,
}

fn parse_trc(text: &str) -> Result<TrcFile, TrcError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut title = String::new();
    let mut frames = Vec::new();
    let mut pending = PendingFrame::default();
    let mut saw_supported_block = false;
    let mut line_index = 0;
    while line_index < lines.len() {
        let line = lines[line_index].trim();
        let current_line = line_index + 1;
        if line.is_empty() {
            line_index += 1;
            continue;
        }
        match line {
            "TITLE" => {
                let (content, next) = read_block(&lines, line_index + 1)?;
                if title.is_empty() {
                    title = content
                        .iter()
                        .filter(|line| !line.trim().is_empty())
                        .map(|line| line.trim())
                        .collect::<Vec<_>>()
                        .join("\n");
                }
                line_index = next;
            }
            "TIMESTEP" => {
                if pending.positions.is_some() {
                    frames.push(finish_frame(&mut pending, frames.len())?);
                }
                let (content, next) = read_block(&lines, line_index + 1)?;
                let values = content
                    .iter()
                    .map(|line| line.trim())
                    .find(|line| !line.is_empty() && !line.starts_with('#'))
                    .ok_or_else(|| parse_error(current_line, "TIMESTEP has no values"))?;
                let fields: Vec<&str> = values.split_whitespace().collect();
                if fields.len() < 2 {
                    return Err(parse_error(
                        current_line,
                        "TIMESTEP requires a step and time",
                    ));
                }
                pending.step = fields[0].parse::<usize>().map_err(|error| {
                    parse_error(current_line + 1, format!("invalid step: {error}"))
                })?;
                pending.time = parse_float(current_line + 1, fields[1], "time")?;
                saw_supported_block = true;
                line_index = next;
            }
            "POSITIONRED" => {
                if pending.positions.is_some() {
                    frames.push(finish_frame(&mut pending, frames.len())?);
                }
                let (content, next) = read_block(&lines, line_index + 1)?;
                let mut positions = Vec::new();
                for (offset, source) in content.iter().enumerate() {
                    let source = source.trim();
                    if source.is_empty() || source.starts_with('#') {
                        continue;
                    }
                    let fields: Vec<&str> = source.split_whitespace().collect();
                    if fields.len() != 3 {
                        return Err(parse_error(
                            line_index + 2 + offset,
                            "POSITIONRED coordinate row requires three values",
                        ));
                    }
                    positions.push([
                        parse_float(line_index + 2 + offset, fields[0], "x coordinate")?,
                        parse_float(line_index + 2 + offset, fields[1], "y coordinate")?,
                        parse_float(line_index + 2 + offset, fields[2], "z coordinate")?,
                    ]);
                }
                if positions.is_empty() {
                    return Err(parse_error(
                        current_line,
                        "POSITIONRED block contains no coordinates",
                    ));
                }
                pending.positions = Some(positions);
                saw_supported_block = true;
                line_index = next;
            }
            "POSITION" => {
                let (_, next) = read_block(&lines, line_index + 1)?;
                // POSITION contains annotated coordinates which the native
                // GROMOS reader intentionally does not use. Skip it as a
                // whole block so following POSITIONRED frames remain aligned.
                line_index = next;
            }
            "GENBOX" => {
                let (content, next) = read_block(&lines, line_index + 1)?;
                pending.dimensions = parse_genbox(&content, current_line)?;
                saw_supported_block = true;
                line_index = next;
            }
            _ => {
                // Any unrecognized block is skipped through its END marker.
                // A free-standing line outside a block is ignored as a
                // comment/metadata line, matching GROMOS's permissive text
                // format.
                line_index += 1;
            }
        }
    }
    if pending.positions.is_some() {
        frames.push(finish_frame(&mut pending, frames.len())?);
    }
    if !saw_supported_block || frames.is_empty() {
        return Err(invalid(
            "no supported blocks were found within the GROMOS trajectory",
        ));
    }
    let expected_atoms = frames[0].positions.len();
    for (index, frame) in frames.iter().enumerate() {
        if frame.positions.len() != expected_atoms {
            return Err(invalid(format!(
                "frame {} contains {} atoms; expected {}",
                index,
                frame.positions.len(),
                expected_atoms
            )));
        }
    }
    let coordinate_frames = frames
        .into_iter()
        .map(|frame| {
            let mut coordinate = CoordinateFrame::new(frame.positions);
            coordinate.velocities = None;
            coordinate.dimensions = frame.dimensions;
            coordinate.step = frame.step;
            coordinate.time = frame.time;
            coordinate
        })
        .collect();
    Ok(TrcFile {
        title,
        coordinates: CoordinateFile::new(coordinate_frames),
    })
}

fn finish_frame(pending: &mut PendingFrame, _frame: usize) -> Result<RawFrame, TrcError> {
    let positions = pending
        .positions
        .take()
        .ok_or_else(|| invalid("frame has no POSITIONRED block"))?;
    Ok(RawFrame {
        step: pending.step,
        time: pending.time,
        positions,
        dimensions: pending.dimensions.take(),
    })
}

struct RawFrame {
    step: usize,
    time: f64,
    positions: Vec<[f64; 3]>,
    dimensions: Option<[f64; 6]>,
}

fn read_block<'a>(lines: &[&'a str], start: usize) -> Result<(Vec<&'a str>, usize), TrcError> {
    let mut content = Vec::new();
    for (index, source) in lines.iter().enumerate().skip(start) {
        if source.trim() == "END" {
            return Ok((content, index + 1));
        }
        content.push(source);
    }
    Err(parse_error(
        start + 1,
        "block is missing its terminating END line",
    ))
}

fn parse_genbox(content: &[&str], line: usize) -> Result<Option<[f64; 6]>, TrcError> {
    let values = content
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    if values.len() < 5 {
        return Err(parse_error(line, "GENBOX requires five data lines"));
    }
    let setting = values[0]
        .parse::<i32>()
        .map_err(|error| parse_error(line + 1, format!("invalid GENBOX setting: {error}")))?;
    if setting == 0 {
        return Ok(None);
    }
    if setting == -1 {
        return Err(invalid(
            "truncated-octahedral GENBOX boxes are not supported",
        ));
    }
    if !matches!(setting, 1 | 2) {
        return Err(invalid(format!(
            "GENBOX setting {setting} is not supported"
        )));
    }
    let lengths = parse_triplet(values[1], line + 2, "GENBOX lengths")?;
    let angles = parse_triplet(values[2], line + 3, "GENBOX angles")?;
    if lengths.iter().any(|value| *value <= 0.0) {
        return Err(invalid("GENBOX lengths must be positive"));
    }
    if angles.iter().any(|value| !(*value > 0.0 && *value < 180.0)) {
        return Err(invalid(
            "GENBOX angles must lie strictly between 0 and 180 degrees",
        ));
    }
    let origin = parse_triplet(values[3], line + 4, "GENBOX origin")?;
    if origin.iter().any(|value| value.abs() > 1.0e-10) {
        return Err(invalid("shifted GENBOX origins are not supported"));
    }
    let euler = parse_triplet(values[4], line + 5, "GENBOX Euler angles")?;
    if euler.iter().any(|value| value.abs() > 1.0e-10) {
        return Err(invalid(
            "yawed, pitched, or rolled GENBOX boxes are not supported",
        ));
    }
    let dimensions = [
        lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2],
    ];
    // Exercise the geometry conversion so degenerate angle combinations are
    // rejected rather than being propagated as invalid dimensions.
    let vectors = crate::mdamath::triclinic_vectors(dimensions);
    let round_trip = triclinic_box(vectors);
    if round_trip.iter().any(|value| !value.is_finite()) {
        return Err(invalid("GENBOX dimensions are not finite"));
    }
    Ok(Some(dimensions))
}

fn parse_triplet(line: &str, number: usize, field: &str) -> Result<[f64; 3], TrcError> {
    let values = line.split_whitespace().collect::<Vec<_>>();
    if values.len() != 3 {
        return Err(parse_error(
            number,
            format!("{field} requires three values"),
        ));
    }
    Ok([
        parse_float(number, values[0], field)?,
        parse_float(number, values[1], field)?,
        parse_float(number, values[2], field)?,
    ])
}

fn parse_float(line: usize, value: &str, field: &str) -> Result<f64, TrcError> {
    let value = value
        .parse::<f64>()
        .map_err(|error| parse_error(line, format!("invalid {field}: {error}")))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(parse_error(line, format!("{field} is not finite")))
    }
}

fn parse_error(line: usize, message: impl Into<String>) -> TrcError {
    TrcError::Parse {
        line,
        message: message.into(),
    }
}

fn invalid(message: impl Into<String>) -> TrcError {
    TrcError::InvalidStructure(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/gromos11")
            .join(name)
    }

    #[test]
    fn reads_gzip_vacuum_trajectory() {
        let file = read_trc(fixture("gromos11_traj_vac_1.trc.gz")).unwrap();
        assert_eq!(file.n_atoms(), 73);
        assert_eq!(file.n_frames(), 3);
        assert_eq!(file.coordinates.frames[0].step, 0);
        assert_eq!(file.coordinates.frames[0].time, 0.0);
        assert_eq!(file.coordinates.frames[0].dimensions, None);
        assert!((file.coordinates.frames[0].positions[0][0] - 0.219782507).abs() < 1e-9);
    }

    #[test]
    fn reads_orthorhombic_and_triclinic_boxes() {
        let orthorhombic = read_trc(fixture("gromos11_traj_solv.trc.gz")).unwrap();
        assert_eq!(orthorhombic.n_atoms(), 2797);
        assert_eq!(orthorhombic.n_frames(), 2);
        assert!(
            (orthorhombic.coordinates.frames[1].dimensions.unwrap()[0] - 3.054416298).abs() < 1e-9
        );
        let triclinic = read_trc(fixture("gromos11_triclinic_solv.trc.gz")).unwrap();
        let dimensions = triclinic.coordinates.frames[0].dimensions.unwrap();
        assert!((dimensions[0] - 3.372394463).abs() < 1e-9);
        assert!((dimensions[5] - 54.514000084).abs() < 1e-9);
    }

    #[test]
    fn rejects_unsupported_genbox_and_atom_counts() {
        assert!(read_trc(fixture("gromos11_genbox_origin.trc.gz")).is_err());
        assert!(read_trc(fixture("gromos11_genbox_euler.trc.gz")).is_err());
        assert!(read_trc(fixture("gromos11_truncOcta_vac.trc.gz")).is_err());
        assert!(read_trc(fixture("gromos11_traj_vac_1_missing_pos.trc.gz")).is_err());
        assert!(read_trc(fixture("gromos11_traj_vac_1_extra_pos.trc.gz")).is_err());
        assert!(read_trc(fixture("gromos11_empty.trc")).is_err());
    }

    #[test]
    fn universe_constructor_preserves_step_and_time() {
        let file = read_trc(fixture("gromos11_traj_vac_1.trc.gz")).unwrap();
        let universe = Universe::from_trc_file(file).unwrap();
        assert_eq!(universe.n_atoms(), 73);
        assert_eq!(universe.n_frames(), 3);
        assert_eq!(universe.trajectory.frames[0].time, 0.0);
    }

    #[test]
    fn concatenates_multiple_files_and_attaches_pdb_topology() {
        let first = fixture("gromos11_traj_vac_1.trc.gz");
        let second = fixture("gromos11_traj_vac_2.trc.gz");
        let file = TrcFile::read_files([&first, &second]).unwrap();
        assert_eq!(file.n_atoms(), 73);
        assert_eq!(file.n_frames(), 6);
        assert_eq!(file.coordinates.frames[0].step, 0);
        assert_eq!(file.coordinates.frames[3].step, 0);
        assert_eq!(file.coordinates.frames[5].time, 100.0);
        assert_eq!(file.frame(4).unwrap().positions[0][0], 0.037026654);
        assert!(file.frame(6).is_none());

        let pdb = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/gromos11/gromos11_traj_vac.pdb.gz");
        let universe = Universe::from_pdb_and_trc_files(&pdb, [&first, &second]).unwrap();
        assert_eq!(universe.n_atoms(), 73);
        assert_eq!(universe.n_frames(), 6);
        assert!(!universe.topology.bonds.is_empty());
        assert_eq!(universe.trajectory.frames[4].step, 10000);
    }

    #[test]
    fn accepts_bzip2_bytes_and_rejects_empty_path_lists() {
        let source = std::fs::read(fixture("gromos11_traj_vac_1.trc.gz")).unwrap();
        let mut decoder = GzDecoder::new(Cursor::new(source));
        let mut plain = Vec::new();
        decoder.read_to_end(&mut plain).unwrap();
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        std::io::Write::write_all(&mut encoder, &plain).unwrap();
        let compressed = encoder.finish().unwrap();
        assert_eq!(TrcFile::from_bytes(&compressed).unwrap().n_frames(), 3);
        assert!(TrcFile::read_files::<Vec<PathBuf>, PathBuf>(Vec::new()).is_err());
    }

    #[test]
    fn reads_cluster_position_blocks_without_timestep_metadata() {
        let file = read_trc(fixture("gromos11_cluster_vac.trj.gz")).unwrap();
        assert_eq!(file.n_atoms(), 73);
        assert_eq!(file.n_frames(), 3);
        assert_eq!(file.coordinates.frames[0].step, 0);
        assert_eq!(file.coordinates.frames[2].time, 0.0);
        assert!((file.coordinates.frames[0].positions[0][0] - 2.373409727).abs() < 1.0e-9);
    }
}
