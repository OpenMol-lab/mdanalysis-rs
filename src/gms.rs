//! Readers for GAMESS (GMS) text output.
//!
//! GAMESS output does not have a single formal trajectory grammar, but the
//! commonly produced US-GAMESS and Firefly output contains two stable pieces
//! of information: an atomic coordinate table in the `(BOHR)` section and
//! either optimization (`RUNTYP=OPTIMIZE`) or surface-mapping coordinate
//! blocks.  This module retains the topology names and nuclear charges and
//! converts the coordinate blocks into [`crate::coordinates::CoordinateFile`]
//! frames.  GAMESS output is read-only; no synthetic output writer is
//! provided.

use crate::coordinates::{CoordinateFile, CoordinateFrame};
use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::fmt;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// An atom from GAMESS' atomic coordinate table.
#[derive(Clone, Debug, PartialEq)]
pub struct GmsAtom {
    /// GAMESS atom label, such as `O`, `CARBON`, or `HYDROGEN`.
    pub name: String,
    /// Nuclear/atomic charge printed by GAMESS (normally an integer value).
    pub atomic_charge: f64,
    /// Coordinates from the topology table, in Bohr as printed by GAMESS.
    pub position: [f64; 3],
}

/// A parsed GAMESS output document.
#[derive(Clone, Debug, PartialEq)]
pub struct GmsFile {
    /// Lower-case GAMESS run type (`optimize` or `surface`).
    pub runtyp: String,
    /// Number of atoms reported by GAMESS, or inferred from the topology
    /// table when the report omits the total-count line.
    pub n_atoms: usize,
    /// Atom names, nuclear charges, and the initial Bohr coordinates.
    pub atoms: Vec<GmsAtom>,
    /// Cartesian coordinate frames in Angstroms.
    pub coordinates: CoordinateFile,
}

/// Compatibility aliases for callers that use topology/data terminology.
pub type GmsStructure = GmsFile;
pub type GmsData = GmsFile;
pub type GmsParser = GmsFile;
pub type GmsReader = GmsFile;

impl GmsFile {
    /// Parse GAMESS output held in memory.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, GmsError> {
        parse_gms(input)
    }

    /// Read uncompressed GAMESS output from any reader.
    pub fn read<R: Read>(mut reader: R) -> Result<Self, GmsError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    /// Parse GAMESS output bytes, transparently decompressing gzip or bzip2
    /// streams based on their magic bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, GmsError> {
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
        let input = std::str::from_utf8(&decoded).map_err(|error| {
            GmsError::InvalidStructure(format!("GMS output is not UTF-8: {error}"))
        })?;
        Self::from_str(input)
    }

    /// Read GAMESS output from a path, transparently decompressing gzip or
    /// bzip2 streams based on their magic bytes.
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, GmsError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Number of parsed coordinate frames.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.coordinates.n_frames()
    }

    /// Read-only access to one coordinate frame.
    #[must_use]
    pub fn frame(&self, index: usize) -> Option<&CoordinateFrame> {
        self.coordinates.frame(index)
    }
}

impl std::str::FromStr for GmsFile {
    type Err = GmsError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

/// Errors produced while parsing GAMESS output.
#[derive(Debug)]
pub enum GmsError {
    Io(io::Error),
    Parse { line: usize, message: String },
    InvalidStructure(String),
}

impl fmt::Display for GmsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse { line, message } => {
                write!(formatter, "GMS parse error on line {line}: {message}")
            }
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid GMS structure: {message}")
            }
        }
    }
}

impl std::error::Error for GmsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { .. } | Self::InvalidStructure(_) => None,
        }
    }
}

impl From<io::Error> for GmsError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Read GAMESS output from a filesystem path.
pub fn read_gms<P: AsRef<Path>>(path: P) -> Result<GmsFile, GmsError> {
    GmsFile::read_file(path)
}

impl CoordinateFile {
    /// Read GAMESS output and retain only its coordinate frames.
    pub fn read_gms<R: Read>(reader: R) -> Result<Self, GmsError> {
        Ok(GmsFile::read(reader)?.coordinates)
    }

    /// Parse GAMESS output and retain only its coordinate frames.
    pub fn from_gms_str(input: &str) -> Result<Self, GmsError> {
        Ok(GmsFile::from_str(input)?.coordinates)
    }
}

impl crate::core::Universe {
    /// Construct a universe from GAMESS output.
    pub fn from_gms(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_gms_file(GmsFile::read_file(path)?)
    }

    /// Construct a universe from GAMESS output held in memory.
    pub fn from_gms_str(input: &str) -> crate::Result<Self> {
        Self::from_gms_file(GmsFile::from_str(input)?)
    }

    /// Construct a universe from parsed GAMESS output.
    pub fn from_gms_file(file: GmsFile) -> crate::Result<Self> {
        if file.atoms.is_empty() {
            return Err(crate::Error::InvalidInput(
                "GMS file contains no atoms".to_owned(),
            ));
        }
        let first = file.coordinates.frames.first().ok_or_else(|| {
            crate::Error::InvalidInput("GMS file contains no coordinate frames".to_owned())
        })?;
        if first.positions.len() != file.atoms.len() {
            return Err(crate::Error::InvalidInput(format!(
                "GMS topology contains {} atoms but first frame contains {}",
                file.atoms.len(),
                first.positions.len()
            )));
        }
        let atoms = file
            .atoms
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let mut atom =
                    crate::core::Atom::new(index, source.name.clone(), first.positions[index]);
                atom.charge = source.atomic_charge;
                atom.element = crate::guesser::guess_element(&source.name, None, None).ok();
                atom.mass = crate::guesser::guess_atom_mass(&source.name);
                atom
            })
            .collect();
        let topology = crate::core::Topology::new(atoms);
        let frames = file
            .coordinates
            .frames
            .into_iter()
            .map(|source| {
                let mut frame = crate::core::Frame::new(source.positions);
                frame.step = source.step;
                frame.time = source.time;
                frame.dimensions = source.dimensions;
                frame.velocities = source.velocities;
                frame
            })
            .collect();
        Ok(Self {
            topology,
            trajectory: crate::core::Trajectory::new(frames),
        })
    }
}

fn parse_gms(input: &str) -> Result<GmsFile, GmsError> {
    let lines: Vec<&str> = input.lines().collect();
    let runtyp = find_runtyp(&lines)?;
    if runtyp != "optimize" && runtyp != "surface" {
        return Err(GmsError::InvalidStructure(format!(
            "unsupported RUNTYP={runtyp}; expected OPTIMIZE or SURFACE"
        )));
    }

    let reported_atoms = find_reported_atom_count(&lines)?;
    let atoms = parse_topology_atoms(&lines)?;
    let n_atoms = reported_atoms.unwrap_or(atoms.len());
    if n_atoms == 0 {
        return Err(GmsError::InvalidStructure(
            "GMS output contains no atoms".to_owned(),
        ));
    }
    if atoms.len() != n_atoms {
        return Err(GmsError::InvalidStructure(format!(
            "topology contains {} atoms but report declares {n_atoms}",
            atoms.len()
        )));
    }

    let coordinates = if runtyp == "optimize" {
        parse_optimize_frames(&lines, n_atoms)?
    } else {
        parse_surface_frames(&lines, n_atoms)?
    };
    if coordinates.frames.is_empty() {
        return Err(GmsError::InvalidStructure(format!(
            "RUNTYP={runtyp} output contains no coordinate frames"
        )));
    }
    Ok(GmsFile {
        runtyp,
        n_atoms,
        atoms,
        coordinates,
    })
}

fn find_runtyp(lines: &[&str]) -> Result<String, GmsError> {
    let mut echoed = None;
    let mut declared = None;
    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with('!') || trimmed.starts_with('*') {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if let Some(value) = find_assignment(line, "RUNTYP") {
            let value = value.to_ascii_lowercase();
            if upper.contains("INPUT CARD>") {
                echoed = Some(value);
            } else {
                declared = Some(value);
            }
        }
    }
    declared
        .or(echoed)
        .ok_or_else(|| GmsError::InvalidStructure("missing RUNTYP= declaration".to_owned()))
}

fn find_reported_atom_count(lines: &[&str]) -> Result<Option<usize>, GmsError> {
    let mut reported = None;
    for (offset, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('!')
            || trimmed.starts_with('*')
            || trimmed.to_ascii_uppercase().contains("INPUT CARD>")
        {
            continue;
        }
        if trimmed
            .to_ascii_uppercase()
            .starts_with("TOTAL NUMBER OF ATOMS")
        {
            let value = find_assignment(line, "TOTAL NUMBER OF ATOMS")
                .ok_or_else(|| parse_error(offset + 1, "missing atom count after '='"))?;
            let count = value.parse::<usize>().map_err(|error| {
                parse_error(offset + 1, format!("invalid total atom count: {error}"))
            })?;
            reported = Some(count);
        }
    }
    Ok(reported)
}

fn find_assignment(line: &str, key: &str) -> Option<String> {
    let upper = line.to_ascii_uppercase();
    let mut search_from = 0;
    while let Some(relative) = upper[search_from..].find(key) {
        let index = search_from + relative;
        let before_is_word = index
            .checked_sub(1)
            .and_then(|position| upper.as_bytes().get(position))
            .is_some_and(|character| character.is_ascii_alphanumeric() || *character == b'_');
        if before_is_word {
            search_from = index + key.len();
            continue;
        }
        let suffix = line.get(index + key.len()..)?.trim_start();
        let value = suffix.strip_prefix('=')?.split_whitespace().next()?;
        return Some(value.to_owned());
    }
    None
}

fn parse_topology_atoms(lines: &[&str]) -> Result<Vec<GmsAtom>, GmsError> {
    let marker = lines.iter().position(|line| {
        let upper = line.to_ascii_uppercase();
        upper.contains("ATOM")
            && upper.contains("ATOMIC")
            && upper.contains("COORDINATES")
            && upper.contains("(BOHR)")
    });
    let marker = marker.ok_or_else(|| {
        GmsError::InvalidStructure("missing ATOM ATOMIC COORDINATES (BOHR) table".to_owned())
    })?;

    let mut atoms = Vec::new();
    for (offset, line) in lines.iter().enumerate().skip(marker + 1) {
        if let Some(atom) = parse_atom_line(line, offset + 1)? {
            atoms.push(atom);
        } else if !atoms.is_empty() {
            break;
        }
    }
    if atoms.is_empty() {
        return Err(GmsError::InvalidStructure(
            "atomic coordinate table contains no atom records".to_owned(),
        ));
    }
    Ok(atoms)
}

fn parse_optimize_frames(lines: &[&str], n_atoms: usize) -> Result<CoordinateFile, GmsError> {
    let mut frames = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if !is_prefixed_marker(line, "NSERCH=") {
            continue;
        }
        let end = lines[index + 1..]
            .iter()
            .position(|candidate| is_prefixed_marker(candidate, "NSERCH="))
            .map_or(lines.len(), |position| index + 1 + position);
        let marker = lines[index + 1..end]
            .iter()
            .position(|candidate| {
                candidate
                    .to_ascii_uppercase()
                    .contains("COORDINATES OF ALL ATOMS ARE")
            })
            .map(|position| index + 1 + position)
            .ok_or_else(|| parse_error(index + 1, "optimization step has no coordinate table"))?;
        let (positions, names) = parse_coordinate_table(lines, marker, n_atoms, 2, end)?;
        let mut frame = CoordinateFrame::new(positions);
        frame.names = names;
        frame.step = frames.len();
        frame.time = frames.len() as f64;
        frames.push(frame);
    }
    Ok(CoordinateFile::new(frames))
}

fn is_prefixed_marker(line: &str, marker: &str) -> bool {
    line.get(1..)
        .is_some_and(|suffix| suffix.starts_with(marker))
        || line
            .trim_start()
            .get(1..)
            .is_some_and(|suffix| suffix.starts_with(marker))
}

fn parse_surface_frames(lines: &[&str], n_atoms: usize) -> Result<CoordinateFile, GmsError> {
    let mut frames = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if !line
            .trim_start()
            .to_ascii_uppercase()
            .starts_with("COORD 1=")
        {
            continue;
        }
        let end = lines[index + 1..]
            .iter()
            .position(|candidate| {
                candidate
                    .trim_start()
                    .to_ascii_uppercase()
                    .starts_with("COORD 1=")
            })
            .map_or(lines.len(), |position| index + 1 + position);
        let energy = lines[index + 1..end].iter().position(|candidate| {
            candidate
                .trim_start()
                .to_ascii_uppercase()
                .starts_with("HAS ENERGY VALUE")
        });
        let energy = energy
            .map(|position| index + 1 + position)
            .ok_or_else(|| parse_error(index + 1, "surface step has no energy marker"))?;
        let (positions, names) = parse_surface_table(lines, energy, n_atoms, end)?;
        let mut frame = CoordinateFrame::new(positions);
        frame.names = names;
        frame.step = frames.len();
        frame.time = frames.len() as f64;
        frames.push(frame);
    }
    Ok(CoordinateFile::new(frames))
}

fn parse_coordinate_table(
    lines: &[&str],
    marker: usize,
    n_atoms: usize,
    coordinate_offset: usize,
    end: usize,
) -> Result<(Vec<[f64; 3]>, Vec<String>), GmsError> {
    let separator = lines[marker + 1..end]
        .iter()
        .position(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && trimmed.chars().all(|character| character == '-')
        })
        .map(|position| marker + 1 + position)
        .ok_or_else(|| parse_error(marker + 1, "coordinate table has no separator"))?;
    let mut positions = Vec::with_capacity(n_atoms);
    let mut names = Vec::with_capacity(n_atoms);
    for (offset, line) in lines
        .iter()
        .enumerate()
        .skip(separator + 1)
        .take(end.saturating_sub(separator + 1))
    {
        if positions.len() == n_atoms {
            break;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < coordinate_offset + 3 {
            if positions.is_empty() {
                continue;
            }
            return Err(parse_error(
                offset + 1,
                "coordinate table ended before all atoms",
            ));
        }
        let position = [
            parse_finite(fields[coordinate_offset], offset + 1, "x coordinate")?,
            parse_finite(fields[coordinate_offset + 1], offset + 1, "y coordinate")?,
            parse_finite(fields[coordinate_offset + 2], offset + 1, "z coordinate")?,
        ];
        positions.push(position);
        names.push(fields[0].to_owned());
    }
    if positions.len() != n_atoms {
        return Err(parse_error(
            lines.len() + 1,
            format!(
                "coordinate table contains {} atoms; expected {n_atoms}",
                positions.len()
            ),
        ));
    }
    Ok((positions, names))
}

fn parse_surface_table(
    lines: &[&str],
    energy: usize,
    n_atoms: usize,
    end: usize,
) -> Result<(Vec<[f64; 3]>, Vec<String>), GmsError> {
    let mut positions = Vec::with_capacity(n_atoms);
    let mut names = Vec::with_capacity(n_atoms);
    for (offset, line) in lines
        .iter()
        .enumerate()
        .skip(energy + 1)
        .take(end.saturating_sub(energy + 1))
    {
        if positions.len() == n_atoms {
            break;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            if positions.is_empty() {
                continue;
            }
            return Err(parse_error(
                offset + 1,
                "surface coordinate table ended early",
            ));
        }
        let position = [
            parse_finite(fields[1], offset + 1, "x coordinate")?,
            parse_finite(fields[2], offset + 1, "y coordinate")?,
            parse_finite(fields[3], offset + 1, "z coordinate")?,
        ];
        positions.push(position);
        names.push(fields[0].to_owned());
    }
    if positions.len() != n_atoms {
        return Err(parse_error(
            lines.len() + 1,
            format!(
                "surface coordinate table contains {} atoms; expected {n_atoms}",
                positions.len()
            ),
        ));
    }
    Ok((positions, names))
}

fn parse_atom_line(line: &str, line_number: usize) -> Result<Option<GmsAtom>, GmsError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 5 {
        return Ok(None);
    }
    if fields[0].eq_ignore_ascii_case("ATOM") || fields[0].eq_ignore_ascii_case("CHARGE") {
        return Ok(None);
    }
    if fields[0]
        .chars()
        .next()
        .is_none_or(|character| !character.is_ascii_alphabetic() && character != '_')
    {
        return Ok(None);
    }
    let atomic_charge = match parse_number(fields[1]) {
        Ok(value) if value.is_finite() => value,
        Ok(_) => return Err(parse_error(line_number, "atomic charge is not finite")),
        Err(_) => return Ok(None),
    };
    let position = [
        parse_finite(fields[2], line_number, "x coordinate")?,
        parse_finite(fields[3], line_number, "y coordinate")?,
        parse_finite(fields[4], line_number, "z coordinate")?,
    ];
    Ok(Some(GmsAtom {
        name: fields[0].to_owned(),
        atomic_charge,
        position,
    }))
}

fn parse_finite(value: &str, line: usize, field: &str) -> Result<f64, GmsError> {
    let parsed = parse_number(value)
        .map_err(|error| parse_error(line, format!("invalid {field}: {error}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(parse_error(line, format!("{field} is not finite")))
    }
}

fn parse_number(value: &str) -> Result<f64, std::num::ParseFloatError> {
    value.replace(['d', 'D'], "e").parse::<f64>()
}

fn parse_error(line: usize, message: impl Into<String>) -> GmsError {
    GmsError::Parse {
        line,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data/gms")
            .join(name)
    }

    const OPTIMIZE: &str = "RUNTYP=OPTIMIZE\nTOTAL NUMBER OF ATOMS = 2\n ATOM ATOMIC COORDINATES (BOHR)\n ATOM CHARGE X Y Z\n --------\n H 1.0 0.0 0.0 0.0\n O 8.0 1.0 2.0 3.0\n1NSERCH= 0\n COORDINATES OF ALL ATOMS ARE (ANGS)\n ATOM CHARGE X Y Z\n --------\n H 1.0 0.1 0.2 0.3\n O 8.0 1.1 1.2 1.3\n1NSERCH= 1\n COORDINATES OF ALL ATOMS ARE (ANGS)\n ATOM CHARGE X Y Z\n --------\n H 1.0 0.4 0.5 0.6\n O 8.0 1.4 1.5 1.6\n";

    const SURFACE: &str = "RUNTYP=SURFACE\nTOTAL NUMBER OF ATOMS = 2\n ATOM ATOMIC COORDINATES (BOHR)\n ATOM CHARGE X Y Z\n --------\n H 1.0 0.0 0.0 0.0\n O 8.0 1.0 2.0 3.0\n COORD 1= 0.0 COORD 2= 0.0\n HAS ENERGY VALUE -1.0\n H 0.1 0.2 0.3\n O 1.1 1.2 1.3\n COORD 1= 0.1 COORD 2= 0.0\n HAS ENERGY VALUE -2.0\n H 0.4 0.5 0.6\n O 1.4 1.5 1.6\n";

    #[test]
    fn parses_optimization_frames_and_topology() {
        let file = GmsFile::from_str(OPTIMIZE).unwrap();
        assert_eq!(file.runtyp, "optimize");
        assert_eq!(file.n_atoms, 2);
        assert_eq!(file.atoms[1].name, "O");
        assert_eq!(file.atoms[1].atomic_charge, 8.0);
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.coordinates.frames[1].step, 1);
        assert_eq!(file.coordinates.frames[1].positions[0], [0.4, 0.5, 0.6]);
    }

    #[test]
    fn parses_surface_frames() {
        let file = GmsFile::from_str(SURFACE).unwrap();
        assert_eq!(file.runtyp, "surface");
        assert_eq!(file.n_frames(), 2);
        assert_eq!(file.coordinates.frames[0].positions[1], [1.1, 1.2, 1.3]);
    }

    #[test]
    fn builds_universe_with_guessed_elements_and_masses() {
        let universe = crate::core::Universe::from_gms_str(OPTIMIZE).unwrap();
        assert_eq!(universe.n_atoms(), 2);
        assert_eq!(universe.n_frames(), 2);
        assert_eq!(universe.topology.atoms[0].element.as_deref(), Some("H"));
        assert_eq!(universe.topology.atoms[1].charge, 8.0);
        assert!((universe.topology.atoms[1].mass - 15.999).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_unsupported_run_type() {
        let input = OPTIMIZE.replacen("OPTIMIZE", "ENERGY", 1);
        assert!(matches!(
            GmsFile::from_str(&input),
            Err(GmsError::InvalidStructure(message)) if message.contains("unsupported RUNTYP")
        ));
    }

    #[test]
    fn rejects_incomplete_frame() {
        let input = OPTIMIZE.replacen("O 8.0 1.4 1.5 1.6\n", "", 1);
        assert!(GmsFile::from_str(&input).is_err());
    }

    #[test]
    fn rejects_topology_count_that_truncates_records() {
        let input = OPTIMIZE.replacen("TOTAL NUMBER OF ATOMS = 2", "TOTAL NUMBER OF ATOMS = 1", 1);
        assert!(GmsFile::from_str(&input).is_err());
    }

    #[test]
    fn does_not_cross_optimization_step_boundaries() {
        let input = OPTIMIZE.replacen(
            "1NSERCH= 0\n COORDINATES OF ALL ATOMS ARE (ANGS)\n ATOM CHARGE X Y Z\n --------\n H 1.0 0.1 0.2 0.3\n O 8.0 1.1 1.2 1.3\n",
            "1NSERCH= 0\n",
            1,
        );
        assert!(GmsFile::from_str(&input).is_err());
    }

    #[test]
    fn reads_gzip_and_bzip2_bytes_without_filename_hints() {
        let gzip = std::fs::read(fixture("c1opt.gms.gz")).unwrap();
        let gzip_file = GmsFile::from_bytes(&gzip).unwrap();
        assert_eq!(gzip_file.n_frames(), 21);

        let mut plain = Vec::new();
        GzDecoder::new(Cursor::new(gzip))
            .read_to_end(&mut plain)
            .unwrap();
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        std::io::Write::write_all(&mut encoder, &plain).unwrap();
        let bzip = encoder.finish().unwrap();
        assert_eq!(GmsFile::from_bytes(&bzip).unwrap().n_atoms, 6);
    }

    #[test]
    fn parses_upstream_optimization_and_surface_fixtures() {
        let asymmetric = GmsFile::read_file(fixture("c1opt.gms.gz")).unwrap();
        assert_eq!(asymmetric.n_atoms, 6);
        assert_eq!(asymmetric.n_frames(), 21);
        assert_eq!(asymmetric.atoms[0].name, "O");
        assert_eq!(asymmetric.atoms[0].atomic_charge, 8.0);

        let symmetric = GmsFile::read_file(fixture("symopt.gms")).unwrap();
        assert_eq!(symmetric.n_atoms, 4);
        assert_eq!(symmetric.n_frames(), 8);
        assert_eq!(symmetric.atoms[0].name, "CARBON");

        let surface = GmsFile::read_file(fixture("surf2wat.gms")).unwrap();
        assert_eq!(surface.runtyp, "surface");
        assert_eq!(surface.n_atoms, 6);
        assert_eq!(surface.n_frames(), 10);
    }
}
