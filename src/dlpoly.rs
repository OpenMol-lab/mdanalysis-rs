//! Readers and writers for the text based DL_POLY CONFIG and HISTORY formats.
//!
//! CONFIG contains one coordinate frame. HISTORY is a sequence of frames,
//! each introduced by a timestep record. The levcfg header flag controls
//! whether velocities and forces follow each position record, while imcon
//! controls the optional three-vector unit cell.

use crate::mdamath::{triclinic_box, triclinic_vectors};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// A single DL_POLY coordinate frame.
#[derive(Clone, Debug, PartialEq)]
pub struct DlpolyFrame {
    pub step: usize,
    pub time: f64,
    pub positions: Vec<[f64; 3]>,
    pub names: Vec<String>,
    pub atom_ids: Vec<usize>,
    pub velocities: Option<Vec<[f64; 3]>>,
    pub forces: Option<Vec<[f64; 3]>>,
    pub dimensions: Option<[f64; 6]>,
}

impl DlpolyFrame {
    #[must_use]
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

/// A DL_POLY CONFIG document (one frame).
#[derive(Clone, Debug, PartialEq)]
pub struct DlpolyConfig {
    pub title: String,
    pub levcfg: i32,
    pub imcon: i32,
    pub megatm: usize,
    pub frame: DlpolyFrame,
}

/// A DL_POLY HISTORY document (one or more frames).
#[derive(Clone, Debug, PartialEq)]
pub struct DlpolyHistory {
    pub title: String,
    pub levcfg: i32,
    pub imcon: i32,
    pub n_atoms: usize,
    pub frames: Vec<DlpolyFrame>,
}

pub type ConfigFile = DlpolyConfig;
pub type HistoryFile = DlpolyHistory;

/// Errors produced while reading or writing DL_POLY text files.
#[derive(Debug)]
pub enum DlpolyError {
    Io(io::Error),
    Parse {
        format: &'static str,
        line: usize,
        message: String,
    },
    InvalidStructure(String),
}

impl fmt::Display for DlpolyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Parse {
                format,
                line,
                message,
            } => write!(formatter, "{format} parse error on line {line}: {message}"),
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid DL_POLY structure: {message}")
            }
        }
    }
}

impl std::error::Error for DlpolyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Parse { .. } | Self::InvalidStructure(_) => None,
        }
    }
}

impl From<io::Error> for DlpolyError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl DlpolyConfig {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, DlpolyError> {
        parse_config(input)
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, DlpolyError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, DlpolyError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    pub fn to_string(&self) -> Result<String, DlpolyError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output)
            .map_err(|error| DlpolyError::InvalidStructure(format!("output is not UTF-8: {error}")))
    }

    pub fn write<W: Write>(&self, writer: W) -> Result<(), DlpolyError> {
        write_config_document(self, writer)
    }

    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), DlpolyError> {
        self.write(File::create(path)?)
    }
}

impl std::str::FromStr for DlpolyConfig {
    type Err = DlpolyError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

impl DlpolyHistory {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> Result<Self, DlpolyError> {
        parse_history(input)
    }

    pub fn read<R: Read>(mut reader: R) -> Result<Self, DlpolyError> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Self::from_str(&input)
    }

    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, DlpolyError> {
        let input = crate::io_utils::read_text_file(path.as_ref())?;
        Self::from_str(&input)
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    pub fn to_string(&self) -> Result<String, DlpolyError> {
        let mut output = Vec::new();
        self.write(&mut output)?;
        String::from_utf8(output)
            .map_err(|error| DlpolyError::InvalidStructure(format!("output is not UTF-8: {error}")))
    }

    pub fn write<W: Write>(&self, writer: W) -> Result<(), DlpolyError> {
        write_history_document(self, writer)
    }

    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), DlpolyError> {
        self.write(File::create(path)?)
    }
}

impl std::str::FromStr for DlpolyHistory {
    type Err = DlpolyError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Self::from_str(input)
    }
}

pub fn read_config<P: AsRef<Path>>(path: P) -> Result<DlpolyConfig, DlpolyError> {
    DlpolyConfig::read_file(path)
}

pub fn write_config_file<P: AsRef<Path>>(
    path: P,
    config: &DlpolyConfig,
) -> Result<(), DlpolyError> {
    config.write_file(path)
}

pub fn write_config<P: AsRef<Path>>(path: P, config: &DlpolyConfig) -> Result<(), DlpolyError> {
    write_config_file(path, config)
}

pub fn read_history<P: AsRef<Path>>(path: P) -> Result<DlpolyHistory, DlpolyError> {
    DlpolyHistory::read_file(path)
}

pub fn write_history_file<P: AsRef<Path>>(
    path: P,
    history: &DlpolyHistory,
) -> Result<(), DlpolyError> {
    history.write_file(path)
}

pub fn write_history<P: AsRef<Path>>(path: P, history: &DlpolyHistory) -> Result<(), DlpolyError> {
    write_history_file(path, history)
}

impl crate::core::Universe {
    pub fn from_dlpoly_config(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dlpoly_config_file(DlpolyConfig::read_file(path)?)
    }

    pub fn from_dlpoly_config_str(input: &str) -> crate::Result<Self> {
        Self::from_dlpoly_config_file(DlpolyConfig::from_str(input)?)
    }

    pub fn from_dlpoly_config_file(config: DlpolyConfig) -> crate::Result<Self> {
        universe_from_frames(vec![config.frame])
    }

    pub fn from_dlpoly_history(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dlpoly_history_file(DlpolyHistory::read_file(path)?)
    }

    pub fn from_dlpoly_history_str(input: &str) -> crate::Result<Self> {
        Self::from_dlpoly_history_file(DlpolyHistory::from_str(input)?)
    }

    pub fn from_dlpoly_history_file(history: DlpolyHistory) -> crate::Result<Self> {
        universe_from_frames(history.frames)
    }

    pub fn from_config(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dlpoly_config(path)
    }

    pub fn from_config_str(input: &str) -> crate::Result<Self> {
        Self::from_dlpoly_config_str(input)
    }

    pub fn from_history(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dlpoly_history(path)
    }

    pub fn from_history_str(input: &str) -> crate::Result<Self> {
        Self::from_dlpoly_history_str(input)
    }
}

fn universe_from_frames(frames: Vec<DlpolyFrame>) -> crate::Result<crate::core::Universe> {
    let first = frames.first().ok_or_else(|| {
        crate::Error::InvalidInput("DL_POLY file contains no coordinate frames".to_owned())
    })?;
    let count = first.positions.len();
    if count == 0 {
        return Err(crate::Error::InvalidInput(
            "DL_POLY file contains no atoms".to_owned(),
        ));
    }
    if first.names.len() != count || first.atom_ids.len() != count {
        return Err(crate::Error::InvalidInput(
            "DL_POLY atom metadata is inconsistent".to_owned(),
        ));
    }
    let atoms = first
        .positions
        .iter()
        .enumerate()
        .map(|(index, position)| {
            let name = &first.names[index];
            let mut atom = crate::core::Atom::new(index, name.clone(), *position);
            atom.atom_type = Some(name.clone());
            atom.element = crate::guesser::guess_element(name, None, None).ok();
            atom.mass = crate::guesser::guess_atom_mass(name);
            atom.resid = 1;
            atom.resname = "SYSTEM".to_owned();
            atom
        })
        .collect();
    let topology = crate::core::Topology::new(atoms);
    let mut result_frames = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.positions.len() != count
            || frame.names.len() != count
            || frame.atom_ids.len() != count
        {
            return Err(crate::Error::InvalidInput(
                "DL_POLY frames contain inconsistent atom counts".to_owned(),
            ));
        }
        let mut result = crate::core::Frame::new(frame.positions);
        result.velocities = frame.velocities;
        result.forces = frame.forces;
        result.dimensions = frame.dimensions;
        result.step = frame.step;
        result.time = frame.time;
        result_frames.push(result);
    }
    Ok(crate::core::Universe {
        topology,
        trajectory: crate::core::Trajectory::new(result_frames),
    })
}

fn parse_config(input: &str) -> Result<DlpolyConfig, DlpolyError> {
    let lines = input.lines().collect::<Vec<_>>();
    let title = lines
        .first()
        .map(|line| line.trim_end().to_owned())
        .ok_or_else(|| parse_error("CONFIG", 1, "missing title line"))?;
    let (levcfg, imcon, megatm) = parse_header(
        lines
            .get(1)
            .copied()
            .ok_or_else(|| parse_error("CONFIG", 2, "missing header line"))?,
        "CONFIG",
        2,
    )?;
    let mut cursor = 2;
    let dimensions = if imcon != 0 {
        let vectors = read_cell(&lines, &mut cursor, "CONFIG")?;
        Some(triclinic_box(vectors))
    } else {
        None
    };
    let frame = parse_atom_records(&lines, &mut cursor, None, levcfg, dimensions, "CONFIG")?;
    let config = DlpolyConfig {
        title,
        levcfg,
        imcon,
        megatm,
        frame,
    };
    validate_config(&config)?;
    Ok(config)
}

fn parse_history(input: &str) -> Result<DlpolyHistory, DlpolyError> {
    let lines = input.lines().collect::<Vec<_>>();
    let title = lines
        .first()
        .map(|line| line.trim_end().to_owned())
        .ok_or_else(|| parse_error("HISTORY", 1, "missing title line"))?;
    let (levcfg, imcon, n_atoms) = parse_header(
        lines
            .get(1)
            .copied()
            .ok_or_else(|| parse_error("HISTORY", 2, "missing header line"))?,
        "HISTORY",
        2,
    )?;
    let mut cursor = 2;
    let mut frames = Vec::new();
    while let Some((line_number, line)) = next_nonempty(&lines, &mut cursor) {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields
            .first()
            .is_none_or(|field| !field.eq_ignore_ascii_case("timestep"))
        {
            return Err(parse_error(
                "HISTORY",
                line_number,
                "expected a timestep record",
            ));
        }
        let step = fields
            .get(1)
            .ok_or_else(|| parse_error("HISTORY", line_number, "timestep is missing its step"))?
            .parse::<usize>()
            .map_err(|error| {
                parse_error("HISTORY", line_number, format!("invalid timestep: {error}"))
            })?;
        let declared = fields
            .get(2)
            .ok_or_else(|| parse_error("HISTORY", line_number, "timestep is missing atom count"))?
            .parse::<usize>()
            .map_err(|error| {
                parse_error(
                    "HISTORY",
                    line_number,
                    format!("invalid atom count: {error}"),
                )
            })?;
        if declared != n_atoms {
            return Err(parse_error(
                "HISTORY",
                line_number,
                format!("timestep declares {declared} atoms; header declares {n_atoms}"),
            ));
        }
        let time = fields
            .iter()
            .rev()
            .find_map(|field| field.parse::<f64>().ok())
            .unwrap_or(step as f64);
        let has_cell = imcon != 0 || lines.get(cursor).is_some_and(|line| is_vector_line(line));
        let dimensions = if has_cell {
            Some(triclinic_box(read_cell(&lines, &mut cursor, "HISTORY")?))
        } else {
            None
        };
        let frame = parse_atom_records(
            &lines,
            &mut cursor,
            Some((step, time)),
            levcfg,
            dimensions,
            "HISTORY",
        )?;
        if frame.n_atoms() != n_atoms {
            return Err(parse_error(
                "HISTORY",
                line_number,
                format!(
                    "frame contains {} atoms; expected {n_atoms}",
                    frame.n_atoms()
                ),
            ));
        }
        frames.push(frame);
    }
    if frames.is_empty() {
        return Err(DlpolyError::InvalidStructure(
            "HISTORY contains no timestep frames".to_owned(),
        ));
    }
    let history = DlpolyHistory {
        title,
        levcfg,
        imcon,
        n_atoms,
        frames,
    };
    validate_history(&history)?;
    Ok(history)
}

fn parse_header(
    line: &str,
    format: &'static str,
    line_number: usize,
) -> Result<(i32, i32, usize), DlpolyError> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 {
        return Err(parse_error(
            format,
            line_number,
            "header requires levcfg, imcon, and atom count",
        ));
    }
    let levcfg = fields[0]
        .parse::<i32>()
        .map_err(|error| parse_error(format, line_number, format!("invalid levcfg: {error}")))?;
    let imcon = fields[1]
        .parse::<i32>()
        .map_err(|error| parse_error(format, line_number, format!("invalid imcon: {error}")))?;
    let count = fields[2].parse::<usize>().map_err(|error| {
        parse_error(format, line_number, format!("invalid atom count: {error}"))
    })?;
    if !(0..=2).contains(&levcfg) {
        return Err(parse_error(
            format,
            line_number,
            "levcfg must be 0, 1, or 2",
        ));
    }
    Ok((levcfg, imcon, count))
}

fn read_cell(
    lines: &[&str],
    cursor: &mut usize,
    format: &'static str,
) -> Result<[[f64; 3]; 3], DlpolyError> {
    let mut vectors = [[0.0; 3]; 3];
    for vector in &mut vectors {
        let line_number = *cursor + 1;
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            parse_error(format, line_number, "file ended while reading unit cell")
        })?;
        *cursor += 1;
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 3 {
            return Err(parse_error(
                format,
                line_number,
                "cell vector requires three values",
            ));
        }
        for (component, field) in vector.iter_mut().zip(fields.iter().take(3)) {
            *component = parse_finite(field, format, line_number, "cell component")?;
        }
    }
    Ok(vectors)
}

fn is_vector_line(line: &str) -> bool {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    fields.len() >= 3
        && fields
            .iter()
            .take(3)
            .all(|field| field.parse::<f64>().is_ok())
}

fn parse_atom_records(
    lines: &[&str],
    cursor: &mut usize,
    step_time: Option<(usize, f64)>,
    levcfg: i32,
    dimensions: Option<[f64; 6]>,
    format: &'static str,
) -> Result<DlpolyFrame, DlpolyError> {
    let mut records = Vec::new();
    while *cursor < lines.len() {
        let line_number = *cursor + 1;
        let raw = lines[*cursor];
        if raw.trim().is_empty() {
            *cursor += 1;
            break;
        }
        if format == "HISTORY" && raw.trim_start().starts_with("timestep") {
            break;
        }
        *cursor += 1;
        let (name, id) = parse_atom_header(raw, format, line_number)?;
        let position = read_vector_line(lines, cursor, format, "position")?;
        let velocity = if levcfg > 0 {
            Some(read_vector_line(lines, cursor, format, "velocity")?)
        } else {
            None
        };
        let force = if levcfg == 2 {
            Some(read_vector_line(lines, cursor, format, "force")?)
        } else {
            None
        };
        records.push((name, id, position, velocity, force));
    }
    if records.is_empty() {
        return Err(DlpolyError::InvalidStructure(format!(
            "{format} contains no atom records"
        )));
    }
    let ids_present = records.iter().all(|record| record.1.is_some());
    if ids_present {
        records.sort_by_key(|record| record.1.expect("checked above"));
    }
    let names = records
        .iter()
        .map(|record| record.0.clone())
        .collect::<Vec<_>>();
    let atom_ids = records
        .iter()
        .enumerate()
        .map(|(index, record)| record.1.unwrap_or(index + 1))
        .collect::<Vec<_>>();
    let positions = records.iter().map(|record| record.2).collect::<Vec<_>>();
    let velocities = (levcfg > 0).then(|| {
        records
            .iter()
            .map(|record| record.3.expect("levcfg supplies velocities"))
            .collect()
    });
    let forces = (levcfg == 2).then(|| {
        records
            .iter()
            .map(|record| record.4.expect("levcfg supplies forces"))
            .collect()
    });
    Ok(DlpolyFrame {
        step: step_time.map_or(0, |value| value.0),
        time: step_time.map_or(0.0, |value| value.1),
        positions,
        names,
        atom_ids,
        velocities,
        forces,
        dimensions,
    })
}

fn parse_atom_header(
    line: &str,
    format: &'static str,
    line_number: usize,
) -> Result<(String, Option<usize>), DlpolyError> {
    let name = line
        .get(..8)
        .unwrap_or(line)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned();
    if name.is_empty() {
        return Err(parse_error(format, line_number, "atom name is empty"));
    }
    let id = line
        .get(8..)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok());
    let id = id.or_else(|| {
        line.split_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<usize>().ok())
    });
    Ok((name, id))
}

fn read_vector_line(
    lines: &[&str],
    cursor: &mut usize,
    format: &'static str,
    field: &str,
) -> Result<[f64; 3], DlpolyError> {
    let line_number = *cursor + 1;
    let line = lines
        .get(*cursor)
        .copied()
        .ok_or_else(|| parse_error(format, line_number, format!("missing {field} vector")))?;
    *cursor += 1;
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 {
        return Err(parse_error(
            format,
            line_number,
            format!("{field} vector requires three values"),
        ));
    }
    Ok([
        parse_finite(fields[0], format, line_number, field)?,
        parse_finite(fields[1], format, line_number, field)?,
        parse_finite(fields[2], format, line_number, field)?,
    ])
}

fn next_nonempty<'a>(lines: &'a [&str], cursor: &mut usize) -> Option<(usize, &'a str)> {
    while *cursor < lines.len() {
        let line_number = *cursor + 1;
        let line = lines[*cursor];
        *cursor += 1;
        if !line.trim().is_empty() {
            return Some((line_number, line));
        }
    }
    None
}

fn parse_finite(
    value: &str,
    format: &'static str,
    line: usize,
    field: &str,
) -> Result<f64, DlpolyError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| parse_error(format, line, format!("invalid {field}: {error}")))?;
    if !parsed.is_finite() {
        return Err(parse_error(format, line, format!("{field} must be finite")));
    }
    Ok(parsed)
}

fn parse_error(format: &'static str, line: usize, message: impl Into<String>) -> DlpolyError {
    DlpolyError::Parse {
        format,
        line,
        message: message.into(),
    }
}

fn validate_config(config: &DlpolyConfig) -> Result<(), DlpolyError> {
    if config.frame.n_atoms() == 0 {
        return Err(DlpolyError::InvalidStructure(
            "CONFIG contains no atoms".to_owned(),
        ));
    }
    validate_frame(&config.frame, config.levcfg)
}

fn validate_history(history: &DlpolyHistory) -> Result<(), DlpolyError> {
    if history.n_atoms == 0 || history.frames.is_empty() {
        return Err(DlpolyError::InvalidStructure(
            "HISTORY must contain atoms and frames".to_owned(),
        ));
    }
    for frame in &history.frames {
        if frame.n_atoms() != history.n_atoms {
            return Err(DlpolyError::InvalidStructure(
                "HISTORY frame atom counts differ".to_owned(),
            ));
        }
        validate_frame(frame, history.levcfg)?;
    }
    Ok(())
}

fn validate_frame(frame: &DlpolyFrame, levcfg: i32) -> Result<(), DlpolyError> {
    let count = frame.positions.len();
    if frame.names.len() != count || frame.atom_ids.len() != count {
        return Err(DlpolyError::InvalidStructure(
            "atom metadata lengths do not match positions".to_owned(),
        ));
    }
    if frame
        .positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(DlpolyError::InvalidStructure(
            "positions must be finite".to_owned(),
        ));
    }
    if levcfg > 0 {
        let velocities = frame.velocities.as_ref().ok_or_else(|| {
            DlpolyError::InvalidStructure("levcfg requires velocities".to_owned())
        })?;
        if velocities.len() != count || velocities.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(DlpolyError::InvalidStructure(
                "velocity data is inconsistent".to_owned(),
            ));
        }
    }
    if levcfg == 2 {
        let forces = frame
            .forces
            .as_ref()
            .ok_or_else(|| DlpolyError::InvalidStructure("levcfg=2 requires forces".to_owned()))?;
        if forces.len() != count || forces.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DlpolyError::InvalidStructure(
                "force data is inconsistent".to_owned(),
            ));
        }
    }
    if let Some(dimensions) = frame.dimensions
        && (dimensions[..3]
            .iter()
            .any(|value| *value <= 0.0 || !value.is_finite())
            || dimensions[3..]
                .iter()
                .any(|value| *value <= 0.0 || *value >= 180.0 || !value.is_finite()))
    {
        return Err(DlpolyError::InvalidStructure(
            "unit cell dimensions are invalid".to_owned(),
        ));
    }
    Ok(())
}

fn write_config_document<W: Write>(
    config: &DlpolyConfig,
    mut writer: W,
) -> Result<(), DlpolyError> {
    validate_config(config)?;
    writeln!(writer, "{}", config.title)?;
    writeln!(
        writer,
        "{} {} {}",
        config.levcfg,
        config.imcon,
        if config.megatm == 0 {
            config.frame.n_atoms()
        } else {
            config.megatm
        }
    )?;
    write_frame(&config.frame, config.levcfg, config.imcon, &mut writer)?;
    Ok(())
}

fn write_history_document<W: Write>(
    history: &DlpolyHistory,
    mut writer: W,
) -> Result<(), DlpolyError> {
    validate_history(history)?;
    writeln!(writer, "{}", history.title)?;
    writeln!(
        writer,
        "{} {} {}",
        history.levcfg, history.imcon, history.n_atoms
    )?;
    for frame in &history.frames {
        writeln!(
            writer,
            "timestep {} {} 0 0 {:.12e}",
            frame.step, history.n_atoms, frame.time
        )?;
        write_frame(frame, history.levcfg, history.imcon, &mut writer)?;
    }
    Ok(())
}

fn write_frame<W: Write>(
    frame: &DlpolyFrame,
    levcfg: i32,
    imcon: i32,
    writer: &mut W,
) -> Result<(), DlpolyError> {
    if imcon != 0 || frame.dimensions.is_some() {
        let dimensions = frame.dimensions.ok_or_else(|| {
            DlpolyError::InvalidStructure("imcon requires unit cell dimensions".to_owned())
        })?;
        for vector in triclinic_vectors(dimensions) {
            writeln!(
                writer,
                "{:.12e} {:.12e} {:.12e}",
                vector[0], vector[1], vector[2]
            )?;
        }
    }
    for (index, position) in frame.positions.iter().enumerate() {
        writeln!(
            writer,
            "{:<8} {:>8}",
            frame.names[index], frame.atom_ids[index]
        )?;
        writeln!(
            writer,
            "{:.12e} {:.12e} {:.12e}",
            position[0], position[1], position[2]
        )?;
        if levcfg > 0 {
            let velocity = frame.velocities.as_ref().expect("validated")[index];
            writeln!(
                writer,
                "{:.12e} {:.12e} {:.12e}",
                velocity[0], velocity[1], velocity[2]
            )?;
        }
        if levcfg == 2 {
            let force = frame.forces.as_ref().expect("validated")[index];
            writeln!(
                writer,
                "{:.12e} {:.12e} {:.12e}",
                force[0], force[1], force[2]
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_MINIMAL: &str = concat!(
        "minimal\n",
        "0 0 3 0.005\n",
        "C\n",
        "0 0 0\n",
        "B\n",
        "1 0 0\n",
        "A\n",
        "0 1 0\n",
    );

    const CONFIG_ORDER: &str = concat!(
        "ordered\n",
        "2 3 3 0.005\n",
        "3 0 0\n",
        "0 3 0\n",
        "0 0 3\n",
        "A       6\n",
        "0 0 0\n",
        "1 2 3\n",
        "4 5 6\n",
        "B       5\n",
        "1 0 0\n",
        "7 8 9\n",
        "10 11 12\n",
        "C       4\n",
        "2 0 0\n",
        "13 14 15\n",
        "16 17 18\n",
    );

    const HISTORY: &str = concat!(
        "history\n",
        "2 3 3 3\n",
        "timestep 1 3 0 1 0.005\n",
        "3 0 0\n",
        "0 3 0\n",
        "0 0 3\n",
        "A       3\n",
        "0 0 0\n",
        "1 2 3\n",
        "4 5 6\n",
        "B       2\n",
        "1 0 0\n",
        "7 8 9\n",
        "10 11 12\n",
        "C       1\n",
        "2 0 0\n",
        "13 14 15\n",
        "16 17 18\n",
        "timestep 2 3 0 1 0.010\n",
        "3 0 0\n",
        "0 3 0\n",
        "0 0 3\n",
        "A       1\n",
        "0.1 0 0\n",
        "1 2 3\n",
        "4 5 6\n",
        "B       2\n",
        "1.1 0 0\n",
        "7 8 9\n",
        "10 11 12\n",
        "C       3\n",
        "2.1 0 0\n",
        "13 14 15\n",
        "16 17 18\n",
    );

    #[test]
    fn config_without_ids_and_ordered_config_parse() {
        let minimal = DlpolyConfig::from_str(CONFIG_MINIMAL).unwrap();
        assert_eq!(minimal.frame.names, vec!["C", "B", "A"]);
        assert_eq!(minimal.frame.atom_ids, vec![1, 2, 3]);
        let ordered = DlpolyConfig::from_str(CONFIG_ORDER).unwrap();
        assert_eq!(ordered.frame.names, vec!["C", "B", "A"]);
        assert_eq!(ordered.frame.atom_ids, vec![4, 5, 6]);
        assert_eq!(
            ordered.frame.velocities.as_ref().unwrap()[0],
            [13.0, 14.0, 15.0]
        );
        let compact = "compact\n0 0 1\nC 42\n0 0 0\n";
        let compact = DlpolyConfig::from_str(compact).unwrap();
        assert_eq!(compact.frame.names, vec!["C"]);
        assert_eq!(compact.frame.atom_ids, vec![42]);
    }

    #[test]
    fn history_reads_multiple_frames_and_sorts_ids() {
        let history = DlpolyHistory::from_str(HISTORY).unwrap();
        assert_eq!(history.n_frames(), 2);
        assert_eq!(history.frames[0].names, vec!["C", "B", "A"]);
        assert_eq!(history.frames[0].step, 1);
        assert_eq!(history.frames[1].positions[0], [0.1, 0.0, 0.0]);
        let universe = crate::core::Universe::from_dlpoly_history_file(history).unwrap();
        assert_eq!(universe.n_atoms(), 3);
        assert_eq!(universe.trajectory.n_frames(), 2);
        assert_eq!(
            universe.trajectory.frames[0].forces,
            Some(vec![
                [16.0, 17.0, 18.0],
                [10.0, 11.0, 12.0],
                [4.0, 5.0, 6.0],
            ])
        );
    }

    #[test]
    fn config_round_trip_preserves_coordinates_and_cell() {
        let config = DlpolyConfig::from_str(CONFIG_ORDER).unwrap();
        let reparsed = DlpolyConfig::from_str(&config.to_string().unwrap()).unwrap();
        assert_eq!(reparsed.frame.names, config.frame.names);
        assert_eq!(reparsed.frame.atom_ids, config.frame.atom_ids);
        assert_eq!(reparsed.frame.positions, config.frame.positions);
        assert_eq!(reparsed.frame.dimensions, config.frame.dimensions);
    }

    #[test]
    fn malformed_history_eof_is_rejected() {
        let input = "history\n0 0 1\ntimestep 1 1 0 1 0.1\nC 1\n";
        assert!(matches!(
            DlpolyHistory::from_str(input),
            Err(DlpolyError::Parse {
                format: "HISTORY",
                ..
            }) | Err(DlpolyError::InvalidStructure(_))
        ));
    }
}
