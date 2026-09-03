//! Macromolecular Transmission Format (MMTF) topology and coordinates.
//!
//! MMTF stores a MessagePack map whose large arrays are wrapped in a small
//! binary codec header.  This module implements the codecs used by the MMTF
//! specification directly, so callers do not need a Python MMTF dependency.

use crate::core::{Atom, Bond, Frame, Residue, Segment, Topology, Trajectory, Universe};
use flate2::read::GzDecoder;
use rmpv::Value;
use std::collections::HashSet;
use std::fmt;
use std::io::{Cursor, Read};
use std::path::Path;

/// A reusable MMTF chemical component definition.
#[derive(Clone, Debug, PartialEq)]
pub struct MmtfGroup {
    pub name: String,
    pub atom_names: Vec<String>,
    pub elements: Vec<String>,
    pub formal_charges: Vec<i32>,
    pub bond_atoms: Vec<(usize, usize)>,
    pub bond_orders: Vec<u8>,
}

/// One global MMTF bond (atom indices are zero-based).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmtfBond {
    pub atom1: usize,
    pub atom2: usize,
    pub order: Option<u8>,
}

/// Parsed MMTF data, including topology metadata and one coordinate set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MmtfFile {
    pub num_atoms: usize,
    pub num_bonds: usize,
    pub num_groups: usize,
    pub num_chains: usize,
    pub num_models: usize,
    pub chains_per_model: Vec<usize>,
    pub groups_per_chain: Vec<usize>,
    pub chain_names: Vec<String>,
    pub chain_ids: Vec<String>,
    pub groups: Vec<MmtfGroup>,
    pub group_types: Vec<usize>,
    pub group_ids: Vec<i32>,
    pub sequence_indices: Vec<i32>,
    pub atom_ids: Vec<i32>,
    pub positions: Vec<[f64; 3]>,
    pub b_factors: Vec<f64>,
    pub occupancies: Vec<f64>,
    pub alt_locs: Vec<String>,
    pub insertion_codes: Vec<String>,
    pub bonds: Vec<MmtfBond>,
    pub unit_cell: Option<[f64; 6]>,
    pub space_group: Option<String>,
}

/// Errors encountered while parsing or validating MMTF data.
#[derive(Debug)]
pub enum MmtfError {
    Io(std::io::Error),
    MessagePack(String),
    InvalidStructure(String),
}

impl fmt::Display for MmtfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "MMTF I/O error: {error}"),
            Self::MessagePack(error) => write!(formatter, "MMTF MessagePack error: {error}"),
            Self::InvalidStructure(error) => write!(formatter, "invalid MMTF structure: {error}"),
        }
    }
}

impl std::error::Error for MmtfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::MessagePack(_) | Self::InvalidStructure(_) => None,
        }
    }
}

impl From<std::io::Error> for MmtfError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl MmtfFile {
    /// Read an MMTF file, transparently accepting gzip-compressed input.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, MmtfError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        Self::from_bytes_maybe_gzip(&bytes)
    }

    /// Alias for [`MmtfFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MmtfError> {
        Self::read_file(path)
    }

    /// Parse an uncompressed MMTF MessagePack document.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MmtfError> {
        let mut input = Cursor::new(bytes);
        let value = rmpv::decode::read_value(&mut input)
            .map_err(|error| MmtfError::MessagePack(error.to_string()))?;
        let map = match value {
            Value::Map(map) => map,
            _ => return Err(invalid("top-level MessagePack value must be a map")),
        };
        parse_map(&map)
    }

    fn from_bytes_maybe_gzip(bytes: &[u8]) -> Result<Self, MmtfError> {
        if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut decoder = GzDecoder::new(bytes);
            let mut decoded = Vec::new();
            decoder.read_to_end(&mut decoded)?;
            Self::from_bytes(&decoded)
        } else {
            Self::from_bytes(bytes)
        }
    }

    /// Return the unit cell as `[a, b, c, alpha, beta, gamma]`, if present.
    #[must_use]
    pub const fn dimensions(&self) -> Option<[f64; 6]> {
        self.unit_cell
    }
}

/// Read an MMTF file from a path, accepting both plain and `.gz` input.
pub fn read_mmtf(path: impl AsRef<Path>) -> Result<MmtfFile, MmtfError> {
    MmtfFile::read_file(path)
}

fn invalid(message: impl Into<String>) -> MmtfError {
    MmtfError::InvalidStructure(message.into())
}

fn map_value<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find_map(|(name, value)| (name.as_str() == Some(key)).then_some(value))
}

fn required<'a>(map: &'a [(Value, Value)], key: &str) -> Result<&'a Value, MmtfError> {
    map_value(map, key).ok_or_else(|| invalid(format!("missing required field {key:?}")))
}

fn value_i64(value: &Value, field: &str) -> Result<i64, MmtfError> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .ok_or_else(|| invalid(format!("field {field:?} is not an integer")))
}

fn value_usize(value: &Value, field: &str) -> Result<usize, MmtfError> {
    let number = value_i64(value, field)?;
    usize::try_from(number).map_err(|_| {
        invalid(format!(
            "field {field:?} contains a negative or oversized integer"
        ))
    })
}

fn value_f64(value: &Value, field: &str) -> Result<f64, MmtfError> {
    let number = match value {
        Value::F32(number) => f64::from(*number),
        Value::F64(number) => *number,
        Value::Integer(_) => value_i64(value, field)? as f64,
        _ => return Err(invalid(format!("field {field:?} is not numeric"))),
    };
    number
        .is_finite()
        .then_some(number)
        .ok_or_else(|| invalid(format!("field {field:?} contains a non-finite value")))
}

fn value_string(value: &Value, field: &str) -> Result<String, MmtfError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("field {field:?} is not a string")))
}

fn array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], MmtfError> {
    match value {
        Value::Array(values) => Ok(values),
        _ => Err(invalid(format!("field {field:?} is not an array"))),
    }
}

fn integer_array(value: &Value, field: &str) -> Result<Vec<i64>, MmtfError> {
    array(value, field)?
        .iter()
        .map(|value| value_i64(value, field))
        .collect()
}

fn parse_map(map: &[(Value, Value)]) -> Result<MmtfFile, MmtfError> {
    let num_atoms = value_usize(required(map, "numAtoms")?, "numAtoms")?;
    let num_bonds = value_usize(required(map, "numBonds")?, "numBonds")?;
    let num_groups = value_usize(required(map, "numGroups")?, "numGroups")?;
    let num_chains = value_usize(required(map, "numChains")?, "numChains")?;
    let num_models = value_usize(required(map, "numModels")?, "numModels")?;
    if num_models == 0 {
        return Err(invalid("numModels must be positive"));
    }

    let chains_per_model = integer_array(required(map, "chainsPerModel")?, "chainsPerModel")?
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| invalid("chainsPerModel contains a negative value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if chains_per_model.len() != num_models || chains_per_model.iter().sum::<usize>() != num_chains
    {
        return Err(invalid(format!(
            "chainsPerModel has length/sum {:?}, expected {num_models}/{num_chains}",
            chains_per_model
        )));
    }
    let groups_per_chain = integer_array(required(map, "groupsPerChain")?, "groupsPerChain")?
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| invalid("groupsPerChain contains a negative value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if groups_per_chain.len() != num_chains || groups_per_chain.iter().sum::<usize>() != num_groups
    {
        return Err(invalid(format!(
            "groupsPerChain has length/sum {:?}, expected {num_chains}/{num_groups}",
            groups_per_chain
        )));
    }

    let chain_ids = decode_strings(required(map, "chainIdList")?, "chainIdList")?;
    if chain_ids.len() != num_chains {
        return Err(invalid(format!(
            "chainIdList has {}, expected {num_chains} entries",
            chain_ids.len()
        )));
    }
    let chain_names = match map_value(map, "chainNameList") {
        Some(value) => {
            let names = decode_strings(value, "chainNameList")?;
            if names.len() != num_chains {
                return Err(invalid(format!(
                    "chainNameList has {}, expected {num_chains} entries",
                    names.len()
                )));
            }
            names
        }
        None => vec![String::new(); num_chains],
    };

    let group_values = array(required(map, "groupList")?, "groupList")?;
    let groups = group_values
        .iter()
        .enumerate()
        .map(|(index, value)| parse_group(value, index))
        .collect::<Result<Vec<_>, _>>()?;
    let group_types = decode_usize_array(required(map, "groupTypeList")?, "groupTypeList")?;
    if group_types.len() != num_groups {
        return Err(invalid(format!(
            "groupTypeList has {}, expected {num_groups} entries",
            group_types.len()
        )));
    }
    if group_types.iter().any(|&index| index >= groups.len()) {
        return Err(invalid(
            "groupTypeList references a missing group definition",
        ));
    }
    let group_ids = decode_i32_array(required(map, "groupIdList")?, "groupIdList")?;
    if group_ids.len() != num_groups {
        return Err(invalid(format!(
            "groupIdList has {}, expected {num_groups} entries",
            group_ids.len()
        )));
    }
    let sequence_indices = match map_value(map, "sequenceIndexList") {
        Some(value) => {
            let values = decode_i32_array(value, "sequenceIndexList")?;
            if values.len() != num_groups {
                return Err(invalid(format!(
                    "sequenceIndexList has {}, expected {num_groups} entries",
                    values.len()
                )));
            }
            values
        }
        None => vec![0; num_groups],
    };

    let x = decode_f64_array(required(map, "xCoordList")?, "xCoordList")?;
    let y = decode_f64_array(required(map, "yCoordList")?, "yCoordList")?;
    let z = decode_f64_array(required(map, "zCoordList")?, "zCoordList")?;
    if x.len() != num_atoms || y.len() != num_atoms || z.len() != num_atoms {
        return Err(invalid(format!(
            "coordinate lengths are {}/{}/{}, expected {num_atoms}",
            x.len(),
            y.len(),
            z.len()
        )));
    }
    let positions = x
        .into_iter()
        .zip(y)
        .zip(z)
        .map(|((x, y), z)| [x, y, z])
        .collect::<Vec<_>>();

    let atom_ids = match map_value(map, "atomIdList") {
        Some(value) => {
            let values = decode_i32_array(value, "atomIdList")?;
            if values.len() != num_atoms {
                return Err(invalid(format!(
                    "atomIdList has {}, expected {num_atoms} entries",
                    values.len()
                )));
            }
            values
        }
        None => (1..=num_atoms)
            .map(|value| i32::try_from(value).unwrap_or(i32::MAX))
            .collect(),
    };
    let b_factors = decode_optional_f64(map, "bFactorList", num_atoms, 0.0)?;
    let occupancies = decode_optional_f64(map, "occupancyList", num_atoms, 1.0)?;
    let alt_locs = decode_optional_strings(map, "altLocList", num_atoms)?;
    let insertion_codes = decode_optional_strings(map, "insCodeList", num_groups)?;

    let mut bonds = Vec::new();
    let mut atom_offset = 0usize;
    for &group_type in &group_types {
        let group = &groups[group_type];
        if group.bond_atoms.len() != group.bond_orders.len() {
            return Err(invalid("group bond atom/order lengths differ"));
        }
        for (&(atom1, atom2), &order) in group.bond_atoms.iter().zip(&group.bond_orders) {
            let atom1 = atom1
                .checked_add(atom_offset)
                .ok_or_else(|| invalid("group bond index overflow"))?;
            let atom2 = atom2
                .checked_add(atom_offset)
                .ok_or_else(|| invalid("group bond index overflow"))?;
            bonds.push(MmtfBond {
                atom1,
                atom2,
                order: Some(order),
            });
        }
        atom_offset = atom_offset
            .checked_add(group.atom_names.len())
            .ok_or_else(|| invalid("atom count overflow"))?;
    }
    if atom_offset != num_atoms {
        return Err(invalid(format!(
            "group atom count is {atom_offset}, expected {num_atoms}"
        )));
    }

    let external_atoms = match map_value(map, "bondAtomList") {
        Some(value) => decode_usize_array(value, "bondAtomList")?,
        None => Vec::new(),
    };
    let external_orders = match map_value(map, "bondOrderList") {
        Some(value) => decode_u8_array(value, "bondOrderList")?,
        None => Vec::new(),
    };
    if external_atoms.len() != external_orders.len().saturating_mul(2) {
        return Err(invalid(format!(
            "bondAtomList has {}, bondOrderList has {} entries",
            external_atoms.len(),
            external_orders.len()
        )));
    }
    for (pair, &order) in external_atoms
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&external_orders)
    {
        bonds.push(MmtfBond {
            atom1: pair[0],
            atom2: pair[1],
            order: Some(order),
        });
    }
    validate_bonds(&bonds, num_atoms, num_bonds)?;

    let unit_cell = parse_unit_cell(map)?;
    let space_group = map_value(map, "spaceGroup")
        .map(|value| value_string(value, "spaceGroup"))
        .transpose()?;
    Ok(MmtfFile {
        num_atoms,
        num_bonds,
        num_groups,
        num_chains,
        num_models,
        chains_per_model,
        groups_per_chain,
        chain_names,
        chain_ids,
        groups,
        group_types,
        group_ids,
        sequence_indices,
        atom_ids,
        positions,
        b_factors,
        occupancies,
        alt_locs,
        insertion_codes,
        bonds,
        unit_cell,
        space_group,
    })
}

fn parse_group(value: &Value, index: usize) -> Result<MmtfGroup, MmtfError> {
    let map = match value {
        Value::Map(map) => map,
        _ => return Err(invalid(format!("group definition {index} is not a map"))),
    };
    let name = value_string(required(map, "groupName")?, "groupName")?;
    let atom_names = array(required(map, "atomNameList")?, "atomNameList")?
        .iter()
        .enumerate()
        .map(|(atom, value)| value_string(value, &format!("group {index} atom {atom} name")))
        .collect::<Result<Vec<_>, _>>()?;
    let elements = array(required(map, "elementList")?, "elementList")?
        .iter()
        .enumerate()
        .map(|(atom, value)| value_string(value, &format!("group {index} atom {atom} element")))
        .collect::<Result<Vec<_>, _>>()?;
    if elements.len() != atom_names.len() {
        return Err(invalid(format!(
            "group {index} atom name/element lengths differ"
        )));
    }
    let formal_charges = match map_value(map, "formalChargeList") {
        Some(value) => {
            let charges = integer_array(value, "formalChargeList")?
                .into_iter()
                .map(|charge| {
                    i32::try_from(charge).map_err(|_| invalid("formal charge overflows i32"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if charges.len() != atom_names.len() {
                return Err(invalid(format!(
                    "group {index} formal charge length differs from atom count"
                )));
            }
            charges
        }
        None => vec![0; atom_names.len()],
    };
    let atom_values = integer_array(required(map, "bondAtomList")?, "bondAtomList")?;
    let order_values = integer_array(required(map, "bondOrderList")?, "bondOrderList")?;
    if atom_values.len() != order_values.len().saturating_mul(2) {
        return Err(invalid(format!(
            "group {index} bond atom/order lengths differ"
        )));
    }
    let mut bond_atoms = Vec::with_capacity(order_values.len());
    for pair in atom_values.as_chunks::<2>().0 {
        let atom1 = usize::try_from(pair[0]).map_err(|_| invalid("negative group bond index"))?;
        let atom2 = usize::try_from(pair[1]).map_err(|_| invalid("negative group bond index"))?;
        if atom1 >= atom_names.len() || atom2 >= atom_names.len() || atom1 == atom2 {
            return Err(invalid(format!("group {index} has an invalid bond index")));
        }
        bond_atoms.push((atom1, atom2));
    }
    let bond_orders = order_values
        .into_iter()
        .map(|order| u8::try_from(order).map_err(|_| invalid("group bond order overflows u8")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MmtfGroup {
        name,
        atom_names,
        elements,
        formal_charges,
        bond_atoms,
        bond_orders,
    })
}

fn validate_bonds(bonds: &[MmtfBond], num_atoms: usize, expected: usize) -> Result<(), MmtfError> {
    if bonds.len() != expected {
        return Err(invalid(format!(
            "decoded {} bonds, expected {expected}",
            bonds.len()
        )));
    }
    let mut seen = HashSet::with_capacity(bonds.len());
    for bond in bonds {
        if bond.atom1 >= num_atoms || bond.atom2 >= num_atoms || bond.atom1 == bond.atom2 {
            return Err(invalid(format!(
                "bond ({}, {}) is out of bounds",
                bond.atom1, bond.atom2
            )));
        }
        let key = if bond.atom1 < bond.atom2 {
            (bond.atom1, bond.atom2)
        } else {
            (bond.atom2, bond.atom1)
        };
        if !seen.insert(key) {
            return Err(invalid(format!(
                "duplicate bond ({}, {})",
                bond.atom1, bond.atom2
            )));
        }
    }
    Ok(())
}

fn parse_unit_cell(map: &[(Value, Value)]) -> Result<Option<[f64; 6]>, MmtfError> {
    let Some(value) = map_value(map, "unitCell") else {
        return Ok(None);
    };
    let values = array(value, "unitCell")?;
    if values.len() != 6 {
        return Err(invalid(format!(
            "unitCell has {}, expected 6 values",
            values.len()
        )));
    }
    let mut cell = [0.0; 6];
    for (target, value) in cell.iter_mut().zip(values) {
        *target = value_f64(value, "unitCell")?;
    }
    if cell[..3].iter().all(|&value| value > 0.0) && cell[3..].iter().all(|&value| value > 0.0) {
        Ok(Some(cell))
    } else {
        Ok(None)
    }
}

fn decode_header<'a>(
    bytes: &'a [u8],
    field: &str,
) -> Result<(u32, usize, i32, &'a [u8]), MmtfError> {
    if bytes.len() < 12 {
        return Err(invalid(format!(
            "encoded field {field:?} is shorter than its header"
        )));
    }
    let codec = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let length = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let parameter = i32::from_be_bytes(bytes[8..12].try_into().unwrap());
    Ok((codec, length, parameter, &bytes[12..]))
}

fn decode_bytes(value: &Value, field: &str) -> Result<Vec<i64>, MmtfError> {
    let bytes = match value {
        Value::Binary(bytes) => bytes,
        _ => return Err(invalid(format!("encoded field {field:?} is not binary"))),
    };
    let (codec, length, parameter, raw) = decode_header(bytes, field)?;
    let mut values = match codec {
        2 => raw
            .iter()
            .map(|&value| i64::from(i8::from_be_bytes([value])))
            .collect::<Vec<_>>(),
        4 => decode_i32_bytes(raw, field)?,
        6 | 8 | 9 => decode_i32_bytes(raw, field)?,
        10 => decode_i16_bytes(raw, field)?,
        5 => {
            return Err(invalid(format!(
                "field {field:?} contains strings, not integers"
            )));
        }
        _ => {
            return Err(invalid(format!(
                "field {field:?} uses unsupported codec {codec}"
            )));
        }
    };
    // Empty character arrays are represented by a header carrying the
    // advertised length and no payload (the common representation for
    // missing altLoc/icode data).
    if codec == 6 && raw.is_empty() {
        return Ok(Vec::new());
    }
    if matches!(codec, 6 | 8 | 9) {
        values = run_length_decode(&values, field)?;
    }
    if codec == 10 {
        values = recursive_index_decode(&values, field)?;
    }
    if matches!(codec, 8 | 10) {
        let mut total = 0i64;
        for value in &mut values {
            total = total
                .checked_add(*value)
                .ok_or_else(|| invalid(format!("field {field:?} delta overflow")))?;
            *value = total;
        }
    }
    if values.len() != length {
        return Err(invalid(format!(
            "decoded field {field:?} has {}, expected {length} values",
            values.len()
        )));
    }
    if matches!(codec, 9 | 10) && parameter == 0 {
        return Err(invalid(format!(
            "field {field:?} has a zero floating-point divider"
        )));
    }
    Ok(values)
}

fn recursive_index_decode(values: &[i64], field: &str) -> Result<Vec<i64>, MmtfError> {
    const MAX_SHORT: i64 = 32_767;
    const MIN_SHORT: i64 = -32_768;
    let mut output = Vec::new();
    let mut pending = 0i64;
    for &value in values {
        if value == MAX_SHORT || value == MIN_SHORT {
            pending = pending
                .checked_add(value)
                .ok_or_else(|| invalid(format!("field {field:?} recursive index overflow")))?;
            continue;
        }
        pending = pending
            .checked_add(value)
            .ok_or_else(|| invalid(format!("field {field:?} recursive index overflow")))?;
        output.push(pending);
        pending = 0;
    }
    if pending != 0 {
        return Err(invalid(format!(
            "field {field:?} ends with an incomplete recursive index"
        )));
    }
    Ok(output)
}

fn decode_i16_bytes(raw: &[u8], field: &str) -> Result<Vec<i64>, MmtfError> {
    let (chunks, remainder) = raw.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(invalid(format!(
            "field {field:?} has a partial 16-bit value"
        )));
    }
    Ok(chunks
        .iter()
        .map(|bytes| i64::from(i16::from_be_bytes(*bytes)))
        .collect())
}

fn decode_i32_bytes(raw: &[u8], field: &str) -> Result<Vec<i64>, MmtfError> {
    let (chunks, remainder) = raw.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(invalid(format!(
            "field {field:?} has a partial 32-bit value"
        )));
    }
    Ok(chunks
        .iter()
        .map(|bytes| i64::from(i32::from_be_bytes(*bytes)))
        .collect())
}

fn run_length_decode(values: &[i64], field: &str) -> Result<Vec<i64>, MmtfError> {
    if !values.len().is_multiple_of(2) {
        return Err(invalid(format!(
            "field {field:?} has an odd run-length payload"
        )));
    }
    let mut output = Vec::new();
    for pair in values.as_chunks::<2>().0 {
        let count = usize::try_from(pair[1])
            .map_err(|_| invalid(format!("field {field:?} has a negative run length")))?;
        if count == 0 {
            return Err(invalid(format!("field {field:?} has a zero run length")));
        }
        let new_len = output
            .len()
            .checked_add(count)
            .ok_or_else(|| invalid("run-length output overflow"))?;
        output.resize(new_len, pair[0]);
    }
    Ok(output)
}

fn decode_i32_array(value: &Value, field: &str) -> Result<Vec<i32>, MmtfError> {
    decode_bytes(value, field)?
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| invalid(format!("field {field:?} value overflows i32")))
        })
        .collect()
}

fn decode_usize_array(value: &Value, field: &str) -> Result<Vec<usize>, MmtfError> {
    decode_bytes(value, field)?
        .into_iter()
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| invalid(format!("field {field:?} contains a negative index")))
        })
        .collect()
}

fn decode_u8_array(value: &Value, field: &str) -> Result<Vec<u8>, MmtfError> {
    decode_bytes(value, field)?
        .into_iter()
        .map(|value| {
            u8::try_from(value).map_err(|_| invalid(format!("field {field:?} value overflows u8")))
        })
        .collect()
}

fn decode_f64_array(value: &Value, field: &str) -> Result<Vec<f64>, MmtfError> {
    let bytes = match value {
        Value::Binary(bytes) => bytes,
        _ => return Err(invalid(format!("encoded field {field:?} is not binary"))),
    };
    let (codec, _length, parameter, _raw) = decode_header(bytes, field)?;
    let integers = decode_bytes(value, field)?;
    if !matches!(codec, 9 | 10) {
        return Err(invalid(format!(
            "floating-point field {field:?} uses non-float codec {codec}"
        )));
    }
    if parameter == 0 {
        return Err(invalid(format!("field {field:?} has a zero divider")));
    }
    Ok(integers
        .into_iter()
        .map(|value| value as f64 / f64::from(parameter))
        .collect())
}

fn decode_strings(value: &Value, field: &str) -> Result<Vec<String>, MmtfError> {
    let bytes = match value {
        Value::Binary(bytes) => bytes,
        _ => return Err(invalid(format!("encoded field {field:?} is not binary"))),
    };
    let (codec, length, parameter, raw) = decode_header(bytes, field)?;
    if codec == 5 {
        let width = usize::try_from(parameter)
            .map_err(|_| invalid(format!("field {field:?} has a negative string width")))?;
        if width == 0 || raw.len() % width != 0 {
            return Err(invalid(format!(
                "field {field:?} has an invalid fixed string width"
            )));
        }
        let strings = raw
            .chunks_exact(width)
            .map(|chunk| {
                let end = chunk
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(chunk.len());
                std::str::from_utf8(&chunk[..end])
                    .map(str::to_owned)
                    .map_err(|_| invalid(format!("field {field:?} contains invalid UTF-8")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if strings.len() != length {
            return Err(invalid(format!(
                "decoded field {field:?} has {}, expected {length} values",
                strings.len()
            )));
        }
        return Ok(strings);
    }
    if codec != 6 {
        return Err(invalid(format!(
            "string field {field:?} uses unsupported codec {codec}"
        )));
    }
    let chars = decode_bytes(value, field)?;
    chars
        .into_iter()
        .map(|value| {
            u8::try_from(value)
                .map(char::from)
                .map(|character| character.to_string())
                .map_err(|_| invalid(format!("field {field:?} contains an invalid character")))
        })
        .collect()
}

fn decode_optional_f64(
    map: &[(Value, Value)],
    field: &str,
    expected: usize,
    default: f64,
) -> Result<Vec<f64>, MmtfError> {
    match map_value(map, field) {
        Some(value) => {
            let values = decode_f64_array(value, field)?;
            if values.len() != expected && !values.is_empty() {
                return Err(invalid(format!(
                    "{field} has {}, expected {expected} entries",
                    values.len()
                )));
            }
            Ok(if values.is_empty() {
                vec![default; expected]
            } else {
                values
            })
        }
        None => Ok(vec![default; expected]),
    }
}

fn decode_optional_strings(
    map: &[(Value, Value)],
    field: &str,
    expected: usize,
) -> Result<Vec<String>, MmtfError> {
    match map_value(map, field) {
        Some(value) => {
            let values = decode_strings(value, field)?;
            if values.len() != expected && !values.is_empty() {
                return Err(invalid(format!(
                    "{field} has {}, expected {expected} entries",
                    values.len()
                )));
            }
            Ok(if values.is_empty() {
                vec![String::new(); expected]
            } else {
                values
            })
        }
        None => Ok(vec![String::new(); expected]),
    }
}

impl Universe {
    /// Construct a Universe from an MMTF file (plain or gzip-compressed).
    pub fn from_mmtf(path: impl AsRef<Path>) -> crate::Result<Self> {
        let file =
            read_mmtf(path).map_err(|error| crate::Error::InvalidInput(error.to_string()))?;
        Self::from_mmtf_file(file)
    }

    /// Construct a Universe from parsed MMTF data.
    pub fn from_mmtf_file(file: MmtfFile) -> crate::Result<Self> {
        let mut atoms = Vec::with_capacity(file.num_atoms);
        let mut group_index = 0usize;
        let mut atom_index = 0usize;
        let mut residues = Vec::with_capacity(file.num_groups);
        let mut segments = Vec::with_capacity(file.num_chains);
        for (chain_index, &groups_in_chain) in file.groups_per_chain.iter().enumerate() {
            let base_segid = file.chain_names[chain_index]
                .trim_matches('\0')
                .trim()
                .to_owned();
            let base_segid = if base_segid.is_empty() {
                "SYSTEM".to_owned()
            } else {
                base_segid
            };
            let chain_id = file.chain_ids[chain_index]
                .trim_matches('\0')
                .trim()
                .to_owned();
            segments.push(Segment::new(chain_index, base_segid.clone()));
            let segment_identity = chain_index;
            for _ in 0..groups_in_chain {
                let group_type = file.group_types[group_index];
                let group = &file.groups[group_type];
                let residue_index = residues.len();
                residues.push(Residue::new(
                    residue_index,
                    file.group_ids[group_index],
                    group.name.clone(),
                    segment_identity,
                ));
                segments[segment_identity]
                    .residue_indices
                    .push(residue_index);
                for (local, name) in group.atom_names.iter().enumerate() {
                    let element = group.elements[local].trim().to_owned();
                    let mut atom = Atom::new(
                        atom_index,
                        name.trim().to_owned(),
                        file.positions[atom_index],
                    );
                    atom.atom_type = (!element.is_empty()).then_some(element.clone());
                    atom.element = (!element.is_empty()).then_some(element);
                    atom.mass = atom
                        .element
                        .as_deref()
                        .and_then(|element| crate::guesser::guess_mass(element).ok())
                        .unwrap_or(0.0);
                    atom.charge = f64::from(group.formal_charges[local]);
                    atom.resid = file.group_ids[group_index];
                    atom.resname = group.name.clone();
                    atom.segid = base_segid.clone();
                    atom.chain_id = chain_id.clone();
                    atom.segment_index = segment_identity;
                    atom.residue_index = residue_index;
                    atom.temp_factor = Some(file.b_factors[atom_index]);
                    atom.occupancy = Some(file.occupancies[atom_index]);
                    atoms.push(atom);
                    residues[residue_index].atom_indices.push(atom_index);
                    atom_index += 1;
                }
                group_index += 1;
            }
        }
        if atom_index != file.num_atoms || group_index != file.num_groups {
            return Err(crate::Error::InvalidInput(
                "MMTF hierarchy does not cover all atoms/groups".to_owned(),
            ));
        }
        let mut topology = Topology {
            atoms,
            residues,
            segments,
            bonds: Vec::new(),
        };
        for source in &file.bonds {
            let mut bond = Bond::new(source.atom1, source.atom2);
            bond.order = source.order;
            topology.add_bond(bond);
        }
        let mut frame = Frame::new(file.positions.clone());
        frame.dimensions = file.unit_cell;
        Ok(Self {
            topology,
            trajectory: Trajectory::new(vec![frame]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data")
            .join(name)
    }

    #[test]
    fn reads_173d_topology_and_coordinates() {
        let file = read_mmtf(fixture("173D.mmtf")).unwrap();
        assert_eq!(
            (
                file.num_atoms,
                file.num_groups,
                file.num_chains,
                file.bonds.len()
            ),
            (512, 124, 8, 458)
        );
        assert_eq!(file.atom_ids[..3], [1, 2, 3]);
        assert_eq!(file.positions[0], [-0.798, 12.632, 23.231]);
        let universe = Universe::from_mmtf_file(file).unwrap();
        assert_eq!(
            (
                universe.n_atoms(),
                universe.n_residues(),
                universe.n_segments()
            ),
            (512, 124, 8)
        );
        assert_eq!(universe.topology.bonds.len(), 458);
        assert_eq!(universe.atoms().atoms[0].name, "O5'");
        assert_eq!(universe.atoms().atoms[0].temp_factor, Some(9.48));
        let dimensions = universe.current_frame().unwrap().dimensions.unwrap();
        assert!((dimensions[0] - 69.9).abs() < 1.0e-5);
        assert!((dimensions[1] - 61.41).abs() < 1.0e-5);
        assert!((dimensions[2] - 54.25).abs() < 1.0e-5);
    }

    #[test]
    fn reads_gzip_and_optional_fields() {
        let file = read_mmtf(fixture("5KIH.mmtf.gz")).unwrap();
        assert_eq!(
            (file.num_atoms, file.num_groups, file.num_models),
            (1140, 36, 2)
        );
        assert_eq!(file.positions[0], [38.428, 16.44, 28.841]);
        assert_eq!(
            file.b_factors.iter().take(3).copied().collect::<Vec<_>>(),
            vec![0.0; 3]
        );
        let universe = Universe::from_mmtf_file(file).unwrap();
        assert_eq!(universe.n_atoms(), 1140);
        assert_eq!(universe.n_frames(), 1);
    }

    #[test]
    fn skinny_files_default_missing_attributes() {
        let one = read_mmtf(fixture("1ubq-less-optional.mmtf")).unwrap();
        assert_eq!(
            (one.num_atoms, one.num_groups, one.num_chains),
            (660, 134, 2)
        );
        assert_eq!(one.atom_ids[0], 1);
        assert_eq!(one.alt_locs[0], "");
        let two = read_mmtf(fixture("3NJW-onlyrequired.mmtf")).unwrap();
        assert_eq!(
            (two.num_atoms, two.num_groups, two.num_chains),
            (169, 44, 2)
        );
        assert_eq!(two.unit_cell, None);
    }

    #[test]
    fn rejects_non_map_and_truncated_data() {
        assert!(MmtfFile::from_bytes(&[0x01]).is_err());
        assert!(matches!(
            MmtfFile::from_bytes(&[0x81, 0xa3, b'f', b'o', b'o', 0x01]),
            Err(MmtfError::InvalidStructure(_))
        ));
    }
}
