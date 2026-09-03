//! SQLite-backed Desmond Molecular Structure (DMS) topology and coordinates.
//!
//! A DMS file is a small SQLite database.  The standard schema stores atoms
//! in `particle`, connectivity in `bond`, and three unit-cell vectors in
//! `global_cell`.  This module reads those tables into ordinary Rust values
//! and provides a [`Universe`](crate::Universe) constructor.

use crate::core::{Atom, Bond, Frame, Topology, Trajectory, Universe};
use crate::mdamath::triclinic_box;
use rusqlite::{Connection, Error as SqlError, OpenFlags, Row};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

/// One row from a DMS `particle` table.
#[derive(Clone, Debug, PartialEq)]
pub struct DmsParticle {
    pub id: i64,
    pub anum: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub mass: f64,
    pub charge: f64,
    pub name: String,
    pub resname: String,
    pub resid: i32,
    pub chain: String,
    pub segid: String,
}

impl DmsParticle {
    #[must_use]
    pub const fn position(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    #[must_use]
    pub const fn velocity(&self) -> [f64; 3] {
        [self.vx, self.vy, self.vz]
    }
}

/// One row from a DMS `bond` table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmsBond {
    pub atom1: i64,
    pub atom2: i64,
    pub order: Option<i64>,
}

impl DmsBond {
    #[must_use]
    pub const fn new(atom1: i64, atom2: i64, order: Option<i64>) -> Self {
        Self {
            atom1,
            atom2,
            order,
        }
    }
}

/// Parsed DMS topology, coordinates, and optional unit-cell vectors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DmsFile {
    pub particles: Vec<DmsParticle>,
    pub bonds: Vec<DmsBond>,
    /// Cell vectors in the order stored by `global_cell`.
    pub global_cell: Option<[[f64; 3]; 3]>,
}

impl DmsFile {
    /// Open and parse a DMS SQLite database.
    pub fn read_file(path: impl AsRef<Path>) -> Result<Self, DmsError> {
        let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(DmsError::Sql)?;
        Self::from_connection(&connection)
    }

    /// Alias for [`DmsFile::read_file`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DmsError> {
        Self::read_file(path)
    }

    /// Parse a DMS database from an existing SQLite connection.
    pub fn from_connection(connection: &Connection) -> Result<Self, DmsError> {
        let particles = read_particles(connection)?;
        if particles.is_empty() {
            return Err(DmsError::InvalidStructure(
                "particle table contains no atoms".to_string(),
            ));
        }
        let bonds = read_bonds(connection, &particles)?;
        let global_cell = read_global_cell(connection)?;
        let file = Self {
            particles,
            bonds,
            global_cell,
        };
        validate_structure(&file)?;
        Ok(file)
    }

    /// Return unit-cell dimensions, or `None` for a missing/zero cell.
    #[must_use]
    pub fn dimensions(&self) -> Option<[f64; 6]> {
        self.global_cell.and_then(|vectors| {
            if vectors.iter().flatten().all(|value| value.is_finite())
                && vectors
                    .iter()
                    .all(|vector| vector.iter().map(|value| value * value).sum::<f64>() > 1.0e-24)
            {
                let dimensions = triclinic_box(vectors);
                (dimensions[..3].iter().all(|value| *value > 0.0)
                    && dimensions[3..].iter().all(|value| *value > 0.0))
                .then_some(dimensions)
            } else {
                None
            }
        })
    }
}

/// Read a DMS SQLite database from a filesystem path.
pub fn read_dms(path: impl AsRef<Path>) -> Result<DmsFile, DmsError> {
    DmsFile::read_file(path)
}

/// Errors produced while reading a DMS database.
#[derive(Debug)]
pub enum DmsError {
    Sql(SqlError),
    InvalidStructure(String),
}

impl fmt::Display for DmsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sql(error) => write!(formatter, "DMS SQLite error: {error}"),
            Self::InvalidStructure(message) => {
                write!(formatter, "invalid DMS structure: {message}")
            }
        }
    }
}

impl std::error::Error for DmsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sql(error) => Some(error),
            Self::InvalidStructure(_) => None,
        }
    }
}

impl From<SqlError> for DmsError {
    fn from(error: SqlError) -> Self {
        Self::Sql(error)
    }
}

fn read_particles(connection: &Connection) -> Result<Vec<DmsParticle>, DmsError> {
    let mut statement = connection.prepare(
        "SELECT id, anum, x, y, z, vx, vy, vz, mass, charge, name, resname, resid, chain, segid \
         FROM particle ORDER BY id",
    )?;
    let rows = statement.query_map([], particle_from_row)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(DmsError::Sql)
}

fn particle_from_row(row: &Row<'_>) -> Result<DmsParticle, SqlError> {
    Ok(DmsParticle {
        id: row.get(0)?,
        anum: row.get(1)?,
        x: row.get(2)?,
        y: row.get(3)?,
        z: row.get(4)?,
        vx: row.get(5)?,
        vy: row.get(6)?,
        vz: row.get(7)?,
        mass: row.get(8)?,
        charge: row.get(9)?,
        name: row.get::<_, String>(10)?.trim().to_string(),
        resname: row.get::<_, String>(11)?.trim().to_string(),
        resid: row.get(12)?,
        chain: row
            .get::<_, Option<String>>(13)?
            .unwrap_or_default()
            .trim()
            .to_string(),
        segid: row
            .get::<_, Option<String>>(14)?
            .unwrap_or_default()
            .trim()
            .to_string(),
    })
}

fn read_bonds(
    connection: &Connection,
    particles: &[DmsParticle],
) -> Result<Vec<DmsBond>, DmsError> {
    let ids: HashSet<i64> = particles.iter().map(|particle| particle.id).collect();
    let mut statement = connection.prepare("SELECT p0, p1, \"order\" FROM bond")?;
    let rows = statement.query_map([], |row| {
        Ok(DmsBond {
            atom1: row.get(0)?,
            atom2: row.get(1)?,
            order: row.get(2)?,
        })
    })?;
    let mut bonds = Vec::new();
    for row in rows {
        let bond = row?;
        if !ids.contains(&bond.atom1) || !ids.contains(&bond.atom2) {
            return Err(DmsError::InvalidStructure(format!(
                "bond references missing particle ({}, {})",
                bond.atom1, bond.atom2
            )));
        }
        bonds.push(bond);
    }
    Ok(bonds)
}

fn read_global_cell(connection: &Connection) -> Result<Option<[[f64; 3]; 3]>, DmsError> {
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'global_cell')",
        [],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(None);
    }
    let mut statement = connection.prepare("SELECT id, x, y, z FROM global_cell ORDER BY id")?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            [row.get(1)?, row.get(2)?, row.get(3)?],
        ))
    })?;
    let mut vectors = [[0.0; 3]; 3];
    let mut count = 0;
    for row in rows {
        let (_id, vector) = row?;
        if count >= 3 {
            return Err(DmsError::InvalidStructure(
                "global_cell must contain at most three vectors".to_string(),
            ));
        }
        vectors[count] = vector;
        count += 1;
    }
    match count {
        0 => Ok(None),
        3 => Ok(Some(vectors)),
        _ => Err(DmsError::InvalidStructure(
            "global_cell must contain either zero or three vectors".to_string(),
        )),
    }
}

fn validate_structure(file: &DmsFile) -> Result<(), DmsError> {
    let mut ids = HashSet::with_capacity(file.particles.len());
    for particle in &file.particles {
        if !ids.insert(particle.id) {
            return Err(DmsError::InvalidStructure(format!(
                "duplicate particle id {}",
                particle.id
            )));
        }
        let values = [
            particle.x,
            particle.y,
            particle.z,
            particle.vx,
            particle.vy,
            particle.vz,
            particle.mass,
            particle.charge,
        ];
        if values.iter().any(|value| !value.is_finite()) || particle.mass < 0.0 {
            return Err(DmsError::InvalidStructure(format!(
                "particle {} contains non-finite coordinates or properties",
                particle.id
            )));
        }
        if particle.name.is_empty() || particle.resname.is_empty() {
            return Err(DmsError::InvalidStructure(format!(
                "particle {} has an empty name or residue name",
                particle.id
            )));
        }
    }
    let mut bond_keys = HashSet::with_capacity(file.bonds.len());
    for bond in &file.bonds {
        if bond.atom1 == bond.atom2 {
            return Err(DmsError::InvalidStructure(format!(
                "bond references particle {} twice",
                bond.atom1
            )));
        }
        let key = if bond.atom1 < bond.atom2 {
            (bond.atom1, bond.atom2)
        } else {
            (bond.atom2, bond.atom1)
        };
        if !bond_keys.insert(key) {
            return Err(DmsError::InvalidStructure(format!(
                "duplicate bond ({}, {})",
                bond.atom1, bond.atom2
            )));
        }
    }
    Ok(())
}

impl Universe {
    /// Construct a universe from a DMS SQLite database.
    pub fn from_dms(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_dms_file(read_dms(path)?)
    }

    /// Construct a universe from parsed DMS data.
    pub fn from_dms_file(file: DmsFile) -> crate::Result<Self> {
        validate_structure(&file)?;
        let mut atoms = Vec::with_capacity(file.particles.len());
        let mut id_to_index = HashMap::with_capacity(file.particles.len());
        for (index, particle) in file.particles.iter().enumerate() {
            let element =
                crate::guesser::guess_element(&particle.name, None, Some(&particle.resname)).ok();
            let mut atom = Atom::new(index, particle.name.clone(), particle.position())
                .with_mass(particle.mass);
            atom.atom_type = element.clone();
            atom.element = element;
            atom.charge = particle.charge;
            atom.resid = particle.resid;
            atom.resname = particle.resname.clone();
            atom.chain_id = particle.chain.clone();
            atom.segid = if particle.segid.is_empty() {
                "SYSTEM".to_string()
            } else {
                particle.segid.clone()
            };
            id_to_index.insert(particle.id, index);
            atoms.push(atom);
        }
        let mut topology = Topology::new(atoms);
        for source in &file.bonds {
            let atom1 = id_to_index.get(&source.atom1).copied().ok_or_else(|| {
                crate::Error::InvalidInput(format!(
                    "bond references missing particle {}",
                    source.atom1
                ))
            })?;
            let atom2 = id_to_index.get(&source.atom2).copied().ok_or_else(|| {
                crate::Error::InvalidInput(format!(
                    "bond references missing particle {}",
                    source.atom2
                ))
            })?;
            let mut bond = Bond::new(atom1, atom2);
            bond.order = source.order.and_then(|order| u8::try_from(order).ok());
            topology.add_bond(bond);
        }
        let mut frame = Frame::new(file.particles.iter().map(DmsParticle::position).collect());
        frame.dimensions = file.dimensions();
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
    fn reads_adk_topology_and_coordinates() {
        let universe = Universe::from_dms(fixture("adk_closed_domains.dms")).unwrap();
        assert_eq!(universe.n_atoms(), 3341);
        assert_eq!(universe.n_residues(), 214);
        assert_eq!(universe.n_segments(), 3);
        assert_eq!(universe.topology.bonds.len(), 3365);
        let position = universe.atoms().atoms[0].position;
        let expected = [-11.0530004501343, 26.6800003051758, 12.7419996261597];
        assert!(
            position
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6)
        );
        assert_eq!(universe.current_frame().unwrap().dimensions, None);
        assert_eq!(universe.current_frame().unwrap().velocities, None);
        assert_eq!(universe.select_atoms("name CA").unwrap().len(), 214);
        assert_eq!(universe.select_atoms("resid 33").unwrap().len(), 12);
        assert_eq!(universe.select_atoms("resname ALA").unwrap().len(), 190);
        assert_eq!(universe.select_atoms("segid NMP").unwrap().len(), 437);
        assert_eq!(universe.select_atoms("segid LID").unwrap().len(), 598);
        assert_eq!(universe.select_atoms("segid CORE").unwrap().len(), 2306);
        assert_eq!(
            universe.atoms().atoms[..7]
                .iter()
                .map(|atom| atom.atom_type.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["N", "H", "H", "H", "C", "H", "C"]
        );
        let first = &universe.topology.atoms[0];
        assert!((first.mass - 14.006999969482422).abs() < 1.0e-6);
        assert!((first.charge + 0.30000001192092896).abs() < 1.0e-6);
        assert_eq!(first.chain_id, "X");
        assert_eq!(first.segid, "CORE");
        assert_eq!(universe.topology.bonds[0].order, Some(1));
    }

    #[test]
    fn reads_particle_metadata_and_cell_dimensions() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE particle (id INTEGER, anum INTEGER, x FLOAT, y FLOAT, z FLOAT, vx FLOAT, vy FLOAT, vz FLOAT, mass FLOAT, charge FLOAT, name TEXT, resname TEXT, resid INTEGER, chain TEXT, segid TEXT);
                 CREATE TABLE bond (p0 INTEGER, p1 INTEGER, \"order\" INTEGER);
                 CREATE TABLE global_cell (id INTEGER, x FLOAT, y FLOAT, z FLOAT);
                 INSERT INTO particle VALUES (10, 6, 1, 2, 3, 0.1, 0.2, 0.3, 12.0, -0.25, ' CA ', ' ALA ', 7, ' A ', ' SEG ');
                 INSERT INTO particle VALUES (20, 8, 4, 5, 6, 0.4, 0.5, 0.6, 16.0, 0.1, ' O ', ' ALA ', 7, ' A ', ' SEG ');
                 INSERT INTO bond VALUES (10, 20, 2);
                 INSERT INTO global_cell VALUES (1, 2, 0, 0);
                 INSERT INTO global_cell VALUES (2, 0, 3, 0);
                 INSERT INTO global_cell VALUES (3, 0, 0, 4);",
            )
            .unwrap();

        let file = DmsFile::from_connection(&connection).unwrap();
        assert_eq!(file.particles[0].name, "CA");
        assert_eq!(file.particles[0].resname, "ALA");
        assert_eq!(file.particles[0].chain, "A");
        assert_eq!(file.particles[0].segid, "SEG");
        assert_eq!(file.particles[0].velocity(), [0.1, 0.2, 0.3]);
        assert_eq!(
            file.global_cell,
            Some([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        );
        let dimensions = file.dimensions().unwrap();
        assert_eq!(&dimensions[..3], &[2.0, 3.0, 4.0]);
        assert_eq!(&dimensions[3..], &[90.0, 90.0, 90.0]);

        let universe = Universe::from_dms_file(file).unwrap();
        assert_eq!(universe.topology.bonds[0].order, Some(2));
        assert_eq!(
            universe.current_frame().unwrap().dimensions,
            Some(dimensions)
        );
        assert_eq!(universe.topology.atoms[0].segid, "SEG");
        assert_eq!(universe.topology.atoms[0].chain_id, "A");
    }

    #[test]
    fn blank_segids_become_system() {
        let universe = Universe::from_dms(fixture("adk_closed_no_segid.dms")).unwrap();
        assert_eq!(universe.n_segments(), 1);
        assert_eq!(universe.select_atoms("segid SYSTEM").unwrap().len(), 3341);
    }

    #[test]
    fn detects_duplicate_bonds() {
        let connection = Connection::open_in_memory().unwrap();
        connection.execute_batch(
            "CREATE TABLE particle (id INTEGER, anum INTEGER, x FLOAT, y FLOAT, z FLOAT, vx FLOAT, vy FLOAT, vz FLOAT, mass FLOAT, charge FLOAT, name TEXT, resname TEXT, resid INTEGER, chain TEXT, segid TEXT); CREATE TABLE bond (p0 INTEGER, p1 INTEGER, \"order\" INTEGER); INSERT INTO particle VALUES (0,0,0,0,0,0,0,0,1,0,'H','H',1,'',''); INSERT INTO particle VALUES (1,0,1,0,0,0,0,0,1,0,'H','H',1,'',''); INSERT INTO bond VALUES (0,1,1); INSERT INTO bond VALUES (1,0,1);",
        ).unwrap();
        assert!(matches!(
            DmsFile::from_connection(&connection),
            Err(DmsError::InvalidStructure(_))
        ));
    }

    #[test]
    fn rejects_bonds_to_missing_particles() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE particle (id INTEGER, anum INTEGER, x FLOAT, y FLOAT, z FLOAT, vx FLOAT, vy FLOAT, vz FLOAT, mass FLOAT, charge FLOAT, name TEXT, resname TEXT, resid INTEGER, chain TEXT, segid TEXT);
                 CREATE TABLE bond (p0 INTEGER, p1 INTEGER, \"order\" INTEGER);
                 INSERT INTO particle VALUES (10, 6, 0, 0, 0, 0, 0, 0, 12, 0, 'C', 'UNK', 1, '', '');
                 INSERT INTO bond VALUES (10, 99, 1);",
            )
            .unwrap();

        let error = DmsFile::from_connection(&connection).unwrap_err();
        assert!(
            matches!(error, DmsError::InvalidStructure(message) if message.contains("missing particle"))
        );
    }

    #[test]
    fn rejects_partial_global_cell() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE particle (id INTEGER, anum INTEGER, x FLOAT, y FLOAT, z FLOAT, vx FLOAT, vy FLOAT, vz FLOAT, mass FLOAT, charge FLOAT, name TEXT, resname TEXT, resid INTEGER, chain TEXT, segid TEXT);
                 CREATE TABLE bond (p0 INTEGER, p1 INTEGER, \"order\" INTEGER);
                 CREATE TABLE global_cell (id INTEGER, x FLOAT, y FLOAT, z FLOAT);
                 INSERT INTO particle VALUES (0, 6, 0, 0, 0, 0, 0, 0, 12, 0, 'C', 'UNK', 1, '', '');
                 INSERT INTO global_cell VALUES (1, 2, 0, 0);
                 INSERT INTO global_cell VALUES (2, 0, 2, 0);",
            )
            .unwrap();
        assert!(matches!(
            DmsFile::from_connection(&connection),
            Err(DmsError::InvalidStructure(message)) if message.contains("zero or three")
        ));
    }
}
