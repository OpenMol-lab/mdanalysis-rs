# mdanalysis-rs

`mdanalysis-rs` is a native Rust toolkit for reading, writing, and analysing
molecular-dynamics structures and trajectories. It provides an owned data
model for topology and coordinates, format-specific readers and writers, atom
selection expressions, unit conversion, geometry helpers, and common
trajectory analyses.

The crate is under active development. The public API is useful for Rust
programs today, but it should be treated as pre-1.0 and may change between
releases.

## Features

- A common `Universe` model containing `Topology` and `Trajectory` data.
- Atoms, residues, segments, bonds, velocities, forces, unit-cell dimensions,
  frame times, integrator steps, and format-specific frame data.
- Selection expressions such as `protein`, `backbone`, `name CA`,
  `resid 1-10`, boolean combinations, spatial selections, and bonded/relative
  selections.
- Geometry and analysis routines for distances, angles, dihedrals, periodic
  boundary operations, RMSD, RMSF, radius of gyration, contacts, MSD, RDF,
  and Kabsch superposition.
- In-memory parsing from strings or byte slices for formats that expose those
  entry points, as well as filesystem helpers.
- Typed errors through the crate-level `mdanalysis_rs::Error` and
  format-specific error types.

## Quick start

The repository can be used as a path dependency while developing:

```toml
[dependencies]
mdanalysis_rs = { path = "." }
```

For a program outside this repository, use the Git dependency until a
crates.io release is published:

```toml
[dependencies]
mdanalysis_rs = { git = "https://github.com/OpenMol-lab/mdanalysis-rs.git" }
```

Load a structure, select atoms, inspect the current frame, and iterate over a
trajectory:

```rust
use mdanalysis_rs::{Result, Universe};

fn main() -> Result<()> {
    let universe = Universe::from_pdb("structure.pdb")?;

    println!(
        "{} atoms, {} residues, {} frames",
        universe.n_atoms(),
        universe.n_residues(),
        universe.n_frames()
    );

    let backbone = universe.select_atoms("backbone")?;
    println!("backbone atoms: {}", backbone.len());
    println!("backbone centre of mass: {:?}", backbone.center_of_mass());

    for frame in universe.trajectory.iter() {
        println!("step={} time={} dimensions={:?}", frame.step, frame.time, frame.dimensions);
    }

    Ok(())
}
```

Topology and trajectory files can be combined with a format-specific
constructor. For example:

```rust
use mdanalysis_rs::Universe;

let universe = Universe::from_psf_and_xtc("topology.psf", "trajectory.xtc")?;
let first_frame = universe.trajectory.frame(0).expect("trajectory is not empty");
println!("first frame has {} atoms", first_frame.n_atoms());
```

Most `Universe::from_*` constructors have a corresponding `*_str` or
`*_bytes` variant when parsing in-memory data is useful. The low-level format
types, such as `PdbStructure`, `DcdFile`, and `XtcFile`, are also re-exported
from the crate root.

## Core model

`Universe` owns both topology and trajectory data:

- `Topology` stores atoms, residues, segments, and bonds. Atom indices are
  zero-based.
- `Trajectory` stores an ordered list of `Frame` values and provides random
  access, slicing, iteration, and a current-frame cursor.
- `Frame` stores positions and optional velocities, forces, dimensions, time,
  step, and named per-frame data in a `BTreeMap<String, Vec<f64>>`.
- `AtomGroup` is an owned, iterable selection of atoms with helpers for masses,
  centres, bounding boxes, and radius of gyration.

Use `Universe::set_frame` when selections or coordinate access should refer to
a particular current frame. Use `universe.trajectory.iter()` when inspecting
all frames without changing the current-frame cursor.

## Atom selections

Selections are parsed by `mdanalysis_rs::selection::Selection` and are also
available through `Universe::select_atoms` and `AtomGroup::select_atoms`.
Common expressions include:

```text
protein
backbone and not name H*
name CA or name N
resname ALA GLY
resid 1-10
around 5.0 name CA
prop z > 10.0
same resname as resid 42
```

The parser supports `name`, `resname`, `resid`/`resnum`, `index`, `bynum`,
`element`, `chainID`, `segid`, `type`, `prop`, `around`, `point`, `same`,
`byres`, `bonded`, `global`, and named `group` expressions, together with
`and`, `or`, `not`, and parentheses. `protein`, `backbone`, `water`, and
nucleic-acid shortcuts are built in.

## Supported formats

The format modules expose typed structures as well as `Universe` adapters.
Support is intentionally format-specific; consult the module-level API docs
for details about optional records and unit handling.

### Structures and topologies

| Format | Access |
| --- | --- |
| PDB / XPDB | Read and write |
| PDBQT | Read and write |
| PSF | Read and write |
| PQR, MOL2, CRD/CARD | Read and write |
| XYZ, GRO | Read and write at the coordinate-file level |
| Amber PRMTOP/TOP | Read |
| Amber INPCRD and NAMD binary coordinates | Read and write |
| GROMACS ITP/TOP and TPR | Read |
| LAMMPS DATA | Read and write |
| DMS, MMTF, GSD, HOOMD XML | Read |
| H5MD, NetCDF, FHI-AIMS, GAMESS | Read (FHI-AIMS also writes) |
| DL_POLY CONFIG | Read and write |
| Tinker XYZ / ARC | Read and write |

### Trajectories

| Format | Access |
| --- | --- |
| DCD | Read and write |
| GROMACS XTC / TRR | Read and write |
| GROMACS TNG | Read |
| GROMOS TRC | Read |
| Amber TRJ / MDCRD | Read |
| TRZ | Read and write |
| H5MD, NetCDF, GSD | Read |
| LAMMPS ASCII dump | Read |
| DL_POLY HISTORY | Read and write |
| Tinker ARC | Read and write |

When a trajectory format does not contain topology fields, use a combined
constructor such as `Universe::from_pdb_and_trr` or
`Universe::from_prmtop_and_trj`. A trajectory-only constructor is available
for several formats; in that case atom metadata is populated from the
trajectory when the format has no separate topology.

## Units

The `units` module defines explicit length, time, energy, velocity, force,
charge, and mass units and checked conversions between compatible dimensions.
The crate's MDAnalysis-style base units are Angstroms (`A`), picoseconds
(`ps`), kilojoules per mole (`kJ/mol`), elementary charge (`e`), and atomic
mass units (`amu`). Some readers retain the source format's native values (for
example, GRO/XTC/TRR coordinates are conventionally in nanometres), while
readers such as H5MD and TNG can convert recognised units. Check the relevant
module documentation before combining data from different formats.

## Analysis example

Analyses implement the `Analysis` trait and can be run over every frame:

```rust
use mdanalysis_rs::{Analysis, Error, RmsdAnalysis, Universe};

fn main() -> mdanalysis_rs::Result<()> {
    let universe = Universe::from_pdb("structure.pdb")?;
    let reference = universe
        .trajectory
        .frame(0)
        .ok_or_else(|| Error::InvalidInput("trajectory is empty".to_owned()))?
        .positions
        .clone();

    let values = RmsdAnalysis::new(reference)
        .with_superposition(true)
        .run(&universe.trajectory)
        .map_err(Error::InvalidInput)?;

    println!("RMSD values: {values:?}");
    Ok(())
}
```

For calculations that do not need a `Universe`, the functions in
`geometry`, `mdamath`, `distances`, `analysis`, and `analysis_algorithms`
operate directly on coordinate slices.

## Development

Run the formatter, checks, tests, and lints from the repository root:

```text
cargo fmt --all -- --check
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

Build the API documentation locally with:

```text
cargo doc --no-deps --open
```

The test suite exercises the parsers, writers, topology construction,
selections, metadata propagation, unit conversions, and analysis helpers.
