//! Shared helpers for path-based text readers.
//!
//! MDAnalysis accepts gzip and bzip2 streams through its `openany` helper.
//! Detecting compression from the bytes keeps the same behavior independent
//! of the filename extension while leaving reader/string APIs unchanged.

use bzip2::read::BzDecoder;
use flate2::read::GzDecoder;
use std::fs;
use std::io::{self, Cursor, Read};
use std::path::Path;

/// Read a UTF-8 text file, transparently decoding gzip or bzip2 streams.
pub(crate) fn read_text_file(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;
    let decoded = decode_bytes(&bytes)?;
    String::from_utf8(decoded).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn decode_bytes(bytes: &[u8]) -> io::Result<Vec<u8>> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = GzDecoder::new(Cursor::new(bytes));
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded)?;
        return Ok(decoded);
    }
    if bytes.starts_with(b"BZh") {
        let mut decoder = BzDecoder::new(Cursor::new(bytes));
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded)?;
        return Ok(decoded);
    }
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::read_text_file;
    use bzip2::Compression as BzCompression;
    use bzip2::write::BzEncoder;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::fs;
    use std::io::Write;

    fn temporary_path(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mdanalysis-rs-io-utils-{}-{}.{suffix}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn reads_plain_gzip_and_bzip2_text() {
        let plain_path = temporary_path("txt");
        fs::write(&plain_path, b"plain\n").unwrap();
        assert_eq!(read_text_file(&plain_path).unwrap(), "plain\n");
        fs::remove_file(&plain_path).unwrap();

        let mut gzip = GzEncoder::new(Vec::new(), Compression::default());
        gzip.write_all(b"gzip\n").unwrap();
        let gzip_path = temporary_path("gz");
        fs::write(&gzip_path, gzip.finish().unwrap()).unwrap();
        assert_eq!(read_text_file(&gzip_path).unwrap(), "gzip\n");
        fs::remove_file(&gzip_path).unwrap();

        let mut bzip2 = BzEncoder::new(Vec::new(), BzCompression::default());
        bzip2.write_all(b"bzip2\n").unwrap();
        let bzip2_path = temporary_path("bz2");
        fs::write(&bzip2_path, bzip2.finish().unwrap()).unwrap();
        assert_eq!(read_text_file(&bzip2_path).unwrap(), "bzip2\n");
        fs::remove_file(&bzip2_path).unwrap();
    }

    #[test]
    fn path_readers_accept_compressed_upstream_fixtures() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mdanalysis/testsuite/MDAnalysisTests/data");
        let xyz = crate::coordinates::read_xyz(root.join("coordinates/test.xyz.bz2")).unwrap();
        assert_eq!(xyz.n_atoms(), 5);
        let gro = crate::coordinates::read_gro(root.join("coordinates/test.gro.bz2")).unwrap();
        assert_eq!(gro.n_atoms(), 5);
        let psf = crate::psf::read_psf(root.join("analysis/1k5i_c36.psf.gz")).unwrap();
        assert!(!psf.atoms.is_empty());
        let pdbqt = crate::pdbqt::read_pdbqt(root.join("tyrosol.pdbqt.bz2")).unwrap();
        assert!(!pdbqt.atoms.is_empty());
    }
}
