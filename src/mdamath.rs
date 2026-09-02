//! Mathematical helpers corresponding to the portable part of
//! `MDAnalysis.lib.mdamath`.

use crate::geometry::Vec3;

pub fn norm(vector: Vec3) -> f64 {
    vector.norm()
}

pub fn normal(first: Vec3, second: Vec3) -> Vec3 {
    first.cross(second).normalized()
}

pub fn pdot(first: &[Vec3], second: &[Vec3]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first.iter().zip(second).map(|(a, b)| a.dot(*b)).collect()
}

pub fn pnorm(vectors: &[Vec3]) -> Vec<f64> {
    vectors.iter().map(|vector| vector.norm()).collect()
}

pub fn angle(first: Vec3, second: Vec3) -> f64 {
    crate::geometry::angle(first, second)
}

pub fn stp(first: Vec3, second: Vec3, third: Vec3) -> f64 {
    first.dot(second.cross(third))
}

pub fn dihedral(ab: Vec3, bc: Vec3, cd: Vec3) -> f64 {
    crate::geometry::dihedral(ab, bc, cd)
}

pub fn sarrus_det(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

/// Convert unit-cell lengths and angles into triclinic vectors.
pub fn triclinic_vectors(dimensions: [f64; 6]) -> [[f64; 3]; 3] {
    let [a, b, c, alpha, beta, gamma] = dimensions;
    let (alpha, beta, gamma) = (alpha.to_radians(), beta.to_radians(), gamma.to_radians());
    let cos_alpha = alpha.cos();
    let cos_beta = beta.cos();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    let ax = a;
    let bx = b * cos_gamma;
    let by = b * sin_gamma;
    let cx = c * cos_beta;
    let cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    let cz = (c * c - cx * cx - cy * cy).max(0.0).sqrt();
    [[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]]
}

/// Convert triclinic vectors into `[a,b,c,alpha,beta,gamma]` dimensions.
pub fn triclinic_box(vectors: [[f64; 3]; 3]) -> [f64; 6] {
    let a = Vec3::from(vectors[0]);
    let b = Vec3::from(vectors[1]);
    let c = Vec3::from(vectors[2]);
    let lengths = [a.norm(), b.norm(), c.norm()];
    if lengths.iter().any(|length| *length <= f64::EPSILON) {
        return [0.0; 6];
    }
    let alpha = b
        .dot(c)
        .clamp(-lengths[1] * lengths[2], lengths[1] * lengths[2]);
    let beta = a
        .dot(c)
        .clamp(-lengths[0] * lengths[2], lengths[0] * lengths[2]);
    let gamma = a
        .dot(b)
        .clamp(-lengths[0] * lengths[1], lengths[0] * lengths[1]);
    [
        lengths[0],
        lengths[1],
        lengths[2],
        (alpha / (lengths[1] * lengths[2])).acos().to_degrees(),
        (beta / (lengths[0] * lengths[2])).acos().to_degrees(),
        (gamma / (lengths[0] * lengths[1])).acos().to_degrees(),
    ]
}

pub fn box_volume(dimensions: [f64; 6]) -> f64 {
    sarrus_det(triclinic_vectors(dimensions))
}

pub fn find_fragments(n_atoms: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n_atoms).collect();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    for &(a, b) in bonds {
        if a < n_atoms && b < n_atoms {
            let ra = root(&mut parent, a);
            let rb = root(&mut parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }
    }
    let mut fragments: Vec<Vec<usize>> = Vec::new();
    for index in 0..n_atoms {
        let component = root(&mut parent, index);
        if let Some(fragment) = fragments.iter_mut().find(|fragment| {
            fragment.first().map(|first| root(&mut parent, *first)) == Some(component)
        }) {
            fragment.push(index);
        } else {
            fragments.push(vec![index]);
        }
    }
    fragments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_and_angles() {
        assert_eq!(norm(Vec3::new(3.0, 4.0, 0.0)), 5.0);
        assert!(
            (angle(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)) - 90.0_f64.to_radians())
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn triclinic_round_trip_and_volume() {
        let dimensions = [10.0, 11.0, 12.0, 90.0, 90.0, 90.0];
        let vectors = triclinic_vectors(dimensions);
        assert_eq!(triclinic_box(vectors), dimensions);
        assert!((box_volume(dimensions) - 1320.0).abs() < 1e-10);
    }

    #[test]
    fn fragments_follow_bonds() {
        assert_eq!(
            find_fragments(4, &[(0, 1), (2, 3)]),
            vec![vec![0, 1], vec![2, 3]]
        );
    }
}
