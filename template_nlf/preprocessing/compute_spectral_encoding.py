# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Compute cotangent stiffness / lumped mass generalized LBO eigenpairs.

Input vertices and faces must use the exact correspondence order used for GT.
Output includes the constant mode at column/index zero.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def cotangent_matrices(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Expected triangular faces [F, 3].')
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area2 <= 1e-12):
        raise ValueError('Degenerate triangles are not supported.')
    rows, cols, weights = [], [], []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        cot = np.sum((triangles[:, j] - triangles[:, i]) * (triangles[:, k] - triangles[:, i]), axis=1) / area2
        rows.extend((faces[:, j], faces[:, k]))
        cols.extend((faces[:, k], faces[:, j]))
        weights.extend((0.5 * cot, 0.5 * cot))
    adjacency = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    stiffness = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    mass = np.zeros(n)
    for corner in range(3):
        np.add.at(mass, faces[:, corner], area2 / 6)
    if np.any(mass <= 0):
        raise ValueError('Isolated vertices are not supported.')
    return stiffness, sparse.diags(mass).tocsr()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vertices', required=True, help='NPY [N,3], template order, mm')
    p.add_argument('--faces', required=True, help='NPY [F,3], zero-based vertex indices')
    p.add_argument('--vertebra-level', required=True, choices=['L1', 'L2', 'L3', 'L4', 'L5'])
    p.add_argument('--output', default='data/canonical')
    p.add_argument('--modes', type=int, default=1024)
    args = p.parse_args()
    vertices, faces = np.load(args.vertices, allow_pickle=False), np.load(args.faces, allow_pickle=False)
    if len(vertices) <= args.modes + 1:
        raise ValueError('Mesh must contain more vertices than requested eigenpairs.')
    stiffness, mass = cotangent_matrices(vertices, faces)
    values, vectors = eigsh(stiffness, k=args.modes + 1, M=mass, sigma=-1e-8, which='LM', v0=np.ones(len(vertices)))
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if abs(values[0]) > 1e-6 or values[1] <= 1e-10:
        raise ValueError('Expected a connected mesh with exactly one constant mode.')
    # Resolve eigenvector signs deterministically; retain these assets with checkpoints.
    for k in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f'{args.vertebra_level}_eigval3.npy', values)
    np.save(output / f'{args.vertebra_level}_eigvec3.npy', vectors)
    np.save(output / f'2000{args.vertebra_level}_nodes3.npy', vertices)
    np.save(output / f'{args.vertebra_level}_faces.npy', faces)

if __name__ == '__main__':
    main()
