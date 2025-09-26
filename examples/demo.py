"""Small demonstration of the HypergraphPercol clustering algorithm."""

from __future__ import annotations

import pathlib

import numpy as np

from hypergraphpercol import HypergraphPercol


def make_dataset(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blobs = []
    centers = [(-4, -4), (0, 0), (5, 4)]
    for cx, cy in centers:
        blobs.append(rng.normal(loc=(cx, cy), scale=0.7, size=(80, 2)))
    return np.vstack(blobs)


def main() -> None:
    data = make_dataset()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cgal_root = repo_root / "CGALDelaunay"

    # Prefer CGAL-based complexes when the helper binaries are available.
    # This keeps the example runnable in environments where the CGAL
    # repositories have not been cloned/built yet (e.g. CI containers).
    default_complex = "auto"
    helper_binary = (
        cgal_root
        / "EdgesCGALDelaunay2D"
        / "build"
        / "EdgesCGALDelaunay2D"
    )
    if not helper_binary.exists():
        default_complex = "rips"
        cgal_root = None

    labels = HypergraphPercol(
        data,
        K=2,
        min_cluster_size=20,
        min_samples=4,
        metric="euclidean",
        complex_chosen=default_complex,
        expZ=2,
        precision="safe",
        verbeux=False,
        cgal_root=cgal_root,
    )
    print("Cluster labels:")
    print(labels)


if __name__ == "__main__":
    main()
