from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .clustering import build_Z_mst_occurrences_components, _kruskal_mst_from_edges
from .delaunay import orderk_delaunay3
from .geometry import kth_radius, minimum_enclosing_ball
from .union_find import UnionFind

N_CPU = max(1, os.cpu_count() or 1)


def _build_graph_KSimplexes(
    M: np.ndarray,
    K: int,
    min_samples: int,
    metric: str,
    complex_chosen: str,
    expZ: float,
    precision: str = "safe",
    verbose: bool = False,
    cgal_root: str | os.PathLike[str] | None = "../../CGALDelaunay",
) -> tuple[list[list[int]], list[int], list[int], list[float], int]:
    if min_samples is None or min_samples <= K:
        min_samples = K + 1
    pre = metric == "precomputed"
    delaunay_possible = not pre and metric == "euclidean" and M.ndim == 2 and M.shape[0] != M.shape[1]
    n, d = M.shape
    if complex_chosen.lower() not in {"orderk_delaunay", "delaunay", "rips"}:
        if not delaunay_possible:
            complex_chosen = "rips"
        else:
            if d > 10 and n > 100:
                complex_chosen = "rips"
            elif d > 10:
                complex_chosen = "delaunay"
            elif d > 5 and n > 1000:
                complex_chosen = "rips"
            else:
                complex_chosen = "orderk_delaunay"
    Simplexes: list[tuple[list[int], float]] = []
    root_path = Path(cgal_root) if cgal_root is not None else None
    if complex_chosen.lower() == "orderk_delaunay":
        simplexes = orderk_delaunay3(M, min_samples - 1, precision=precision, verbose=verbose, root=root_path)
        if verbose:
            print(f"Simplexes sans filtration : {len(simplexes)}")
        if simplexes:
            def _sqr_radius(simplex: Sequence[int]) -> float:
                pts = M[np.asarray(simplex, dtype=np.int64)]
                _, radius_sq = minimum_enclosing_ball(pts)
                return radius_sq

            radii_sq = Parallel(n_jobs=N_CPU, prefer="processes")(
                delayed(_sqr_radius)(s) for s in simplexes
            )
            if expZ != 2:
                radii_sq = np.asarray(radii_sq, dtype=np.float64) ** (expZ / 2)
            Simplexes = [(list(s), float(radii_sq[i])) for i, s in enumerate(simplexes)]
    else:
        import gudhi

        r = kth_radius(M, min_samples - 1, metric, pre)
        r2 = r**2
        if complex_chosen.lower() == "rips":
            r2 = r
            expZ_local = expZ * 2
            if precision == "exact":
                mx = 2 * np.quantile(r, 0.99)
            else:
                mx = (1 + 1 / math.sqrt(d)) * np.quantile(r, 0.99)
            if pre or metric != "euclidean":
                D = M if pre else pairwise_distances(M, metric=metric)
                st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=mx).create_simplex_tree(max_dimension=K)
            else:
                st = gudhi.RipsComplex(points=M, max_edge_length=mx).create_simplex_tree(max_dimension=K)
        else:
            expZ_local = expZ
            st = gudhi.DelaunayCechComplex(points=M).create_simplex_tree()
        for simplex, filt in st.get_skeleton(K):
            if len(simplex) != K + 1:
                continue
            simplex = list(sorted(simplex))
            max_kth_radius2 = max(r2[p] for p in simplex)
            filt = max(filt, max_kth_radius2)
            if expZ_local != 2:
                filt = filt ** (expZ_local / 2)
            Simplexes.append((simplex, float(filt)))
    faces_raw: list[list[int]] = []
    e_u: list[int] = []
    e_v: list[int] = []
    e_w: list[float] = []
    nS = 0
    for simplex, weight in Simplexes:
        if len(simplex) <= K:
            continue
        for vertices in itertools.combinations(range(len(simplex)), K + 1):
            nS += 1
            vert = tuple(sorted(vertices))
            base = len(faces_raw)
            for drop in range(K + 1):
                face = [simplex[vert[i]] for i in range(K + 1) if i != drop]
                faces_raw.append(face)
            for idx in range(K):
                e_u.append(base + idx)
                e_v.append(base + idx + 1)
                e_w.append(float(weight))
    return faces_raw, e_u, e_v, e_w, nS


def HypergraphPercol(
    M: np.ndarray,
    K: int = 2,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    metric: str = "euclidean",
    DBSCAN_threshold: float | None = None,
    label_all_points: bool = False,
    return_multi_clusters: bool = False,
    complex_chosen: str = "auto",
    expZ: float = 2,
    precision: str = "safe",
    dim_reducer: bool | str = False,
    threshold_variance_dim_reduction: float = 0.999,
    verbeux: bool = False,
    cgal_root: str | os.PathLike[str] | None = None,
) -> np.ndarray | tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
    n, d = M.shape
    M = np.ascontiguousarray(M, dtype=np.float64)
    if min_cluster_size is None:
        min_cluster_size = round(math.sqrt(n))
    X = np.copy(M)
    pre = metric == "precomputed"
    delaunay_possible = not pre and metric == "euclidean" and M.ndim == 2 and M.shape[0] != M.shape[1]
    if min_samples is None or min_samples <= K:
        min_samples = K + 1
    if str(dim_reducer).lower() in {"pca", "umap"} and delaunay_possible:
        pca = PCA(n_components=threshold_variance_dim_reduction, svd_solver="full", whiten=False)
        X2 = pca.fit_transform(M)
        r = pca.n_components_
        ratio = pca.explained_variance_ratio_.sum()
        if r < d and str(dim_reducer).lower() == "pca":
            X = X2
            if verbeux:
                print(f"Dimension réduite par PCA : {d} → {r} (variance {ratio:.3f})")
        elif r < d and str(dim_reducer).lower() == "umap":
            from umap import UMAP

            reducer = UMAP(n_components=r, n_neighbors=max(2 * 2 * (K + 1), min_samples), metric=metric)
            X = reducer.fit_transform(M)
            if verbeux:
                print(f"Dimension réduite par UMAP : {d} → {r}")
    faces_raw, e_u, e_v, e_w, nS = _build_graph_KSimplexes(
        X,
        K,
        min_samples,
        metric,
        complex_chosen,
        expZ,
        precision=precision,
        verbose=verbeux,
        cgal_root=cgal_root,
    )
    if verbeux:
        print(f"{K}-simplices={nS}")
    if not faces_raw:
        if K > d:
            print("Warning: K too high compared to the dimension of the data. No clustering possible with such a K.")
        if return_multi_clusters:
            return np.full(n, -1, dtype=np.int64), [(-1, 1.0, 1.0)] * n
        return np.full(n, 0, dtype=np.int64)
    faces_raw_arr = np.asarray(faces_raw, dtype=np.int64, order="C")
    e_u_arr = np.asarray(e_u, dtype=np.int64)
    e_v_arr = np.asarray(e_v, dtype=np.int64)
    e_w_arr = np.asarray(e_w, dtype=np.float64)
    faces_unique, inv = np.unique(faces_raw_arr, axis=0, return_inverse=True)
    if verbeux:
        print(f"Faces uniques: {faces_unique.shape[0]} (compression {faces_raw_arr.shape[0]}→{faces_unique.shape[0]})")
    u = inv[e_u_arr]
    v = inv[e_v_arr]
    w = e_w_arr
    uu = np.minimum(u, v)
    vv = np.maximum(u, v)
    order = np.lexsort((vv, uu))
    uu = uu[order]
    vv = vv[order]
    ww = w[order]
    change = np.r_[True, (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1])]
    gidx = np.flatnonzero(change)
    ww = np.minimum.reduceat(ww, gidx)
    uu = uu[gidx]
    vv = vv[gidx]
    if verbeux:
        print(f"Arêtes uniques (u<v): {uu.size} (avant dédup {u.size})")
    UF_faces = UnionFind(faces_unique.shape[0])
    mst_faces_sorted = _kruskal_mst_from_edges(faces_unique.shape[0], uu, vv, ww, UF_faces)
    if verbeux:
        m = faces_unique.shape[0]
        e_mst = len(mst_faces_sorted)
        comps = max(0, m - e_mst) if m else 0
        print(f"MST faces: {e_mst} arêtes, composantes estimées: {comps}")
    labels_points_unique, labels_points_multiple = build_Z_mst_occurrences_components(
        faces_unique,
        mst_faces_sorted,
        min_cluster_size=min_cluster_size,
        verbose=verbeux,
        distinct_mode="owner",
        DBSCAN_threshold=DBSCAN_threshold,
    )
    labels_points_unique = np.asarray(labels_points_unique)

    def knn_fill_weighted(X_data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        from sklearn.neighbors import KNeighborsClassifier

        X_data = np.asarray(X_data)
        y = labels.copy()
        mask_u = y == -1
        if not mask_u.any():
            return y
        mask_l = ~mask_u
        if not mask_l.any():
            return y
        k = min(k, int(mask_l.sum()))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        clf.fit(X_data[mask_l], y[mask_l])
        y[mask_u] = clf.predict(X_data[mask_u])
        return y

    if label_all_points and delaunay_possible:
        labels_points_unique = knn_fill_weighted(M, labels_points_unique, min_samples)
    if return_multi_clusters:
        return labels_points_unique, labels_points_multiple
    return labels_points_unique
