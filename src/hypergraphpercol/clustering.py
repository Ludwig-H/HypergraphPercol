from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .union_find import UnionFind

try:  # pragma: no cover - prefer compiled implementation
    from ._cython import build_leaf_dfs_intervals as _cython_build_leaf_dfs_intervals  # type: ignore
except ImportError:  # pragma: no cover - fallback used when extension unavailable
    _cython_build_leaf_dfs_intervals = None


def tree_to_labels(
    single_linkage_tree: np.ndarray,
    *,
    min_cluster_size: int = 20,
    DBSCAN_threshold: float | None = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = True,
    match_reference_implementation: bool = False,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_persistence: float = 0.0,
    max_cluster_size: int = 0,
    cluster_selection_epsilon_max: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from hdbscan._hdbscan_tree import condense_tree, compute_stability, get_clusters
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "hdbscan is required to convert trees into cluster labels"
        ) from exc

    condensed = condense_tree(single_linkage_tree, min_cluster_size)
    stability = compute_stability(condensed)
    if DBSCAN_threshold is None:
        labels, probabilities, stabilities = get_clusters(
            condensed,
            stability,
            cluster_selection_method,
            allow_single_cluster,
            match_reference_implementation,
            cluster_selection_epsilon,
            max_cluster_size,
            cluster_selection_epsilon_max,
        )
    else:
        labels, probabilities, stabilities = get_clusters(
            condensed,
            stability,
            "leaf",
            allow_single_cluster,
            match_reference_implementation,
            DBSCAN_threshold,
            max_cluster_size,
            DBSCAN_threshold,
        )
    return labels, probabilities, stabilities, condensed, single_linkage_tree


def _build_leaf_dfs_intervals_python(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = int(left.size)
    m = t + 1
    n_nodes = m + t
    first = np.full(n_nodes, -1, dtype=np.int64)
    last = np.full(n_nodes, -1, dtype=np.int64)
    leaf_order = np.empty(m, dtype=np.int64)
    root = m + t - 1
    stack: list[tuple[int, int]] = [(int(root), 0)]
    k = 0
    while stack:
        node, state = stack.pop()
        if node < m:
            first[node] = last[node] = k
            leaf_order[k] = node
            k += 1
        else:
            idx = node - m
            if state == 0:
                stack.append((node, 1))
                stack.append((int(right[idx]), 0))
                stack.append((int(left[idx]), 0))
            else:
                a = int(left[idx])
                b = int(right[idx])
                fa = int(first[a])
                fb = int(first[b])
                la = int(last[a])
                lb = int(last[b])
                if fa == -1 or fb == -1:
                    raise RuntimeError("Invalid tree: child interval missing")
                first[node] = fa if fa <= fb else fb
                last[node] = la if la >= lb else lb
    if k != m:
        raise RuntimeError("DFS traversal did not visit all leaves")
    pos = np.empty(m, dtype=np.int64)
    pos[leaf_order] = np.arange(m, dtype=np.int64)
    return pos, first, last, leaf_order


def build_leaf_dfs_intervals(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left64 = np.asarray(left, dtype=np.int64)
    right64 = np.asarray(right, dtype=np.int64)
    if _cython_build_leaf_dfs_intervals is not None:
        return _cython_build_leaf_dfs_intervals(left64, right64)
    return _build_leaf_dfs_intervals_python(left64, right64)


def prune_linkage_by_inclusion(Z_full: np.ndarray, K: int, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z_full, dtype=np.float64, order="C")
    t = int(Z.shape[0])
    m = t + 1
    if m <= 1:
        return np.zeros((0, 4), np.float64), np.array([0], dtype=np.int64)
    left = Z[:, 0].astype(np.int64, copy=False)
    right = Z[:, 1].astype(np.int64, copy=False)
    weight = Z[:, 2].astype(np.float64, copy=False)
    usize = np.rint(Z[:, 3]).astype(np.int64, copy=False)
    for i in range(t):
        parent = m + i
        a = int(left[i])
        b = int(right[i])
        if not (0 <= a < parent and 0 <= b < parent):
            raise ValueError("SciPy linkage convention violated: child >= parent")
    n_nodes = m + t
    size_node = np.zeros(n_nodes, dtype=np.int64)
    size_node[:m] = int(K)
    size_node[m:] = usize
    inc = np.zeros(t, dtype=bool)
    for i in range(t):
        a = int(left[i])
        b = int(right[i])
        p = m + i
        sa = int(size_node[a])
        sb = int(size_node[b])
        su = int(size_node[p])
        inc[i] = su == (sa if sa >= sb else sb)
    pos, first, last, leaf_order = build_leaf_dfs_intervals(left, right)
    UF = UnionFind(m)
    nxt = np.arange(m + 1, dtype=np.int64)

    def _next(idx: int) -> int:
        j = idx
        while nxt[j] != j:
            nxt[j] = nxt[nxt[j]]
            j = nxt[j]
        return j

    def union_interval(L: int, R: int, to_pos: int) -> None:
        root_to = UF.find(to_pos)
        i = _next(L)
        while i <= R:
            UF.union(i, root_to)
            nxt[i] = _next(i + 1)
            i = nxt[i]

    for i in range(t):
        if not inc[i]:
            continue
        a = int(left[i])
        b = int(right[i])
        sa = int(size_node[a])
        sb = int(size_node[b])
        winner = a if sa >= sb else b
        loser = b if sa >= sb else a
        Lw, Rw = int(first[winner]), int(last[winner])
        Ll, Rl = int(first[loser]), int(last[loser])
        rep_pos = UF.find(Lw)
        union_interval(Ll, Rl, rep_pos)

    rep_pos = np.array([UF.find(i) for i in range(m)], dtype=np.int64)
    uniq_rep, inverse = np.unique(rep_pos, return_inverse=True)
    Lp = int(uniq_rep.size)
    cls_of_pos = inverse
    min_pos_per_class = np.full(Lp, m + 1, dtype=np.int64)
    for p in range(m):
        c = int(cls_of_pos[p])
        if p < min_pos_per_class[c]:
            min_pos_per_class[c] = p
    surv_idx = leaf_order[min_pos_per_class]
    if Lp <= 1:
        return np.zeros((0, 4), np.float64), surv_idx
    U: list[int] = []
    V: list[int] = []
    W: list[float] = []
    S: list[float] = []
    for i in range(t):
        if inc[i]:
            continue
        a = int(left[i])
        b = int(right[i])
        La = int(first[a])
        Lb = int(first[b])
        ca = int(cls_of_pos[UF.find(La)])
        cb = int(cls_of_pos[UF.find(Lb)])
        if ca == cb:
            continue
        if ca > cb:
            ca, cb = cb, ca
        U.append(ca)
        V.append(cb)
        W.append(float(weight[i]))
        S.append(float(usize[i]))
    if not U:
        raise RuntimeError("No non-inclusive edges between classes")
    U_arr = np.asarray(U, dtype=np.int64)
    V_arr = np.asarray(V, dtype=np.int64)
    W_arr = np.asarray(W, dtype=np.float64)
    S_arr = np.asarray(S, dtype=np.float64)
    UFc = UnionFind(Lp)
    comp_id = np.arange(Lp, dtype=np.int64)
    Z_pruned = np.empty((Lp - 1, 4), dtype=np.float64)
    created = 0
    for i in range(U_arr.size):
        ru = UFc.find(int(U_arr[i]))
        rv = UFc.find(int(V_arr[i]))
        if ru == rv:
            continue
        ida = int(comp_id[ru])
        idb = int(comp_id[rv])
        if ida > idb:
            ida, idb = idb, ida
        Z_pruned[created, 0] = float(ida)
        Z_pruned[created, 1] = float(idb)
        Z_pruned[created, 2] = float(W_arr[i])
        Z_pruned[created, 3] = float(S_arr[i])
        UFc.union(ru, rv)
        root = UFc.find(ru)
        comp_id[root] = Lp + created
        created += 1
        if created == Lp - 1:
            break
    if created != Lp - 1:
        raise RuntimeError("Incomplete quotient after pruning")
    if verbose:
        print(f"[PRUNE] leaves={m}, classes={Lp}, Z_pruned shape={Z_pruned.shape}")
    return Z_pruned, surv_idx


def _kruskal_mst_from_edges(n_nodes: int, rows: Sequence[int], cols: Sequence[int], weights: Sequence[float], UF: UnionFind) -> list[tuple[int, int, float]]:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    if rows.size == 0:
        return []
    order = np.argsort(weights, kind="stable")
    rows = rows[order]
    cols = cols[order]
    weights = weights[order]
    mst_u = np.empty(max(0, n_nodes - 1), dtype=np.int64)
    mst_v = np.empty_like(mst_u)
    mst_w = np.empty_like(mst_u, dtype=float)
    taken = 0
    for i in range(rows.size):
        a = int(rows[i])
        b = int(cols[i])
        ra = UF.find(a)
        rb = UF.find(b)
        if ra == rb:
            continue
        mst_u[taken] = a
        mst_v[taken] = b
        mst_w[taken] = float(weights[i])
        UF.union(ra, rb)
        taken += 1
        if taken == mst_u.size:
            break
    return list(zip(mst_u[:taken].tolist(), mst_v[:taken].tolist(), mst_w[:taken].tolist()))


def build_Z_mst_occurrences_gc(
    face_vertices: np.ndarray,
    mst_faces_sorted: Iterable[tuple[int, int, float]] | tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    min_cluster_size: int,
    verbose: bool = False,
    distinct_mode: str = "owner",
    DBSCAN_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _fmt_bytes(n: float) -> str:
        n = float(n)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if n < 1024 or unit == "GiB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GiB"

    def _rss() -> float | None:
        try:
            import psutil

            return float(psutil.Process().memory_info().rss)
        except Exception:
            return None

    F = np.asarray(face_vertices, dtype=np.int64, order="C")
    m = int(F.shape[0])
    if m == 0:
        return np.zeros((0, 4), np.float64), np.zeros(0, np.int64), F
    K = int(F.shape[1])
    if isinstance(mst_faces_sorted, tuple) and len(mst_faces_sorted) == 3:
        u_arr = np.asarray(mst_faces_sorted[0], dtype=np.int64)
        v_arr = np.asarray(mst_faces_sorted[1], dtype=np.int64)
        w_arr = np.asarray(mst_faces_sorted[2], dtype=np.float64)
    else:
        data = list(mst_faces_sorted)
        if data:
            u_arr = np.fromiter((u for u, _, _ in data), count=len(data), dtype=np.int64)
            v_arr = np.fromiter((v for _, v, _ in data), count=len(data), dtype=np.int64)
            w_arr = np.fromiter((w for _, _, w in data), count=len(data), dtype=np.float64)
        else:
            u_arr = v_arr = np.empty(0, dtype=np.int64)
            w_arr = np.empty(0, dtype=np.float64)
    if u_arr.size == 0:
        return np.zeros((0, 4), np.float64), np.zeros(m, np.int64), F
    order = np.argsort(w_arr, kind="stable")
    u_arr = u_arr[order]
    v_arr = v_arr[order]
    w_arr = w_arr[order]
    next_face = np.full(m, -1, dtype=np.int64)
    head = np.arange(m, dtype=np.int64)
    tail = np.arange(m, dtype=np.int64)
    comp_sz = np.ones(m, dtype=np.int64)
    UF = UnionFind(m)
    cid = np.arange(m, dtype=np.int64)
    nxt = int(m)
    n_points = int(F.max()) + 1
    owner_root = np.full(n_points, -1, dtype=np.int64)
    distinct_count = np.zeros(m, dtype=np.int64)
    for f in range(m):
        row = F[f]
        for j in range(K):
            v = int(row[j])
            if owner_root[v] == -1:
                owner_root[v] = f
                distinct_count[f] += 1
    Z = np.empty((max(0, m - 1), 4), dtype=np.float64)
    taken = 0
    for a, b, w in zip(u_arr, v_arr, w_arr):
        ra = UF.find(int(a))
        rb = UF.find(int(b))
        if ra == rb:
            continue
        sa = int(comp_sz[ra])
        sb = int(comp_sz[rb])
        small, big = (ra, rb) if sa < sb else (rb, ra)
        root_big = UF.find(big)
        uniq_add = 0
        i = int(head[small])
        while i != -1:
            row = F[i]
            for j in range(K):
                v = int(row[j])
                pr = owner_root[v]
                if pr == -1 or UF.find(int(pr)) != root_big:
                    owner_root[v] = root_big
                    uniq_add += 1
            i = int(next_face[i])
        new_count = int(distinct_count[root_big]) + int(uniq_add)
        Z[taken, 0] = float(cid[big])
        Z[taken, 1] = float(cid[small])
        Z[taken, 2] = float(w)
        Z[taken, 3] = float(new_count)
        taken += 1
        if head[small] != -1:
            if head[big] == -1:
                head[big] = head[small]
                tail[big] = tail[small]
            else:
                next_face[int(tail[big])] = int(head[small])
                tail[big] = int(tail[small])
            comp_sz[big] = comp_sz[big] + comp_sz[small]
            head[small] = -1
            tail[small] = -1
            comp_sz[small] = 0
        UF.union(big, small)
        root = UF.find(big)
        head[root] = head[big]
        tail[root] = tail[big]
        comp_sz[root] = comp_sz[big]
        distinct_count[root] = new_count
        cid[root] = nxt
        nxt += 1
        if taken == m - 1:
            break
    Z = Z if taken == Z.shape[0] else Z[:taken].copy()
    Z_pruned, surv_idx = prune_linkage_by_inclusion(Z_full=Z, K=K, verbose=verbose)
    if Z_pruned.shape[0] <= 1 or Z_pruned[-1, 3] <= min_cluster_size:
        return Z_pruned, np.full(m, -2, dtype=np.int64), np.zeros(m, np.float64), F
    labels_faces, probabilities_faces, *_ = tree_to_labels(
        single_linkage_tree=Z_pruned,
        min_cluster_size=min_cluster_size,
        DBSCAN_threshold=DBSCAN_threshold,
    )
    ret_labels = np.full(m, -2, dtype=np.int64)
    ret_prob = np.zeros(m, dtype=np.float64)
    ret_labels[surv_idx] = labels_faces
    ret_prob[surv_idx] = probabilities_faces
    return Z, ret_labels, ret_prob, F


def build_Z_mst_occurrences_components(
    face_vertices: np.ndarray,
    mst_faces_sorted: Sequence[tuple[int, int, float]],
    *,
    min_cluster_size: int,
    verbose: bool = False,
    distinct_mode: str = "owner",
    DBSCAN_threshold: float | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    def _fmt_bytes(n: float) -> str:
        n = float(n)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if n < 1024 or unit == "GiB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
            n /= 1024
        return f"{n:.1f} GiB"

    def _rss() -> float | None:
        try:
            import psutil

            return float(psutil.Process().memory_info().rss)
        except Exception:
            return None

    face_vertices = np.asarray(face_vertices, dtype=np.int64, order="C")
    n_faces = int(face_vertices.shape[0])
    if n_faces == 0:
        return np.zeros(0, dtype=np.int64), []
    UF_face = UnionFind(n_faces)
    for u, v, _ in mst_faces_sorted:
        UF_face.union(u, v)
    comp_labels = np.fromiter((UF_face.find(i) for i in range(n_faces)), count=n_faces, dtype=np.int64)
    order_faces = np.argsort(comp_labels, kind="mergesort")
    labels_sorted = comp_labels[order_faces]
    diff_faces = labels_sorted[1:] != labels_sorted[:-1]
    starts = np.r_[0, 1 + np.flatnonzero(diff_faces)]
    ends = np.r_[starts[1:], labels_sorted.size]
    uniq = labels_sorted[starts]
    faces_ordered = np.arange(n_faces, dtype=np.int64)[order_faces]
    n_points_total = int(face_vertices.max()) + 1
    labels_points_unique = np.full(n_points_total, -1, dtype=np.int64)
    labels_points_multiple: list[list[tuple[int, float]]] = [[] for _ in range(n_points_total)]
    first = np.full(n_points_total, -2, dtype=np.int64)
    conflict = np.zeros(n_points_total, dtype=bool)
    next_cluster_id = 0
    if verbose:
        r = _rss()
        if r is not None:
            print(
                f"[COMP-F:0] components={uniq.size}, faces={n_faces}, points={n_points_total}, RSS={_fmt_bytes(r)}"
            )
    if len(mst_faces_sorted):
        u_arr = np.asarray([uv[0] for uv in mst_faces_sorted], dtype=np.int64)
        v_arr = np.asarray([uv[1] for uv in mst_faces_sorted], dtype=np.int64)
        w_arr = np.asarray([uv[2] for uv in mst_faces_sorted], dtype=np.float64)
        comp_u = comp_labels[u_arr]
        order_edges = np.argsort(comp_u, kind="mergesort")
        comp_u_sorted = comp_u[order_edges]
        diff_edges = comp_u_sorted[1:] != comp_u_sorted[:-1]
        starts_e = np.r_[0, 1 + np.flatnonzero(diff_edges)]
        ends_e = np.r_[starts_e[1:], comp_u_sorted.size]
        uniq_e = comp_u_sorted[starts_e]
    else:
        u_arr = v_arr = np.empty(0, dtype=np.int64)
        w_arr = np.empty(0, dtype=np.float64)
        order_edges = np.empty(0, dtype=np.int64)
        starts_e = ends_e = uniq_e = np.empty(0, dtype=np.int64)
    for j in range(uniq.size):
        f_start, f_end = int(starts[j]), int(ends[j])
        faces = faces_ordered[f_start:f_end]
        if faces.size == 0:
            continue
        comp_id = int(uniq[j])
        if order_edges.size:
            pos = np.searchsorted(uniq_e, comp_id)
            if pos < uniq_e.size and uniq_e[pos] == comp_id:
                e_start = int(starts_e[pos])
                e_end = int(ends_e[pos])
                idx_edges = order_edges[e_start:e_end]
            else:
                idx_edges = np.empty(0, dtype=np.int64)
        else:
            idx_edges = np.empty(0, dtype=np.int64)
        faces_sorted = np.sort(faces)
        u_sel = u_arr[idx_edges]
        v_sel = v_arr[idx_edges]
        w_sel = w_arr[idx_edges]
        new_u = np.searchsorted(faces_sorted, u_sel).astype(np.int64, copy=False)
        new_v = np.searchsorted(faces_sorted, v_sel).astype(np.int64, copy=False)
        faces_compact = face_vertices[faces_sorted]
        if verbose:
            r = _rss()
            est_Z = max(0, faces_compact.shape[0] - 1) * 32
            print(
                f"[COMP-F:1] comp {j + 1}/{uniq.size} | faces={faces_compact.shape[0]}, edges={new_u.size}, "
                f"Z_est≈{_fmt_bytes(est_Z)}, RSS={_fmt_bytes(r) if r else 'n/a'}"
            )
        Z_i, labels_faces_i, probabilities_faces_i, F_i = build_Z_mst_occurrences_gc(
            faces_compact,
            (new_u, new_v, w_sel),
            min_cluster_size=min_cluster_size,
            verbose=verbose,
            distinct_mode=distinct_mode,
            DBSCAN_threshold=DBSCAN_threshold,
        )
        if len(labels_faces_i) and np.any(labels_faces_i != -1):
            max_local = int(labels_faces_i[labels_faces_i != -1].max())
            offset = next_cluster_id
            for f_loc in range(F_i.shape[0]):
                lbl = int(labels_faces_i[f_loc])
                if lbl in (-2, -1):
                    continue
                lbl_g = lbl + offset
                row = F_i[f_loc]
                for t in range(row.size):
                    v = int(row[t])
                    labels_points_multiple[v].append((lbl_g, probabilities_faces_i[f_loc]))
                    if first[v] == -2:
                        first[v] = lbl_g
                    elif first[v] != lbl_g:
                        conflict[v] = True
            next_cluster_id = offset + max_local + 1
        if verbose and (j % 1 == 0):
            valid = (first != -2) & (~conflict)
            r = _rss()
            print(
                f"[COMP-F:2] comp {j + 1}/{uniq.size} done | cumul points labellisés={int(valid.sum())} "
                f"| conflits={int(conflict.sum())} | RSS={_fmt_bytes(r) if r else 'n/a'}"
            )
    mask_ok = (~conflict) & (first != -2)
    labels_points_unique[mask_ok] = first[mask_ok]
    ret_labels_points_multiple: list[list[tuple[int, float, float]]] = [[] for _ in range(n_points_total)]
    for v, labels in enumerate(labels_points_multiple):
        if not labels:
            ret_labels_points_multiple[v] = [(-1, 1.0, 1.0)]
            continue
        grouped: defaultdict[int, list[float]] = defaultdict(list)
        for lbl, prob in labels:
            grouped[lbl].append(prob)
        aggregated: list[tuple[int, float, float]] = []
        total = float(len(labels))
        for lbl, probs in grouped.items():
            count = len(probs)
            aggregated.append((lbl, count / total, float(np.mean(probs))))
        ret_labels_points_multiple[v] = aggregated
    if verbose:
        uvals = np.unique(labels_points_unique)
        r = _rss()
        print(
            "Clusters finaux :",
            uvals[uvals != -1].size,
            "| bruit :",
            int(np.sum(labels_points_unique == -1)),
            f"| RSS={_fmt_bytes(r) if r else 'n/a'}",
        )
    return labels_points_unique, ret_labels_points_multiple
