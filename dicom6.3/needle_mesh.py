import numpy as np


_LENGTH_EPS_MM = 1e-12
_DIR_EPS = 1e-15
_SEARCH_CLASSES = ("CTV", "PTV", "OAR")


def _channel_sort_key(channel_id):
    try:
        return (0, int(channel_id))
    except (TypeError, ValueError):
        return (1, str(channel_id))


def _as_xyz(values):
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return arr


def structure_search_bounds(masks, origin, spacing, names=None):
    """Axis-aligned bounding box of selected structure masks, in world mm.

    The box is the min/max of occupied voxel centres, expanded by half a
    voxel so the full voxel is inside the search space.

    Parameters
    ----------
    masks : dict[str, ndarray]
        Binary structure masks on the image grid, shape (nz, ny, nx).
    origin : array-like (3,)
        World coordinate of the first voxel (x0, y0, z0).
    spacing : array-like (3,)
        Voxel size (dx, dy, dz) in mm.
    names : sequence of str or None
        Mask keys to include. None uses every mask.

    Returns
    -------
    xyz_min, xyz_max : ndarray (3,)
        Inclusive world-coordinate bounds. Both are None if no voxels
        are occupied.
    """
    origin = _as_xyz(origin)
    spacing = _as_xyz(spacing)
    keys = list(masks.keys()) if names is None else list(names)
    combined = None
    for name in keys:
        mask = masks.get(name)
        if mask is None:
            continue
        combined = mask if combined is None else (combined | mask)

    if combined is None or not np.any(combined):
        return None, None

    iz, iy, ix = np.where(combined)
    xs = origin[0] + ix.astype(np.float64) * spacing[0]
    ys = origin[1] + iy.astype(np.float64) * spacing[1]
    zs = origin[2] + iz.astype(np.float64) * spacing[2]
    xyz_min = np.array([xs.min(), ys.min(), zs.min()], dtype=np.float64)
    xyz_max = np.array([xs.max(), ys.max(), zs.max()], dtype=np.float64)
    half = 0.5 * np.abs(spacing)
    return xyz_min - half, xyz_max + half


def line_aabb_intersection(point, direction, box_min, box_max):
    """Intersect the infinite line ``point + t * direction`` with an AABB.

    ``direction`` must be a unit vector. Returns ``(t_enter, t_exit)`` in
    millimetres along the line, or ``(None, None)`` if there is no hit.
    """
    p0 = _as_xyz(point)
    d = _as_xyz(direction)
    bmin = _as_xyz(box_min)
    bmax = _as_xyz(box_max)

    t_enter = -np.inf
    t_exit = np.inf
    for i in range(3):
        if abs(d[i]) < _DIR_EPS:
            if p0[i] < bmin[i] or p0[i] > bmax[i]:
                return None, None
            continue
        t1 = (bmin[i] - p0[i]) / d[i]
        t2 = (bmax[i] - p0[i]) / d[i]
        t_lo, t_hi = (t1, t2) if t1 <= t2 else (t2, t1)
        t_enter = max(t_enter, t_lo)
        t_exit = min(t_exit, t_hi)
        if t_enter > t_exit:
            return None, None
    if not np.isfinite(t_enter) or not np.isfinite(t_exit):
        return None, None
    return float(t_enter), float(t_exit)


def mesh_channel_in_search_space(channel_start, channel_end, box_min, box_max,
                                 mesh_step=1.0):
    """Mesh one needle along the line through its start and end points.

    Current plan dwells are ignored. The catheter is the infinite straight
    line through ``channel_start`` and ``channel_end``. Candidate vertices
    are the samples of that line that lie inside the PTV/OAR search box.

    Parameters
    ----------
    channel_start, channel_end : array-like (3,)
        World coordinates of the needle endpoints from the RTPLAN channel.
    box_min, box_max : array-like (3,)
        Search-space AABB in world mm.
    mesh_step : float
        Spacing between mesh points (mm).

    Returns
    -------
    mesh_points : ndarray (K, 3)
        World-coordinate samples along the needle, inside the search box.
    t_enter, t_exit : float or None
        Arc-length parameters (mm from ``channel_start``) where the line
        enters and leaves the search box.
    """
    p0 = _as_xyz(channel_start)
    p1 = _as_xyz(channel_end)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length <= _LENGTH_EPS_MM or mesh_step <= 0.0:
        return np.empty((0, 3), dtype=np.float64), None, None

    direction = vec / length
    t_enter, t_exit = line_aabb_intersection(p0, direction, box_min, box_max)
    if t_enter is None or t_exit is None or t_exit < t_enter:
        return np.empty((0, 3), dtype=np.float64), None, None

    n_step = int(np.floor((t_exit - t_enter) / mesh_step + 1e-9)) + 1
    t_mesh = t_enter + np.arange(n_step, dtype=np.float64) * mesh_step
    if t_mesh[-1] < t_exit - 1e-9:
        t_mesh = np.append(t_mesh, t_exit)
    mesh_points = p0[np.newaxis, :] + t_mesh[:, np.newaxis] * direction[np.newaxis, :]
    return mesh_points, t_enter, t_exit


def _region_label(in_ctv, in_ptv, in_oar):
    in_target = bool(in_ctv or in_ptv)
    if in_ctv and in_oar:
        return "CTV+OAR"
    if in_ptv and in_oar:
        return "PTV+OAR"
    if in_target and in_oar:
        return "TARGET+OAR"
    if in_ctv:
        return "CTV"
    if in_ptv:
        return "PTV"
    if in_oar:
        return "OAR"
    return "OUTSIDE"


def label_mesh_points(mesh_points, masks, origin, spacing, structure_classes):
    """Label each mesh vertex by the structures that contain it.

    A vertex may sit in a target, an OAR, both, or neither (still inside
    the search box). OAR vertices are kept; they are not excluded.

    Parameters
    ----------
    mesh_points : ndarray (K, 3)
    masks : dict[str, ndarray]  — structure masks on the image grid
    origin, spacing : array-like (3,)
    structure_classes : dict[str, str]
        Structure name → ``"CTV"``, ``"PTV"``, ``"OAR"``, or other.

    Returns
    -------
    labels : list[dict]
        One dict per vertex with keys ``region``, ``organ_label``,
        ``structures``, ``in_ctv``, ``in_ptv``, ``in_oar``.
    """
    pts = np.asarray(mesh_points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return []

    origin = _as_xyz(origin)
    spacing = _as_xyz(spacing)
    vox = np.round((pts - origin) / spacing).astype(int)
    sample = next(iter(masks.values()), None)
    if sample is None:
        return [
            {
                "region": "OUTSIDE",
                "organ_label": "OUTSIDE",
                "structures": (),
                "in_ctv": False,
                "in_ptv": False,
                "in_oar": False,
            }
            for _ in range(n)
        ]

    nz, ny, nx = sample.shape
    ix, iy, iz = vox[:, 0], vox[:, 1], vox[:, 2]
    in_bounds = (
        (iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    )

    hits = {}
    for name, mask in masks.items():
        cls = structure_classes.get(name)
        if cls not in _SEARCH_CLASSES:
            continue
        hit = np.zeros(n, dtype=bool)
        valid = np.where(in_bounds)[0]
        if len(valid):
            hit[valid] = mask[iz[valid], iy[valid], ix[valid]]
        hits[name] = (hit, cls)

    labels = []
    for i in range(n):
        names = []
        in_ctv = in_ptv = in_oar = False
        for name, (hit, cls) in hits.items():
            if not hit[i]:
                continue
            names.append(name)
            if cls == "CTV":
                in_ctv = True
            elif cls == "PTV":
                in_ptv = True
            elif cls == "OAR":
                in_oar = True
        region = _region_label(in_ctv, in_ptv, in_oar)
        organ_label = "+".join(names) if names else "OUTSIDE"
        labels.append({
            "region": region,
            "organ_label": organ_label,
            "structures": tuple(names),
            "in_ctv": in_ctv,
            "in_ptv": in_ptv,
            "in_oar": in_oar,
        })
    return labels


def count_mesh_labels(labels):
    """Return region and organ occupancy counts for a list of mesh labels."""
    region_counts = {}
    organ_counts = {}
    for lab in labels:
        region = lab.get("region", "OUTSIDE")
        region_counts[region] = region_counts.get(region, 0) + 1
        names = lab.get("structures") or ()
        if not names:
            organ_counts["OUTSIDE"] = organ_counts.get("OUTSIDE", 0) + 1
            continue
        for name in names:
            organ_counts[name] = organ_counts.get(name, 0) + 1
    return region_counts, organ_counts


def mesh_point_directions(mesh_points):
    """Unit tangent at each vertex of an ordered catheter mesh."""
    pts = np.asarray(mesh_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("mesh_points must have shape (K, 3).")
    n = len(pts)
    dirs = np.zeros((n, 3), dtype=np.float64)
    if n == 0:
        return dirs
    if n == 1:
        dirs[0] = np.array([0.0, 0.0, 1.0])
        return dirs

    segs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(segs, axis=1)
    fallback = np.array([0.0, 0.0, 1.0])
    last = fallback
    for k in range(n - 1):
        if lengths[k] > _LENGTH_EPS_MM:
            last = segs[k] / lengths[k]
        dirs[k] = last
    dirs[-1] = last
    return dirs


def assemble_mesh_dwells(needle_meshes, needle_labels=None):
    """Flatten every needle mesh into candidate dwell positions.

    Each mesh vertex is one candidate, including vertices inside OARs.
    Catheter direction at a vertex is the unit tangent of the segment that
    starts there (the last vertex reuses the previous segment). Empty
    meshes are omitted.

    Parameters
    ----------
    needle_meshes : dict
        channel_id → (K, 3) mesh points.
    needle_labels : dict or None
        channel_id → list of K label dicts from ``label_mesh_points``.

    Returns
    -------
    dwells : dict
        ``positions`` (N, 3), ``directions`` (N, 3), ``channel_ids`` (N,),
        ``local_index`` (N,), ``labels`` (list of N dicts).
    """
    positions = []
    directions = []
    channel_ids = []
    local_index = []
    labels = []

    empty = {
        "positions": np.zeros((0, 3), dtype=np.float64),
        "directions": np.zeros((0, 3), dtype=np.float64),
        "channel_ids": np.zeros(0, dtype=int),
        "local_index": np.zeros(0, dtype=int),
        "labels": [],
    }

    for channel_id in sorted(needle_meshes.keys(), key=_channel_sort_key):
        pts = np.asarray(needle_meshes[channel_id], dtype=np.float64)
        if pts.size == 0:
            continue
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"Channel {channel_id}: mesh_points must have shape (K, 3)."
            )
        n = len(pts)
        positions.append(pts)
        directions.append(mesh_point_directions(pts))
        channel_ids.append(np.full(n, channel_id))
        local_index.append(np.arange(n, dtype=int))

        if needle_labels is None:
            labels.extend([
                {
                    "region": "OUTSIDE",
                    "organ_label": "OUTSIDE",
                    "structures": (),
                    "in_ctv": False,
                    "in_ptv": False,
                    "in_oar": False,
                }
                for _ in range(n)
            ])
        else:
            ch_labels = needle_labels.get(channel_id)
            if ch_labels is None:
                try:
                    ch_labels = needle_labels.get(int(channel_id))
                except (TypeError, ValueError):
                    ch_labels = None
            if ch_labels is None or len(ch_labels) != n:
                raise ValueError(
                    f"Channel {channel_id}: expected {n} mesh labels, "
                    f"got {0 if ch_labels is None else len(ch_labels)}."
                )
            labels.extend(ch_labels)

    if not positions:
        return empty

    return {
        "positions": np.vstack(positions),
        "directions": np.vstack(directions),
        "channel_ids": np.concatenate(channel_ids),
        "local_index": np.concatenate(local_index),
        "labels": labels,
    }
