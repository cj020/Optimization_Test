import numpy as np


_LENGTH_EPS_MM = 1e-12
_DIR_EPS = 1e-12
_TARGET_CLASSES = ("CTV", "PTV")
_OAR_CLASS = "OAR"


def structures_world_aabb(structures, names, margin=0.0):
    """Axis-aligned world bounding box of the named structures.

    Uses contour coordinates (x, y, z) so the box covers every PTV/CTV/OAR
    point. ``margin`` expands the box equally on all sides (mm).
    """
    name_set = set(names)
    xs, ys, zs = [], [], []
    used = []

    for s in structures:
        if s["name"] not in name_set:
            continue
        got = False
        for contour in s.get("contours", []):
            pts = np.asarray(contour["points"], dtype=np.float64)
            if pts.size == 0:
                continue
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
            zs.append(np.full(len(pts), float(contour["z"]), dtype=np.float64))
            got = True
        if got:
            used.append(s["name"])

    if not xs:
        return None

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    m = float(margin)
    return {
        "xmin": float(x.min()) - m,
        "xmax": float(x.max()) + m,
        "ymin": float(y.min()) - m,
        "ymax": float(y.max()) + m,
        "zmin": float(z.min()) - m,
        "zmax": float(z.max()) + m,
        "structures": used,
    }


def masks_world_aabb(masks, origin, spacing, names, margin=0.0):
    """Fallback AABB from rasterized masks when contour points are unavailable."""
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    combined = None
    used = []
    for name in names:
        if name not in masks:
            continue
        m = masks[name]
        if not np.any(m):
            continue
        combined = m if combined is None else (combined | m)
        used.append(name)

    if combined is None or not np.any(combined):
        return None

    iz, iy, ix = np.where(combined)
    xs = origin[0] + ix.astype(np.float64) * spacing[0]
    ys = origin[1] + iy.astype(np.float64) * spacing[1]
    zs = origin[2] + iz.astype(np.float64) * spacing[2]
    pad = float(margin)
    return {
        "xmin": float(xs.min()) - pad,
        "xmax": float(xs.max()) + pad,
        "ymin": float(ys.min()) - pad,
        "ymax": float(ys.max()) + pad,
        "zmin": float(zs.min()) - pad,
        "zmax": float(zs.max()) + pad,
        "structures": used,
    }


def search_space_aabb(structures, masks, origin, spacing,
                      names, margin=0.0):
    """Search space enclosing all named PTV/CTV/OAR structures."""
    box = structures_world_aabb(structures, names, margin=margin)
    if box is None:
        box = masks_world_aabb(masks, origin, spacing, names, margin=margin)
    return box


def _line_aabb_t_range(p0, direction, box):
    """Intersection of the infinite line p0 + t * direction with an AABB.

    Returns (t_enter, t_exit) in mm along the unit direction, or None if the
    line misses the box.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    t_enter = -np.inf
    t_exit = np.inf

    for i, ax in enumerate(("x", "y", "z")):
        lo = box[f"{ax}min"]
        hi = box[f"{ax}max"]
        if abs(d[i]) < _DIR_EPS:
            if p0[i] < lo or p0[i] > hi:
                return None
            continue
        t1 = (lo - p0[i]) / d[i]
        t2 = (hi - p0[i]) / d[i]
        t_lo, t_hi = (t1, t2) if t1 <= t2 else (t2, t1)
        t_enter = max(t_enter, t_lo)
        t_exit = min(t_exit, t_hi)
        if t_enter > t_exit:
            return None

    if not np.isfinite(t_enter) or not np.isfinite(t_exit):
        return None
    return float(t_enter), float(t_exit)


def mesh_needle_in_search_space(p_start, p_end, search_box, mesh_step=1.0):
    """Mesh one needle along the line through its start and end points.

    Existing dwell positions are ignored. Candidate vertices lie on the
    infinite line through ``p_start`` and ``p_end``, clipped to the PTV/OAR
    search-space box, and are spaced ``mesh_step`` mm apart. Points inside
    OAR are kept.

    Returns
    -------
    mesh_points : ndarray (K, 3)
    t_enter : float or None
        Arc-length parameter (mm from ``p_start``) where the line enters the box.
    t_exit : float or None
        Arc-length parameter where the line leaves the box.
    """
    p_start = np.asarray(p_start, dtype=np.float64)
    p_end = np.asarray(p_end, dtype=np.float64)
    vec = p_end - p_start
    length = np.linalg.norm(vec)
    if length < _LENGTH_EPS_MM or search_box is None:
        return np.empty((0, 3), dtype=np.float64), None, None

    direction = vec / length
    t_range = _line_aabb_t_range(p_start, direction, search_box)
    if t_range is None:
        return np.empty((0, 3), dtype=np.float64), None, None

    t_enter, t_exit = t_range
    if t_exit < t_enter:
        return np.empty((0, 3), dtype=np.float64), None, None

    step = float(mesh_step)
    t_mesh = np.arange(t_enter, t_exit + step / 2.0, step)
    if len(t_mesh) == 0:
        t_mesh = np.array([0.5 * (t_enter + t_exit)], dtype=np.float64)
    mesh_points = p_start[np.newaxis, :] + t_mesh[:, np.newaxis] * direction[np.newaxis, :]
    return mesh_points, t_enter, t_exit


def label_mesh_points(positions, masks, structures, origin, spacing):
    """Mark each mesh vertex with the PTV/CTV/OAR structures it falls in.

    Overlapping structures are all recorded. Vertices in OAR are not dropped.

    Returns
    -------
    labels : dict
        ``structure_names`` (N,) object array of name tuples,
        ``role`` (N,) in {TARGET, OAR, TARGET+OAR, NONE},
        ``in_target`` (N,) bool, ``in_oar`` (N,) bool.
    """
    positions = np.asarray(positions, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    n = len(positions)

    empty = {
        "structure_names": np.empty(0, dtype=object),
        "role": np.empty(0, dtype=object),
        "in_target": np.zeros(0, dtype=bool),
        "in_oar": np.zeros(0, dtype=bool),
    }
    if n == 0 or not masks:
        return empty

    class_of = {s["name"]: s.get("classification", "OTHER") for s in structures}
    shape = next(iter(masks.values())).shape
    nz, ny, nx = shape

    vox = np.round((positions - origin) / spacing).astype(int)
    ix, iy, iz = vox[:, 0], vox[:, 1], vox[:, 2]
    in_bounds = (iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    valid = np.where(in_bounds)[0]

    hits = [[] for _ in range(n)]
    in_target = np.zeros(n, dtype=bool)
    in_oar = np.zeros(n, dtype=bool)

    for name, mask in masks.items():
        cls = class_of.get(name, "OTHER")
        if cls not in _TARGET_CLASSES and cls != _OAR_CLASS:
            continue
        inside = np.zeros(n, dtype=bool)
        if len(valid):
            inside[valid] = mask[iz[valid], iy[valid], ix[valid]]
        for i in np.where(inside)[0]:
            hits[i].append((cls, name))
        if cls in _TARGET_CLASSES:
            in_target |= inside
        else:
            in_oar |= inside

    structure_names = np.empty(n, dtype=object)
    role = np.empty(n, dtype=object)
    for i in range(n):
        ordered = sorted(
            hits[i],
            key=lambda item: (0 if item[0] in _TARGET_CLASSES else 1, item[1]),
        )
        structure_names[i] = tuple(name for _, name in ordered)
        if in_target[i] and in_oar[i]:
            role[i] = "TARGET+OAR"
        elif in_target[i]:
            role[i] = "TARGET"
        elif in_oar[i]:
            role[i] = "OAR"
        else:
            role[i] = "NONE"

    return {
        "structure_names": structure_names,
        "role": role,
        "in_target": in_target,
        "in_oar": in_oar,
    }


def format_structure_label(names):
    """Compact display string for a vertex's structure tuple."""
    if not names:
        return "none"
    return "+".join(names)


def _channel_sort_key(channel_id):
    try:
        return (0, int(channel_id))
    except (TypeError, ValueError):
        return (1, str(channel_id))


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


def assemble_mesh_dwells(needle_meshes, labels=None):
    """Flatten every needle mesh into candidate dwell positions.

    Each mesh vertex is one candidate, including vertices in OAR. Catheter
    direction at a vertex is the unit tangent of the segment that starts
    there (the last vertex reuses the previous segment). Empty meshes are
    omitted.

    Returns
    -------
    dwells : dict
        ``positions`` (N, 3), ``directions`` (N, 3), ``channel_ids`` (N,),
        ``local_index`` (N,) — vertex index within that channel's mesh.
        If ``labels`` is given, also ``structure_names``, ``role``,
        ``in_target``, ``in_oar``.
    """
    positions = []
    directions = []
    channel_ids = []
    local_index = []

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

    if not positions:
        out = {
            "positions": np.zeros((0, 3), dtype=np.float64),
            "directions": np.zeros((0, 3), dtype=np.float64),
            "channel_ids": np.zeros(0, dtype=int),
            "local_index": np.zeros(0, dtype=int),
        }
    else:
        out = {
            "positions": np.vstack(positions),
            "directions": np.vstack(directions),
            "channel_ids": np.concatenate(channel_ids),
            "local_index": np.concatenate(local_index),
        }

    if labels:
        for key in ("structure_names", "role", "in_target", "in_oar"):
            if key in labels:
                out[key] = labels[key]
    return out
