import numpy as np


def mesh_needle_in_ptv(channel_dwells, ptv_mask, origin, spacing,
                        margin=5.0, mesh_step=1.0):
    """
    For one needle (channel), find the segment inside the PTV, extend it by
    `margin` mm on each end, and return mesh points spaced `mesh_step` mm apart.

    The catheter is modelled as a straight line through the ordered dwell
    positions.  A dense probe (0.25 mm steps) determines PTV entry/exit,
    then the extended segment is re-sampled at `mesh_step`.

    Parameters
    ----------
    channel_dwells : ndarray (M, 3)
        Ordered dwell positions (world coords) for this channel.
    ptv_mask : ndarray (nz, ny, nx), bool
        Combined PTV/CTV mask on the image grid.
    origin : ndarray (3,)   — (x0, y0, z0)
    spacing : ndarray (3,)  — (dx, dy, dz)
    margin : float
        Extension beyond PTV boundary (mm) on each end.
    mesh_step : float
        Spacing between mesh points (mm).

    Returns
    -------
    mesh_points : ndarray (K, 3)
        World-coordinate mesh points along the needle.
    t_entry : float
        Arc-length parameter where PTV starts (before margin).
    t_exit : float
        Arc-length parameter where PTV ends (before margin).
    """
    if len(channel_dwells) < 2:
        return np.empty((0, 3)), None, None

    # Catheter direction from first to last dwell
    p0 = channel_dwells[0]
    p1 = channel_dwells[-1]
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length == 0:
        return np.empty((0, 3)), None, None
    direction = vec / length

    # Project all dwells onto the line: point = p0 + t * direction
    t_dwells = np.dot(channel_dwells - p0, direction)
    t_min_dwell = t_dwells.min()
    t_max_dwell = t_dwells.max()

    # Search range: extend beyond dwell endpoints
    search_pad = margin + 15.0
    t_lo = t_min_dwell - search_pad
    t_hi = t_max_dwell + search_pad

    # Dense probe along the line (0.25 mm resolution)
    probe_step = 0.25
    t_probe = np.arange(t_lo, t_hi + probe_step, probe_step)
    probe_pts = p0[np.newaxis, :] + t_probe[:, np.newaxis] * direction[np.newaxis, :]

    # Convert to voxel indices (vectorised)
    vox = np.round((probe_pts - origin) / spacing).astype(int)
    ix, iy, iz = vox[:, 0], vox[:, 1], vox[:, 2]
    nz, ny, nx = ptv_mask.shape

    in_bounds = (iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    inside = np.zeros(len(t_probe), dtype=bool)
    valid = np.where(in_bounds)[0]
    inside[valid] = ptv_mask[iz[valid], iy[valid], ix[valid]]

    if not np.any(inside):
        return np.empty((0, 3)), None, None

    # PTV entry / exit along the line
    ptv_t = t_probe[inside]
    t_entry = float(ptv_t.min())
    t_exit = float(ptv_t.max())

    # Extend by margin on each side
    t_mesh_start = t_entry - margin
    t_mesh_end = t_exit + margin

    # Generate mesh points at mesh_step intervals
    t_mesh = np.arange(t_mesh_start, t_mesh_end + mesh_step / 2.0, mesh_step)
    mesh_points = p0[np.newaxis, :] + t_mesh[:, np.newaxis] * direction[np.newaxis, :]

    return mesh_points, t_entry, t_exit
