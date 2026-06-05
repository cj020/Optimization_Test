import sys
import time
from math import dist
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

# TG43 radial dose function table for Ir-192, distances in cm, used for interpolation
r_vals_g = np.array([0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00])
g_vals_g = np.array([0.991, 0.991, 0.997, 0.998, 1.000, 1.002, 1.004, 1.005, 1.003, 0.999, 0.991, 0.968, 0.935])

# interpolation function
g_interp = interp1d(
    r_vals_g,
    g_vals_g,
    kind='linear',
    bounds_error=False,
    fill_value="extrapolate"
)

# TG43 anisotropy function table for Ir-192, distances in cm, theta in degrees, used for interpolation
r_vals_F = np.array([0, 0.25, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00])
theta_vals_F = np.array([0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
                         95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 
                        160, 165, 168, 170, 173, 175, 177, 178, 179, 180])     

# interpolation function
# anisotropy table
# shape = (Nr, Ntheta)
F_table = np.array([
    [0.672,0.671,0.669,0.663,0.671,0.694,0.735,0.762,0.803,0.852,0.892,0.917,0.936,0.955,0.964,0.973,0.986,0.990,
     0.993,0.996,0.997,0.999,1.000,1.000,1.000,0.999,0.998,0.996,0.994,0.990,0.985,0.972,0.963,0.952,0.937,0.918,
     0.891,0.839,0.783,0.748,0.711,0.659,0.614,0.542,0.474,0.440,0.442],
    [0.672,0.671,0.669,0.663,0.671,0.694,0.735,0.762,0.803,0.852,0.892,0.917,0.936,0.955,0.964,0.973,0.986,0.990,
     0.993,0.996,0.997,0.999,1.000,1.000,1.000,0.999,0.998,0.996,0.994,0.990,0.985,0.972,0.963,0.952,0.937,0.918,
     0.891,0.839,0.783,0.748,0.711,0.659,0.614,0.542,0.474,0.440,0.442],
    [0.654,0.652,0.651,0.652,0.665,0.690,0.731,0.760,0.799,0.850,0.887,0.913,0.933,0.951,0.962,0.972,0.979,0.984,
     0.989,0.993,0.995,1.000,1.000,1.000,0.999,0.998,0.996,0.993,0.991,0.986,0.979,0.971,0.962,0.950,0.935,0.915,
     0.888,0.841,0.787,0.751,0.715,0.663,0.620,0.550,0.487,0.453,0.452],
    [0.617,0.615,0.615,0.629,0.653,0.682,0.725,0.756,0.791,0.845,0.878,0.904,0.928,0.944,0.957,0.969,0.975,0.982,
     0.988,0.993,0.997,0.995,0.998,1.000,0.995,0.995,0.993,0.992,0.986,0.985,0.977,0.968,0.959,0.945,0.932,0.908,
     0.881,0.845,0.793,0.758,0.724,0.673,0.631,0.566,0.512,0.480,0.473],
    [0.626,0.629,0.638,0.650,0.676,0.703,0.744,0.770,0.804,0.851,0.886,0.911,0.932,0.948,0.960,0.971,0.979,0.985,
     0.990,0.994,0.996,1.000,0.999,1.000,0.999,0.998,0.996,0.993,0.989,0.984,0.979,0.971,0.961,0.949,0.933,0.914,
     0.887,0.853,0.806,0.770,0.741,0.693,0.652,0.599,0.564,0.534,0.514],
    [0.647,0.652,0.664,0.677,0.698,0.725,0.763,0.785,0.817,0.861,0.893,0.917,0.936,0.951,0.964,0.973,0.981,0.987,
     0.993,0.996,0.998,1.000,0.999,1.000,1.001,1.000,0.997,0.993,0.991,0.985,0.982,0.974,0.965,0.952,0.938,0.919,
     0.895,0.861,0.819,0.786,0.760,0.715,0.678,0.631,0.599,0.571,0.555],
    [0.672,0.678,0.688,0.699,0.719,0.743,0.780,0.799,0.829,0.870,0.899,0.921,0.941,0.953,0.967,0.975,0.983,0.990,
     0.994,0.997,0.999,1.000,1.001,1.000,1.003,1.004,0.999,0.994,0.995,0.988,0.984,0.976,0.967,0.955,0.942,0.922,
     0.900,0.871,0.831,0.802,0.776,0.733,0.701,0.660,0.632,0.606,0.591],
    [0.695,0.699,0.711,0.719,0.737,0.760,0.794,0.812,0.839,0.878,0.904,0.922,0.943,0.955,0.968,0.978,0.983,0.990,
     0.994,0.998,0.999,1.001,1.001,1.000,1.003,1.002,1.001,0.996,0.995,0.990,0.983,0.977,0.967,0.959,0.943,0.925,
     0.905,0.878,0.840,0.812,0.791,0.750,0.720,0.684,0.659,0.635,0.625],
    [0.738,0.744,0.751,0.759,0.775,0.792,0.821,0.835,0.857,0.889,0.912,0.932,0.949,0.958,0.967,0.978,0.983,0.989,
     0.995,0.995,0.999,1.002,1.001,1.000,1.000,0.999,1.000,0.995,0.992,0.987,0.983,0.978,0.966,0.961,0.946,0.932,
     0.913,0.890,0.857,0.834,0.818,0.785,0.760,0.729,0.712,0.693,0.680],
    [0.774,0.777,0.783,0.789,0.802,0.816,0.841,0.854,0.873,0.898,0.920,0.936,0.953,0.961,0.970,0.980,0.986,0.989,
     0.993,0.996,0.999,1.001,1.001,1.000,1.002,0.999,1.002,0.994,0.994,0.988,0.985,0.979,0.970,0.963,0.951,0.937,
     0.919,0.898,0.874,0.855,0.838,0.809,0.791,0.766,0.751,0.734,0.722]
])

# F_table.shape == (len(r_vals_F), len(theta_vals_F))

F_interp = RegularGridInterpolator(
    (r_vals_F, theta_vals_F),
    F_table,
    bounds_error=False,
    fill_value=None
)

def voxel_coordinates(volume, spacing, origin):
    """Calculate the coordinates of the center of each voxel in the volume.
    Returns z, y, x arrays in mm (world coordinates).
    DICOM ImagePositionPatient already gives the center of voxel (0,0,0)."""
    
    nz, ny, nx = volume.shape
    
    # Create a grid of voxel coordinates
    z = np.arange(nz) * spacing[2] + origin[2]
    y = np.arange(ny) * spacing[1] + origin[1]
    x = np.arange(nx) * spacing[0] + origin[0]
    
    return z, y, x 

def rtdose_voxel_centers(dose_ds):
    """Return z, y, x center coordinates in mm (patient coords).
    DICOM ImagePositionPatient already gives the center of voxel (0,0,0)."""
    origin = np.array(dose_ds.ImagePositionPatient, float)
    dy, dx = [float(x) for x in dose_ds.PixelSpacing]
    z_off = np.array(dose_ds.GridFrameOffsetVector, float)
    ny, nx = dose_ds.Rows, dose_ds.Columns

    # z of each frame: IPP_z + offset[k]; pick corner vs center per your vendor
    z = origin[2] + z_off  # or + 0.5 * dz if offsets are to slice centers
    y = origin[1] + np.arange(ny) * dy
    x = origin[0] + np.arange(nx) * dx
    
    return z, y, x, (dx, dy, np.median(np.diff(z_off)) if len(z_off) > 1 else 1.0)

def beta_0(r, L):
    """
    Calculate the beta angle for r = 1 cm and theta = 90 deg , which can be used as a constant value in the G_L function.
    """
    # For r = 1 cm and theta = 90 deg, the voxel position is directly in front of the source along the direction vector, so the beta angle can be calculated using the geometry of the source and the voxel position. Assuming the source is centered at the origin and extends from -L/2 to L/2 along the z-axis, the end point of the source closest to the voxel is at (0, 0, L/2) if cos_dir_from_dwell_to_voxel >= 0, or (0, 0, -L/2) if cos_dir_from_dwell_to_voxel < 0. The vector from this end point to the voxel position (which is at (0, 0, r)) is then (0, 0, r - L/2) or (0, 0, r + L/2), respectively. The vector from the middle of the source to the voxel position is (0, 0, r). The beta angle can then be calculated as the angle between these two vectors.

    end_point_1_L = (L/2, 0, 0) # end point of the source closest to the voxel for theta = 0 in (x, y, z) format
    end_point_2_L = (-L/2, 0, 0) # end point of the source farthest from the voxel for theta = 0 in (x, y, z) format
    voxel_pos = (r, 0, 0) # voxel position for r = 1 cm, in (z, y, x) format
    
    vec_x_end_point_1_L_to_voxel = voxel_pos[2] - end_point_1_L[0] # vector from the end point of the source to the voxel position
    vec_y_end_point_1_L_to_voxel = 0
    vec_z_end_point_1_L_to_voxel = voxel_pos[0] - end_point_1_L[2]
    vec_end_point_1_L_to_voxel_norm = np.sqrt(vec_x_end_point_1_L_to_voxel**2 + vec_y_end_point_1_L_to_voxel**2 + vec_z_end_point_1_L_to_voxel**2)
    vec_x_end_point_1_L_to_voxel_normalized = vec_x_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)
    vec_y_end_point_1_L_to_voxel_normalized = vec_y_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)
    vec_z_end_point_1_L_to_voxel_normalized = vec_z_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)

    vec_x_end_point_2_L_to_voxel = voxel_pos[2] - end_point_2_L[0] # vector from the end point of the source to the voxel position
    vec_y_end_point_2_L_to_voxel = 0
    vec_z_end_point_2_L_to_voxel = voxel_pos[0] - end_point_2_L[2]
    vec_end_point_2_L_to_voxel_norm = np.sqrt(vec_x_end_point_2_L_to_voxel**2 + vec_y_end_point_2_L_to_voxel**2 + vec_z_end_point_2_L_to_voxel**2)
    vec_x_end_point_2_L_to_voxel_normalized = vec_x_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)
    vec_y_end_point_2_L_to_voxel_normalized = vec_y_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)
    vec_z_end_point_2_L_to_voxel_normalized = vec_z_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)

    cos_angle = (vec_x_end_point_1_L_to_voxel_normalized * vec_x_end_point_2_L_to_voxel_normalized + vec_y_end_point_1_L_to_voxel_normalized * vec_y_end_point_2_L_to_voxel_normalized + vec_z_end_point_1_L_to_voxel_normalized * vec_z_end_point_2_L_to_voxel_normalized) / 1.0
    beta_angle = np.arccos(cos_angle)

    return beta_angle

def beta_angle_function(dwell_pos, voxel_pos, L, direction_L): 
    """
    Calculate the beta angle for a given position and voxel.

    Parameters:
    pos (tuple): The position of the dwell point in world coordinates (x, y, z).
    voxel_pos (tuple): The voxel coordinates (z, y, x).
    L (float): The length of the source.
    direction_L (tuple): The direction vector of the catheter. in (dx, dy, dz) format and should be normalized to unit vector.
    cos_dir_from_dwell_to_voxel (float): The cosine of the angle between the direction vector and the vector from the dwell position to the voxel.

    Returns:
    float: The beta angle for the given position and voxel.
    """

    end_point_1_L = (dwell_pos[0] + direction_L[0] * (L/2), dwell_pos[1] + direction_L[1] * (L/2), dwell_pos[2] + direction_L[2] * (L/2))
    end_point_2_L = (dwell_pos[0] - direction_L[0] * (L/2), dwell_pos[1] - direction_L[1] * (L/2), dwell_pos[2] - direction_L[2] * (L/2))
    
    # end point 1
    vec_x_end_point_1_L_to_voxel = voxel_pos[2] - end_point_1_L[0]
    vec_y_end_point_1_L_to_voxel = voxel_pos[1] - end_point_1_L[1]
    vec_z_end_point_1_L_to_voxel = voxel_pos[0] - end_point_1_L[2]

    # norm of the vector from the end point of the source to the voxel center
    vec_end_point_1_L_to_voxel_norm = np.sqrt(vec_x_end_point_1_L_to_voxel**2 + vec_y_end_point_1_L_to_voxel**2 + vec_z_end_point_1_L_to_voxel**2)
    
    # normalize the vector from the end point of the source to the voxel center
    vec_x_end_point_1_L_to_voxel_normalized = vec_x_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)
    vec_y_end_point_1_L_to_voxel_normalized = vec_y_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)
    vec_z_end_point_1_L_to_voxel_normalized = vec_z_end_point_1_L_to_voxel / (vec_end_point_1_L_to_voxel_norm + 1e-6)

     # end point 2
    vec_x_end_point_2_L_to_voxel = voxel_pos[2] - end_point_2_L[0]
    vec_y_end_point_2_L_to_voxel = voxel_pos[1] - end_point_2_L[1]
    vec_z_end_point_2_L_to_voxel = voxel_pos[0] - end_point_2_L[2]

    # norm of the vector from the end point of the source to the voxel center
    vec_end_point_2_L_to_voxel_norm = np.sqrt(vec_x_end_point_2_L_to_voxel**2 + vec_y_end_point_2_L_to_voxel**2 + vec_z_end_point_2_L_to_voxel**2)
    
    # normalize the vector from the end point of the source to the voxel center
    vec_x_end_point_2_L_to_voxel_normalized = vec_x_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)
    vec_y_end_point_2_L_to_voxel_normalized = vec_y_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)
    vec_z_end_point_2_L_to_voxel_normalized = vec_z_end_point_2_L_to_voxel / (vec_end_point_2_L_to_voxel_norm + 1e-6)

    # calculate the cosine of the angle between the vector from the end point of the source to the voxel center and the vector from the middle of the source to the voxel center
    cos_angle = (vec_x_end_point_1_L_to_voxel_normalized * vec_x_end_point_2_L_to_voxel_normalized + vec_y_end_point_1_L_to_voxel_normalized * vec_y_end_point_2_L_to_voxel_normalized + vec_z_end_point_1_L_to_voxel_normalized * vec_z_end_point_2_L_to_voxel_normalized) / 1.0 
    beta_angle = np.arccos(cos_angle)

    return beta_angle

def G_L(r, L, beta, theta):
    """Calculate the geometry function G_L for a given distance, source length, beta angle, and theta angle."""
    # r: distance from the source to the point of interest
    # L: active length of the source
    # beta: angle between the vector from the end point of the source to the voxel center and the vector from the middle of the source to the voxel center
    # theta: angle between the direction vector of the source and the vector from the middle of the source to the voxel center

    theta_zero = np.where(theta == 0, 1, 0) # determine the sign of theta for the geometry function calculation  
    G = theta_zero * (1 / (r**2 - L**2 / 4)) + (1 - theta_zero) * (beta / (L * r * np.sin(theta)))

    return G

def compute_dose_single_dwell_cropped(dwell_pos, norm_dir, voxel_z, voxel_y, voxel_x,
                                      L, S_k, Lambda, GL_0, r_cutoff_mm=50.0):
    """
    Compute TG-43 dose rate from a single dwell, operating only on a bounding-box
    sub-volume within r_cutoff_mm of the dwell position.

    Returns
    -------
    dose_sub : array (sub_nz, sub_ny, sub_nx) float32 — dose rate in cGy/h
    iz0, iy0, ix0 : int — starting indices into the full volume
    """

    r_min_mm = 2.0  # ~source capsule outer radius for Ir-192 HDR
    
    # Compute bounding box in world coordinates. 
    # Bounding-box: find index range within r_cutoff_mm along each axis
    ix0 = int(np.searchsorted(voxel_x, dwell_pos[0] - r_cutoff_mm))
    ix1 = int(np.searchsorted(voxel_x, dwell_pos[0] + r_cutoff_mm, side='right'))
    iy0 = int(np.searchsorted(voxel_y, dwell_pos[1] - r_cutoff_mm))
    iy1 = int(np.searchsorted(voxel_y, dwell_pos[1] + r_cutoff_mm, side='right'))
    iz0 = int(np.searchsorted(voxel_z, dwell_pos[2] - r_cutoff_mm))
    iz1 = int(np.searchsorted(voxel_z, dwell_pos[2] + r_cutoff_mm, side='right'))

    ix0 = max(ix0, 0); ix1 = min(ix1, len(voxel_x))
    iy0 = max(iy0, 0); iy1 = min(iy1, len(voxel_y))
    iz0 = max(iz0, 0); iz1 = min(iz1, len(voxel_z))

    if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
        return None, iz0, iy0, ix0

    # Extract sub-volume coordinates for the bounding box
    sub_x = voxel_x[ix0:ix1]
    sub_y = voxel_y[iy0:iy1]
    sub_z = voxel_z[iz0:iz1]

    zz, yy, xx = np.meshgrid(sub_z, sub_y, sub_x, indexing='ij')

    dx = xx - dwell_pos[0]
    dy = yy - dwell_pos[1]
    dz = zz - dwell_pos[2]

    # Compute distance r (mm) — clamp to source outer radius to avoid singularity
    r = np.sqrt(dx**2 + dy**2 + dz**2, dtype=np.float32)
    np.maximum(r, r_min_mm, out=r)

    mask = r < r_cutoff_mm
    if not np.any(mask):
        return None, iz0, iy0, ix0
    
    # Polar angle theta: angle between catheter direction and dwell-to-voxel vector
    # theta = 0 along catheter axis, theta = pi/2 perpendicular

    inv_r = np.float32(1.0) / r
    cos_theta = (dx * inv_r * norm_dir[0] + dy * inv_r * norm_dir[1] + dz * inv_r * norm_dir[2])
    np.clip(cos_theta, -1.0, 1.0, out=cos_theta)
    theta = np.arccos(cos_theta) # radians

    # Beta angle: angle subtended by source at the point of interest
    # Source endpoints: dwell_pos ± (L/2) * norm_dir
    half_L = np.float32(L / 2.0)
    end1 = dwell_pos + half_L * norm_dir
    end2 = dwell_pos - half_L * norm_dir

    # Vectors from each source endpoint to voxel
    v1x = xx - end1[0]
    v1y = yy - end1[1]
    v1z = zz - end1[2]
    r1 = np.sqrt(v1x**2 + v1y**2 + v1z**2, dtype=np.float32)
    r1 = np.maximum(r1, 0.1, out=r1)

    v2x = xx - end2[0]
    v2y = yy - end2[1]
    v2z = zz - end2[2]
    r2 = np.sqrt(v2x**2 + v2y**2 + v2z**2, dtype=np.float32)
    np.maximum(r2, 0.1, out=r2)

    # Beta = angle between vectors (voxel - end1) and (voxel - end2)
    cos_beta = (v1x * v2x + v1y * v2y + v1z * v2z) / (r1 * r2)
    np.clip(cos_beta, -1.0, 1.0, out=cos_beta)
    beta_angle = np.arccos(cos_beta)
    
    del v1x, v1y, v1z, v2x, v2y, v2z, r1, r2, cos_beta, xx, yy, zz # free memory

    # Geometry function G_L(r, theta)
    sin_theta = np.sin(theta)
    small_angle = sin_theta < 1e-6

    G = np.where(
        small_angle,
        1.0 / (r**2 - half_L**2), # theta ≈ 0 or pi (on-axis)
        beta_angle / (L * r * sin_theta)  # general case
    )

    # Radial dose function g(r)
    r_cm = r * np.float32(0.1)
    g_r = g_interp(r_cm)

    # Anisotropy function F(r, theta)
    theta_deg = np.degrees(theta) # convert angle from radians to degrees

    # Prepare points for interpolation (flattened)
    points = np.column_stack([r_cm[mask].ravel(), theta_deg[mask].ravel()]) # shape (N_masked_voxels, 2)
    F_values_flat = F_interp(points)

    # Build full F array
    F_r_theta = np.ones_like(r, dtype=np.float32) # default to 1.0 for out-of-bounds (e.g. r > 10 cm or theta > 180°)
    F_r_theta[mask] = F_values_flat

    # TG-43 dose rate: D_dot = S_k * Lambda * (G/G_ref) * g(r) * F(r,theta )
    dose_sub = np.zeros(r.shape, dtype=np.float32)
    dose_sub[mask] = S_k * Lambda * (G[mask] / GL_0) * g_r[mask] * F_r_theta[mask]
    np.maximum(dose_sub, 0.0, out=dose_sub)

    # Clamp any negative values from extrapolation
    dose_sub = np.maximum(dose_sub, 0.0)

    return dose_sub, iz0, iy0, ix0

def compute_dose_single_dwell_vectorized(dwell_pos, norm_dir, volume_shape, voxel_z, voxel_y, voxel_x, L, S_k, Lambda, GL_0, r_cutoff_mm=50.0):
    """
    Compute the TG-43 dose rate from a single dwell position to all voxels (vectorized).
    Includes computation of distance, theta, beta, G, g, F in this function.

    Parameters
    ----------
    dwell_pos : array (3,) — dwell position in world coords (x, y, z) in mm
    norm_dir : array (3,) — normalized catheter direction vector (dx, dy, dz)
    volume_shape : tuple (nz, ny, nx)
    voxel_z, voxel_y, voxel_x : 1D arrays of voxel center coordinates in mm
    L : float — active source length in mm
    S_k : float — air kerma strength in cGy·cm²/h
    Lambda : float — dose-rate constant in cGy/(h·U)
    r_cutoff_mm : float — ignore voxels further than this (mm)

    Returns
    -------
    dose_rate : array (nz, ny, nx) in cGy/h
    """

    nz, ny, nx = volume_shape

    # Build 3D coordinate grids
    zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij')

    # Vector from dwell to each voxel
    dx = xx - dwell_pos[0]
    dy = yy - dwell_pos[1]
    dz = zz - dwell_pos[2]

    # Distance r (mm) — clamp to source outer radius to avoid singularity
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_min_mm = 2.0  # ~source capsule outer radius for Ir-192 HDR
    r = np.maximum(r, r_min_mm)

    # Mask: only compute for voxels within cutoff
    mask = r < r_cutoff_mm

    # Polar angle theta: angle between catheter direction and dwell-to-voxel vector
    # theta = 0 along catheter axis, theta = pi/2 perpendicular
    cos_theta = (dx / r * norm_dir[0] + dy / r * norm_dir[1] + dz / r * norm_dir[2]) 
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # radians

    # Beta angle: angle subtended by source at the point of interest
    # Source endpoints: dwell_pos ± (L/2) * norm_dir
    half_L = L / 2.0
    end1 = dwell_pos + half_L * norm_dir  # (x, y, z)
    end2 = dwell_pos - half_L * norm_dir
    
    # Vectors from each source endpoint to voxel
    v1x = xx - end1[0]
    v1y = yy - end1[1]
    v1z = zz - end1[2]
    r1 = np.sqrt(v1x**2 + v1y**2 + v1z**2)
    r1 = np.maximum(r1, 0.1)

    v2x = xx - end2[0]
    v2y = yy - end2[1]
    v2z = zz - end2[2]
    r2 = np.sqrt(v2x**2 + v2y**2 + v2z**2)
    r2 = np.maximum(r2, 0.1)

    # Beta = angle between vectors (voxel - end1) and (voxel - end2)
    cos_beta = (v1x * v2x + v1y * v2y + v1z * v2z) / (r1 * r2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta_angle = np.arccos(cos_beta)

    # Reference geometry: G(r0=1cm, theta0=90°)
    G_ref = GL_0
    
    # Geometry function G_L(r, theta)
    sin_theta = np.sin(theta)
    small_angle = sin_theta < 1e-6

    G = np.where(
        small_angle,
        1.0 / (r**2 - (L / 2.0)**2),  # theta ≈ 0 or pi (on-axis)
        beta_angle / (L * r * sin_theta)      # general case
    )
   
    r_cm = r / 10.0 # convert distance from mm to cm

    # Radial dose function g(r)
    g_r = g_interp(r_cm)

    theta_deg = np.degrees(theta) # convert angle from radians to degrees

    # Prepare points for interpolation (flattened)
    points = np.column_stack([r_cm[mask].ravel(), theta_deg[mask].ravel()]) # shape (N_masked_voxels, 2)
    F_values_flat = F_interp(points)

    # Build full F array
    F_r_theta = np.ones_like(r, dtype=np.float64) # default to 1.0 for out-of-bounds
    F_r_theta[mask] = F_values_flat

    # TG-43 dose rate: D_dot = S_k * Lambda * (G/G_ref) * g(r) * F(r,theta)
    dose_rate = np.zeros(volume_shape, dtype=np.float64)
    dose_rate[mask] = S_k * Lambda * (G[mask]/G_ref) * g_r[mask] * F_r_theta[mask]

    # Clamp any negative values from extrapolation
    dose_rate = np.maximum(dose_rate, 0.0)

    return dose_rate.astype(np.float32)

def compute_dose_single_dwell_vectorized_old(dwell_pos, norm_dir, volume_shape, voxel_z, voxel_y, voxel_x, L, S_k, Lambda, GL_0, r_cutoff_mm=50.0):
    """
    Compute the TG-43 dose rate from a single dwell position to all voxels (vectorized).

    Parameters
    ----------
    dwell_pos : array (3,) — dwell position in world coords (x, y, z) in mm
    norm_dir : array (3,) — normalized catheter direction vector (dx, dy, dz)
    volume_shape : tuple (nz, ny, nx)
    voxel_z, voxel_y, voxel_x : 1D arrays of voxel center coordinates in mm
    L : float — active source length in mm
    S_k : float — air kerma strength in cGy·cm²/h
    Lambda : float — dose-rate constant in cGy/(h·U)
    r_cutoff_mm : float — ignore voxels further than this (mm)

    Returns
    -------
    dose_rate : array (nz, ny, nx) in cGy/h
    """

    nz, ny, nx = volume_shape

    # Build 3D coordinate grids
    zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij')

    # Vector from dwell to each voxel
    dx = xx - dwell_pos[0]
    dy = yy - dwell_pos[1]
    dz = zz - dwell_pos[2]

    # Distance r (mm) — clamp to source outer radius to avoid singularity
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r_min_mm = 2.0  # ~source capsule outer radius for Ir-192 HDR
    r = np.maximum(r, r_min_mm)

    # Mask: only compute for voxels within cutoff
    mask = r < r_cutoff_mm

    # Polar angle theta: angle between catheter direction and dwell-to-voxel vector
    # theta = 0 along catheter axis, theta = pi/2 perpendicular
    cos_theta = (dx / r * norm_dir[0] + dy / r * norm_dir[1] + dz / r * norm_dir[2]) 
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # radians

    # Beta angle: 
    voxel_pos = (zz, yy, xx)  # shape (nz, ny, nx)
    beta_angle = beta_angle_function(dwell_pos, voxel_pos, L, norm_dir)  # shape (nz, ny, nx)

    # Reference geometry: G(r0=1cm, theta0=90°)
    G_ref = GL_0
    
    r_cm = r / 10.0 # convert distance from mm to cm
    theta_deg = np.degrees(theta) # convert angle from radians to degrees

    # TG-43 dose rate: D_dot = S_k * Lambda * (G/G_ref) * g(r) * F(r,theta)
    dose_rate = np.zeros(volume_shape, dtype=np.float64)
    dose_rate[mask] = S_k * Lambda * G_L(r, L, beta_angle, theta)[mask]/G_ref * g_interp(r_cm[mask]) * F_interp((r_cm[mask], theta_deg[mask]))

    # Clamp any negative values from extrapolation
    dose_rate = np.maximum(dose_rate, 0.0)

    return dose_rate.astype(np.float32)

def print_progress_bar(current, total, bar_length=40, prefix='', suffix='', elapsed=None):
    """Print a progress bar to stdout."""
    fraction = current / total if total > 0 else 0
    filled = int(bar_length * fraction)
    bar = '#' * filled + '-' * (bar_length - filled)
    pct = 100.0 * fraction

    eta_str = ''
    if elapsed is not None and current > 0:
        eta = elapsed * (total - current) / current
        if eta > 60:
            eta_str = f" ETA: {eta/60:.1f}min"
        else:
            eta_str = f" ETA: {eta:.0f}s"

    sys.stdout.write(f'\r  {prefix} |{bar}| {pct:5.1f}% ({current}/{total}){suffix}{eta_str}   ')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def dose_contribution(dwell_pos, norm_dwell_dir, dwell_times, volume, spacing, origin, L, S_k, Lambda):
    """Calculate the dose contribution from all dwell positions to the volume grid.
    
    Parameters
    ----------
    dwell_pos : array (N, 3) — dwell positions (x, y, z) in mm
    norm_dwell_dir : array (N, 3) — normalized direction vectors
    dwell_times : array (N,) — irradiation time per dwell in seconds
    volume : array (nz, ny, nx) — CT/dose volume (used for shape only)
    spacing : tuple (dx, dy, dz) in mm
    origin : tuple (x0, y0, z0) in mm
    L : float — active source length in mm
    S_k : float — air kerma strength in cGy·cm²/h
    Lambda : float — dose-rate constant in cGy/(h·U)

    Returns
    -------
    total_dose : array (nz, ny, nx) — accumulated dose in cGy."""
    
    nz, ny, nx = volume.shape

    # Pre-compute 1D voxel coordinates once (sorted, for searchsorted in bounding-box)
    voxel_z, voxel_y, voxel_x = voxel_coordinates(volume, spacing, origin) # get the coordinates of the center of each voxel in the volume
    
    beta0 = beta_0(r=10, L=L) # pre-calculate the beta angle at theta = 90 deg, r = 1 cm.
    GLref  = G_L(r=10, L=L, beta=beta0, theta=np.pi/2) # geometry function at theta = 90 deg
    
    total_dose = np.zeros((nz, ny, nx), dtype=np.float64)
    n_dwells = len(dwell_pos)
    active_dwells = int(np.sum(dwell_times > 0))

    print(f"  Total dwells: {n_dwells}, Active (time > 0): {active_dwells}")
    t_start = time.time() # start time for dose calculation
    computed = 0 # counter for the number of dwell positions that have been computed for dose contribution

    for i in range(n_dwells):
        if dwell_times[i] <= 0:
            continue

        dose_sub, iz0, iy0, ix0 = compute_dose_single_dwell_cropped(
            dwell_pos=dwell_pos[i],
            norm_dir=norm_dwell_dir[i],
            voxel_z=voxel_z,
            voxel_y=voxel_y,
            voxel_x=voxel_x,
            L=L,
            S_k=S_k,
            Lambda=Lambda,
            GL_0=GLref
        )

        # dose = (sum of dose_sub) (cGy/h) * time (s) / 3600 (s/h) = cGy
        
        if dose_sub is not None:
            snz, sny, snx = dose_sub.shape
            total_dose[iz0:iz0+snz, iy0:iy0+sny, ix0:ix0+snx] += dose_sub * (dwell_times[i] / 3600.0)
        
        computed += 1
        print_progress_bar(computed, active_dwells, prefix='Dose calc', elapsed=time.time() - t_start)

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s ({elapsed/active_dwells:.2f}s per active dwell)")

    return total_dose.astype(np.float32)

def gamma_index_3d(dose_vol_1, dose_vol_2, spacing, gamma_dist=3.0, gamma_percentage=3.0, cut_off=0.1, ref_dose=None, sat_mask=None):
    """
    3D gamma analysis based on Daniel A. Low 1998 Med Phys paper.
    
    Both volumes are expected in the same absolute units (e.g. Gy).
    Dose-difference criterion is global: percentage of ref_dose.

    Parameters
    ----------
    dose_vol_1 : array (nz, ny, nx) — reference dose volume
    dose_vol_2 : array (nz, ny, nx) — evaluation dose volume (same shape & units)
    spacing : tuple (dx, dy, dz) — voxel sizes in mm
    gamma_dist : float — distance-to-agreement criterion in mm
    gamma_percentage : float — dose difference criterion in %
    cut_off : float — fraction of ref_dose below which voxels are excluded (0-1)
    ref_dose : float or None — global normalization dose for % criterion;
               if None, uses max of dose_vol_1
    sat_mask : array (bool) or None — True for saturated reference voxels to
               exclude from pass-rate evaluation

    Returns
    -------
    gamma_map : array (nz, ny, nx) — gamma values (0 where below threshold)
    pass_rate : float — percentage of evaluated voxels with gamma <= 1
    """

    if dose_vol_1.shape != dose_vol_2.shape: # check if the two dose volumes have the same shape
        print("  ERROR: Matrix sizes do not match.")
        return np.zeros(dose_vol_1.shape, dtype=np.float32), 0.0
    
    nonzero_1 = dose_vol_1[dose_vol_1 > 0]
    nonzero_2 = dose_vol_2[dose_vol_2 > 0]

    if len(nonzero_1) == 0 or len(nonzero_2) == 0:
        print("  WARNING: One of the dose distributions is all zeros.")
        return np.zeros(dose_vol_1.shape, dtype=np.float32), 0.0
       
    dose_vol_1 = dose_vol_1.astype(np.float32)
    dose_vol_2 = dose_vol_2.astype(np.float32)
 
    print(f"  Using ABSOLUTE dose comparison (both volumes in same units):")
    print(f"    Volume 1 (ref):  max={np.max(dose_vol_1):.4f}, mean(>0)={np.mean(nonzero_1):.4f}")
    print(f"    Volume 2 (eval): max={np.max(dose_vol_2):.4f}, mean(>0)={np.mean(nonzero_2):.4f}")

    voxel_sizes = np.array([float(spacing[0]), float(spacing[1]), float(spacing[2])])

    if ref_dose is None or ref_dose == 0: # if ref_dose is not provided or is zero, use the maximum dose in dose_vol_1 as the reference dose for normalization. This ensures that the gamma analysis is based on relative dose differences rather than absolute values, which can be more meaningful for comparing dose distributions.
        ref_dose = np.max(dose_vol_1)

    dim = dose_vol_1.shape  # (nz, ny, nx)

    # Build search offsets
    ranges_indices = np.ceil(gamma_dist / voxel_sizes).astype(int) # number of voxels to search in each direction based on the gamma distance criterion and voxel sizes

    dx_indices = np.arange(-ranges_indices[0], ranges_indices[0] + 1)
    dy_indices = np.arange(-ranges_indices[1], ranges_indices[1] + 1)
    dz_indices = np.arange(-ranges_indices[2], ranges_indices[2] + 1)

    xx_ind, yy_ind, zz_ind = np.meshgrid(dx_indices, dy_indices, dz_indices, indexing='ij')
    xx_ind = xx_ind.ravel()
    yy_ind = yy_ind.ravel()
    zz_ind = zz_ind.ravel()

    # Distance squared (normalized by gamma_dist^2)
    dr = np.sqrt((xx_ind * voxel_sizes[0])**2 + (yy_ind * voxel_sizes[1])**2 + (zz_ind * voxel_sizes[2])**2)
    
    # Sphere filter: discard offsets outside the gamma distance sphere
    sphere_mask = dr <= gamma_dist
    xx_ind = xx_ind[sphere_mask]
    yy_ind = yy_ind[sphere_mask]
    zz_ind = zz_ind[sphere_mask]
    dr = dr[sphere_mask]
    
    dr2 = dr**2 / gamma_dist**2
    dr2b = np.maximum(dr2, 1.0)

    N_offsets = len(xx_ind)

    # Determine evaluation region (bounding box of voxels above threshold)
    mask = (dose_vol_1 > (ref_dose * cut_off)) | (dose_vol_2 > (ref_dose * cut_off))

    # Find bounding box
    z_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))

    if not np.any(z_any):
        print("  WARNING: No voxels above cutoff threshold.")
        return np.zeros(dim, dtype=np.float32), 0.0

    iz = np.arange(np.argmax(z_any), len(z_any) - np.argmax(z_any[::-1]))
    iy = np.arange(np.argmax(y_any), len(y_any) - np.argmax(y_any[::-1]))
    ix = np.arange(np.argmax(x_any), len(x_any) - np.argmax(x_any[::-1]))

    # Dose difference normalization factor
    f = (ref_dose * gamma_percentage / 100.0)**2

    # Extract sub-volumes for the evaluation region
    dose_vol_1_sub = dose_vol_1[np.ix_(iz, iy, ix)]

    print(f"  Gamma criteria: {gamma_percentage}% / {gamma_dist}mm")
    print(f"  Cutoff: {cut_off*100:.0f}% of max")
    print(f"  Evaluation region: {dose_vol_1_sub.shape} (from full {dim})")
    print(f"  Search offsets: {N_offsets}")

    # Pre-compute evaluation mask
    dose_vol_2_sub = dose_vol_2[np.ix_(iz, iy, ix)]
    eval_mask = (dose_vol_1_sub > ref_dose * cut_off) | (dose_vol_2_sub > ref_dose * cut_off)

    # Exclude saturated reference voxels from evaluation (unreliable due to clipping)
    if sat_mask is not None:
        sat_sub = sat_mask[np.ix_(iz, iy, ix)]
        n_sat = int(np.sum(sat_sub & eval_mask))
        if n_sat > 0:
            print(f"  Excluding {n_sat} saturated reference voxels from evaluation")
            eval_mask = eval_mask & ~sat_sub

    print(f"  Evaluated voxels: {int(np.sum(eval_mask))}")

    t_start = time.time()
    gammamap_s = None

    for k in range(N_offsets):
        if k % 50 == 0:
            print_progress_bar(k, N_offsets, prefix='Gamma', elapsed=time.time() - t_start)

        # Shifted indices, clamped to valid range
        ixx2 = np.clip(ix + xx_ind[k], 0, dim[2] - 1)
        iyy2 = np.clip(iy + yy_ind[k], 0, dim[1] - 1)
        izz2 = np.clip(iz + zz_ind[k], 0, dim[0] - 1)

        # Extract shifted dose_vol_2
        tmp2d = dose_vol_2[np.ix_(izz2, iyy2, ixx2)]

        # Dose difference term (squared, normalized)
        tmp3 = (dose_vol_1_sub - tmp2d)**2 / f

        # Combined gamma squared: (dose_diff^2/f + dist^2/gamma_dist^2) / max(dist^2/gamma_dist^2, 1)
        tmp4 = (tmp3 + dr2[k]) / dr2b[k]

        if gammamap_s is None:
            gammamap_s = tmp4.copy()
        else:
            np.minimum(gammamap_s, tmp4, out=gammamap_s)

        # Early termination: if all evaluated voxels already pass, stop
        if gammamap_s is not None and k % 20 == 19:
            if np.all(gammamap_s[eval_mask] <= 1.0):
                print(f"\n  Early termination at offset {k+1}/{N_offsets} (all voxels pass)")
                break

    print_progress_bar(N_offsets, N_offsets, prefix='Gamma', elapsed=time.time() - t_start)
    elapsed = time.time() - t_start
    print(f"  Gamma computation completed in {elapsed:.1f}s")

    # Evaluate pass rate on voxels above cutoff (eval_mask and dose_vol_2_sub computed above)
    eval_values = gammamap_s[eval_mask]

    N_eval = len(eval_values)
    N_pass = np.sum(eval_values <= 1.0)
    pass_rate = (N_pass / N_eval * 100.0) if N_eval > 0 else 0.0

    # Take sqrt to get actual gamma values
    gammamap_s = np.sqrt(gammamap_s)

    # Zero out regions below threshold
    gammamap_s[(dose_vol_1_sub <= ref_dose * cut_off) & (dose_vol_2_sub <= ref_dose * cut_off)] = 0.0

    # Place back into full-size array
    gamma_map = np.zeros(dim, dtype=np.float32)
    gamma_map[np.ix_(iz, iy, ix)] = gammamap_s

    return gamma_map, pass_rate