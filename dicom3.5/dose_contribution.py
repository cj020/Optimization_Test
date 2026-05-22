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
    Returns z, y, x arrays in mm (world coordinates)."""
    
    nz, ny, nx = volume.shape
    
    # Create a grid of voxel coordinates
    z = (np.arange(nz) + 0.5) * spacing[2] + origin[2]  # z coordinates of voxel centers (si +0.5) in mm
    y = (np.arange(ny) + 0.5) * spacing[1] + origin[1]
    x = (np.arange(nx) + 0.5) * spacing[0] + origin[0]
    
    return z,y,x 

def beta_0(r, L):
    """
    Calculate the beta angle for r = 1 cm and theta = 90, which can be used as a constant value in the G_L function for theta = 0.
    """
    # For r = 1 cm and theta = 90, the voxel position is directly in front of the source along the direction vector, so the beta angle can be calculated using the geometry of the source and the voxel position. Assuming the source is centered at the origin and extends from -L/2 to L/2 along the z-axis, the end point of the source closest to the voxel is at (0, 0, L/2) if cos_dir_from_dwell_to_voxel >= 0, or (0, 0, -L/2) if cos_dir_from_dwell_to_voxel < 0. The vector from this end point to the voxel position (which is at (0, 0, r)) is then (0, 0, r - L/2) or (0, 0, r + L/2), respectively. The vector from the middle of the source to the voxel position is (0, 0, r). The beta angle can then be calculated as the angle between these two vectors.

    end_point_L = (L/2, 0, 0) # end point of the source closest to the voxel for theta = 0 in (x, y, z) format
    voxel_pos = (r, 0, 0) # voxel position for r = 1 cm, in (z, y, x) format
    vec_x_end_point_L_to_voxel = voxel_pos[2] - end_point_L[0] # vector from the end point of the source to the voxel position
    vec_y_end_point_L_to_voxel = 0
    vec_z_end_point_L_to_voxel = voxel_pos[0] - end_point_L[2]
    vec_end_point_L_to_voxel_norm = np.sqrt(vec_x_end_point_L_to_voxel**2 + vec_y_end_point_L_to_voxel**2 + vec_z_end_point_L_to_voxel**2)
    vec_x_end_point_L_to_voxel_normalized = vec_x_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)
    vec_y_end_point_L_to_voxel_normalized = vec_y_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)
    vec_z_end_point_L_to_voxel_normalized = vec_z_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)

    vec_x_middle_point_L_to_voxel = 0 # vector from the end point of the source to the voxel position
    vec_y_middle_point_L_to_voxel = 0
    vec_z_middle_point_L_to_voxel = voxel_pos[0]
    vec_middle_point_L_to_voxel_norm = np.sqrt(vec_x_middle_point_L_to_voxel**2 + vec_y_middle_point_L_to_voxel**2 + vec_z_middle_point_L_to_voxel**2)
    vec_x_middle_point_L_to_voxel_normalized = vec_x_middle_point_L_to_voxel / (vec_middle_point_L_to_voxel_norm + 1e-6)
    vec_y_middle_point_L_to_voxel_normalized = vec_y_middle_point_L_to_voxel / (vec_middle_point_L_to_voxel_norm + 1e-6)
    vec_z_middle_point_L_to_voxel_normalized = vec_z_middle_point_L_to_voxel / (vec_middle_point_L_to_voxel_norm + 1e-6)

    cos_angle = (vec_x_end_point_L_to_voxel_normalized * vec_x_middle_point_L_to_voxel_normalized + vec_y_end_point_L_to_voxel_normalized * vec_y_middle_point_L_to_voxel_normalized + vec_z_end_point_L_to_voxel_normalized * vec_z_middle_point_L_to_voxel_normalized) / 1.0
    beta_angle = np.arccos(cos_angle)

    return beta_angle

def beta(dwell_pos, voxel_pos, L, direction_L, cos_dir_from_dwell_to_voxel): 
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

    cos_sign = np.where(cos_dir_from_dwell_to_voxel >= 0, 1, -1) # determine the sign of the direction vector based on the cosine value

    end_point_L = (dwell_pos[0] + cos_sign * direction_L[0] * (L/2), dwell_pos[1] + cos_sign * direction_L[1] * (L/2), dwell_pos[2] + cos_sign * direction_L[2] * (L/2))
    
    vec_x_end_point_L_to_voxel = voxel_pos[2] - end_point_L[0]
    vec_y_end_point_L_to_voxel = voxel_pos[1] - end_point_L[1]
    vec_z_end_point_L_to_voxel = voxel_pos[0] - end_point_L[2]

    # norm of the vector from the end point of the source to the voxel center
    vec_end_point_L_to_voxel_norm = np.sqrt(vec_x_end_point_L_to_voxel**2 + vec_y_end_point_L_to_voxel**2 + vec_z_end_point_L_to_voxel**2)
    
    # normalize the vector from the end point of the source to the voxel center
    vec_x_end_point_L_to_voxel_normalized = vec_x_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)
    vec_y_end_point_L_to_voxel_normalized = vec_y_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)
    vec_z_end_point_L_to_voxel_normalized = vec_z_end_point_L_to_voxel / (vec_end_point_L_to_voxel_norm + 1e-6)

    # calculate vector from the middle of the source to the voxel center
    vec_x_source_center_to_voxel = voxel_pos[2] - dwell_pos[0]
    vec_y_source_center_to_voxel = voxel_pos[1] - dwell_pos[1]
    vec_z_source_center_to_voxel = voxel_pos[0] - dwell_pos[2]
    vec_source_center_to_voxel_norm = np.sqrt(vec_x_source_center_to_voxel**2 + vec_y_source_center_to_voxel**2 + vec_z_source_center_to_voxel**2)
    vec_x_source_center_to_voxel_normalized = vec_x_source_center_to_voxel / (vec_source_center_to_voxel_norm + 1e-6)
    vec_y_source_center_to_voxel_normalized = vec_y_source_center_to_voxel / (vec_source_center_to_voxel_norm + 1e-6)
    vec_z_source_center_to_voxel_normalized = vec_z_source_center_to_voxel / (vec_source_center_to_voxel_norm + 1e-6)  

    # calculate the cosine of the angle between the vector from the end point of the source to the voxel center and the vector from the middle of the source to the voxel center
    cos_angle = (vec_x_end_point_L_to_voxel_normalized * vec_x_source_center_to_voxel_normalized + vec_y_end_point_L_to_voxel_normalized * vec_y_source_center_to_voxel_normalized + vec_z_end_point_L_to_voxel_normalized * vec_z_source_center_to_voxel_normalized) / 1.0 
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

def compute_dose_single_dwell_vectorized(dwell_pos, norm_dir, volume_shape, voxel_z, voxel_y, voxel_x, L, S_k, Lambda, GL_0, r_cutoff_mm=50.0):
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
    beta_angle = beta(dwell_pos, voxel_pos, L, norm_dir, cos_theta)  # shape (nz, ny, nx)

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
    bar = '█' * filled + '░' * (bar_length - filled)
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
    """Calculate the dose contribution from all dwell position to the volume grid.
    
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
    voxel_z, voxel_y, voxel_x = voxel_coordinates(volume, spacing, origin) # get the coordinates of the center of each voxel in the volume
    
    beta0 = beta_0(r=10, L=L) # pre-calculate the beta angle at theta = 0, r = 1 cm.
    GLref  = G_L(r=10, L=L, beta=beta0, theta=np.pi/2) # geometry function at theta = 90

    total_dose = np.zeros((nz, ny, nx), dtype=np.float64)
    n_dwells = len(dwell_pos)
    active_dwells = np.sum(dwell_times > 0)

    print(f"  Total dwells: {n_dwells}, Active (time > 0): {active_dwells}")
    t_start = time.time() # start time for dose calculation
    computed = 0 # counter for the number of dwell positions that have been computed for dose contribution

    for i in range(n_dwells):
        if dwell_times[i] <= 0:
            continue

        dose_rate = compute_dose_single_dwell_vectorized(
            dwell_pos=dwell_pos[i],
            norm_dir=norm_dwell_dir[i],
            volume_shape=(nz, ny, nx),
            voxel_z=voxel_z,
            voxel_y=voxel_y,
            voxel_x=voxel_x,
            L=L,
            S_k=S_k,
            Lambda=Lambda,
            GL_0=GLref
        )

        # dose = dose_rate (cGy/h) * time (s) / 3600 (s/h) = cGy
        total_dose += dose_rate.astype(np.float64) * (dwell_times[i] / 3600.0)
        computed += 1
        print_progress_bar(computed, active_dwells, prefix='Dose calc', elapsed=time.time() - t_start)

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s ({elapsed/active_dwells:.2f}s per active dwell)")

    return total_dose.astype(np.float32)
