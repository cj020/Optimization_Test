import numpy as np

def voxel_coordinates(volume, spacing, origin):
    """Calculate the coordinates of the center of each voxel in the volume."""
    
    nz, ny, nx = volume.shape
    
    # Create a grid of voxel coordinates
    z = (np.arange(nz) + 0.5) * spacing[2] + origin[2]  # z coordinates of voxel centers (si +0.5) in mm
    y = (np.arange(ny) + 0.5) * spacing[1] + origin[1]
    x = (np.arange(nx) + 0.5) * spacing[0] + origin[0]
    
    return z,y,x 

def distance_to_all(pos, volume, spacing, origin):
    """Calculate the distance from a given position to all other positions."""
      
    # Create a grid of voxel coordinates
    z, y, x = voxel_coordinates(volume, spacing, origin)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel

    # Calculate the distance from the given position to each voxel center, create another (nz * ny * nx) array where each element is the distance from the given position to that voxel center
    dist = np.sqrt((xx - pos[0])**2 + (yy - pos[1])**2 + (zz - pos[2])**2)
    
    return dist

def direction_to_all(pos, norm_dir_vec, volume, spacing, origin):
    """Calculate the angle as well as the cosine of the angle between the direction vector and the vector from a given position to all other positions."""
      
    # Create a grid of voxel coordinates
    z, y, x = voxel_coordinates(volume, spacing, origin)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel

    # Calculate the vector from the given position to each voxel center, create three (nz * ny * nx) arrays where each element is the vector component from the given position to that voxel center
    vec_x = xx - pos[0]
    vec_y = yy - pos[1]
    vec_z = zz - pos[2]
    vec_norm = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2) # create another (nz * ny * nx) array where each element is the norm of the vector from the given position to that voxel center
    vec_x_normalized = vec_x / (vec_norm + 1e-6) # normalize the vector from the given position to each voxel center, add a small value to avoid division by zero
    vec_y_normalized = vec_y / (vec_norm + 1e-6)
    vec_z_normalized = vec_z / (vec_norm + 1e-6)
    
    # Calculate the cosine of the angle between the direction vector and the vector from the given position to each voxel center, create another (nz * ny * nx) array where each element is the cosine of the angle between the direction vector and the vector from the given position to that voxel center
    cos_dir_to_voxel = (vec_x_normalized * norm_dir_vec[0] + vec_y_normalized * norm_dir_vec[1] + vec_z_normalized * norm_dir_vec[2]) / 1.0
    angle_to_voxel = np.arccos(cos_dir_to_voxel) # create another (nz * ny * nx) array where each element is the angle between the direction vector and the vector from the given position to that voxel center

    return cos_dir_to_voxel, angle_to_voxel

def dose_contribution(dwell_pos, norm_dwell_dir, dwell_count, rtdose, volume, spacing, origin):
    """Calculate the dose contribution from all dwell position to the volume grid."""
    
    nz, ny, nx = volume.shape

    distance = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32) # initialize arrays to store the dose contribution from the dwell position to each voxel in the volume grid
    cos_dir_to_voxel = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32)
    angle_to_voxel = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32)

    for i in range(dwell_count):
        # Calculate the dose contribution from the current dwell position
        dist = distance_to_all(dwell_pos[i], volume, spacing, origin)
        distance[i] = dist
    
        # Calculate the angle between the local direction vector of the current dwell position and the vector from the current dwell position to each voxel center
        cos_dir, angle = direction_to_all(dwell_pos[i], norm_dwell_dir[i], volume, spacing, origin)
        cos_dir_to_voxel[i] = cos_dir
        angle_to_voxel[i] = angle

    return distance, cos_dir_to_voxel, angle_to_voxel

def beta(dwell_pos, voxel_pos, L, direction): 
    """
    Calculate the beta angle for a given position and voxel.

    Parameters:
    pos (tuple): The position of the dwell point in world coordinates (x, y, z).
    voxel (tuple): The voxel coordinates (z, y, x).
    L (float): The length of the source.
    direction (tuple): The direction vector of the catheter.

    Returns:
    float: The beta angle for the given position and voxel.
    """
    
    