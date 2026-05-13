import numpy as np

def voxel_coordinates(volume, spacing, origin):
    """Calculate the coordinates of the center of each voxel in the volume."""
    
    nz, ny, nx = volume.shape
    
    # Create a grid of voxel coordinates
    z = (np.arange(nz) + 0.5) * spacing[2] + origin[2]  # z coordinates of voxel centers (si +0.5) in mm
    y = (np.arange(ny) + 0.5) * spacing[1] + origin[1]
    x = (np.arange(nx) + 0.5) * spacing[0] + origin[0]
    
    return z,y,x 

def distance_to_all(pos, voxel_z, voxel_y, voxel_x):
    """Calculate the distance from a given position to all other positions."""
      
    # Create a grid of voxel coordinates
    
    zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel
    dist = np.zeros(zz.shape, dtype=np.float32) # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel.
    
    # Calculate the distance from the given position to each voxel center, create another (nz * ny * nx) array where each element is the distance from the given position to that voxel center
    dist = np.sqrt((xx - pos[0])**2 + (yy - pos[1])**2 + (zz - pos[2])**2)
    
    return dist

def direction_to_all(pos, norm_dir_vec, voxel_z, voxel_y, voxel_x):
    """Calculate the angle as well as the cosine of the angle between the direction vector and the vector from a given position to all other positions."""
      
    # Create a grid of voxel coordinates
    zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel
    cos_dir_to_voxel = np.zeros(zz.shape, dtype=np.float32) # initialize an array to store the cosine of the angle between the direction vector and the vector from the given position to each voxel center
    angle_to_voxel = np.zeros(zz.shape, dtype=np.float32) # initialize an array to store the angle between the direction vector and the vector from the given position to each voxel center 

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

    if cos_dir_from_dwell_to_voxel >= 0:
        end_point_L = (dwell_pos[0] + direction_L[0] * (L/2), dwell_pos[1] + direction_L[1] * (L/2), dwell_pos[2] + direction_L[2] * (L/2))
    else:
        end_point_L = (dwell_pos[0] - direction_L[0] * (L/2), dwell_pos[1] - direction_L[1] * (L/2), dwell_pos[2] - direction_L[2] * (L/2))

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

def beta_to_all(dwell_pos, voxel_z, voxel_y, voxel_x, L, direction_L, cos_dir_from_dwell_to_voxel):
    """Calculate the beta angle from a given position to all other positions.
    
    Parameters:
    pos (tuple): The position of the dwell point in world coordinates (x, y, z).
    voxel_z, voxel_y, voxel_x (arrays): The coordinates of the center of each voxel in the volume.
    L (float): The length of the source.
    direction_L: The direction vector of the catheter. in (dx, dy, dz) format and should be normalized to unit vector.
    cos_dir_from_dwell_to_voxel (array): The cosine of the angle between the direction vector and the vector from the dwell position to each voxel center.
    """ 

    # Create a grid of voxel coordinates
    zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel

    beta_angle = np.zeros(zz.shape, dtype=np.float32) # initialize an array to store the beta angle from the given position to each voxel center

    for k in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            for i in range(zz.shape[2]):
                voxel_pos = (zz[k,j,i], yy[k,j,i], xx[k,j,i])
                cos_dir_from_dwell_to_voxel_individual = cos_dir_from_dwell_to_voxel[k,j,i]
                beta_angle[k,j,i] = beta(dwell_pos, voxel_pos, L, direction_L, cos_dir_from_dwell_to_voxel_individual)

    return beta_angle   

def dose_contribution(dwell_pos, norm_dwell_dir, dwell_count, rtdose, volume, spacing, origin, L=3.5):
    """Calculate the dose contribution from all dwell position to the volume grid."""
    
    nz, ny, nx = volume.shape
    voxel_z, voxel_y, voxel_x = voxel_coordinates(volume, spacing, origin) # get the coordinates of the center of each voxel in the volume
    # zz, yy, xx = np.meshgrid(voxel_z, voxel_y, voxel_x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel

    distance = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32) # initialize arrays to store the dose contribution from the dwell position to each voxel in the volume grid
    cos_dir_to_voxel = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32)
    angle_to_voxel = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32)
    beta = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32)

    for i in range(dwell_count):
        # Calculate the dose contribution from the current dwell position
        dist = distance_to_all(dwell_pos[i], voxel_z, voxel_y, voxel_x)
        distance[i] = dist
    
        # Calculate the angle between the local direction vector of the current dwell position and the vector from the current dwell position to each voxel center
        cos_dir, angle = direction_to_all(dwell_pos[i], norm_dwell_dir[i], voxel_z, voxel_y, voxel_x)
        cos_dir_to_voxel[i] = cos_dir
        angle_to_voxel[i] = angle
        beta[i] = beta_to_all(dwell_pos[i], voxel_z, voxel_y, voxel_x, L, norm_dwell_dir[i], cos_dir)

    return distance, cos_dir_to_voxel, angle_to_voxel, beta


    
    