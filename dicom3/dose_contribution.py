import numpy as np

def distance_to_all(pos, volume, spacing, origin):
    """Calculate the distance from a given position to all other positions."""
    
    nz, ny, nx = volume.shape
    
    # Create a grid of voxel coordinates
    z = (np.arange(nz) + 0.5) * spacing[2] + origin[2]  # z coordinates of voxel centers (si +0.5) in mm
    y = (np.arange(ny) + 0.5) * spacing[1] + origin[1]
    x = (np.arange(nx) + 0.5) * spacing[0] + origin[0]
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij') # creates three (nz * ny * nx) arrays where each stores one coordinate component (x, y, or z) for every voxel

    # Calculate the distance from the given position to each voxel center, create another (nz * ny * nx) array where each element is the distance from the given position to that voxel center
    dist = np.sqrt((xx - pos[0])**2 + (yy - pos[1])**2 + (zz - pos[2])**2)
    
    return dist

def dose_contribution(dwell_pos, dwell_count, rtdose, volume, spacing, origin):
    """Calculate the dose contribution from all dwell position to the volume grid."""
    
    nz, ny, nx = volume.shape

    distance = np.zeros((dwell_count, nz, ny, nx), dtype=np.float32) # initialize arrays to store the dose contribution from the dwell position to each voxel in the volume grid
    
    for i in range(10):
        # Calculate the dose contribution from the current dwell position
        dist = distance_to_all(dwell_pos[i], volume, spacing, origin)
        distance[i] = dist
    
    return distance