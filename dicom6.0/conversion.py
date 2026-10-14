import numpy as np

def world_to_voxel(pos, origin, spacing):
    x = int((pos[0] - origin[0]) / spacing[0])
    y = int((pos[1] - origin[1]) / spacing[1])
    z = int((pos[2] - origin[2]) / spacing[2])
    return z, y, x