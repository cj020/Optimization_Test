import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from conversion import world_to_voxel

level = 150
width = 100
vmin = level - width / 2
vmax = level + width / 2

def _can_show_interactive():
    return "agg" not in plt.get_backend().lower()

def start_ui(volume, spacing, origin, dwell_positions, show=None):
    """
    Image slice viewer with dwell positions overlaid.

    Parameters
    ----------
    show : bool or None
        If None, opens the viewer only when matplotlib has a GUI backend.
        If False, skip the viewer (e.g. batch runs with Agg).
    """
    if show is None:
        show = _can_show_interactive()
    if not show:
        print("Image viewer skipped (non-interactive backend). Pass show=True with a GUI backend to enable.")
        return

    z_max = volume.shape[0] 

    fig, ax = plt.subplots() 
    plt.subplots_adjust(bottom=0.25) 
    
    # initial slice
    img = ax.imshow(volume[0], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Image Viewer")

    # slider axis
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) 
    slider = Slider(ax_slider, "Slice", 0, z_max - 1, valinit=0, valstep=1) 

    voxel_points = [
    world_to_voxel(p, origin, spacing)
    for p in dwell_positions] # the dwell positions converted to voxel coordinates (z, y, x)

    # update function
    def update(val): 
        z = int(val) 
        img.set_data(volume[z]) 
        img.set_clim(vmin, vmax)

        ax.clear() # clear previous plot
        ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax)

        # plot dwell points in this slice
        for (zz, yy, xx) in voxel_points:
            if zz == z:
                ax.plot(xx, yy, 'ro', markersize=4)

        z_mm = origin[2] + z * spacing[2]
        ax.set_title(f"Slice {z} | z = {z_mm:.2f} mm")

        fig.canvas.draw_idle() # redraw the figure
    
    slider.on_changed(update) 

    plt.show() 