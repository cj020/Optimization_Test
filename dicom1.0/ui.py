import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

level = 150
width = 100
vmin = level - width / 2
vmax = level + width / 2

def start_ui(volume):
    z_max = volume.shape[0] # z_max = volume.shape[0]: retrieves the number of slices in the volume, which is the size of the first dimension of the NumPy array. This value is used to set the range of the slider for navigating through the slices in the CT viewer.

    fig, ax = plt.subplots() # plt.subplots(): creates a new figure and a set of subplots (in this case, just one subplot) for displaying the CT images. The function returns a figure object (fig) and an axes object (ax) that can be used to customize the appearance of the plot and display the images.
    plt.subplots_adjust(bottom=0.25) # plt.subplots_adjust(bottom=0.25): adjusts the layout of the subplots to make room for the slider at the bottom of the figure. The bottom parameter specifies the amount of space to leave at the bottom of the figure, as a fraction of the total figure height. In this case, it leaves 25% of the figure height for the slider, allowing it to be displayed without overlapping with the CT images.

    # initial slice
    img = ax.imshow(volume[0], cmap="gray", vmin=vmin, vmax=vmax)
    # img = ax.imshow(volume[0], cmap="gray") # ax.imshow(volume[0], cmap="gray"): displays the first slice of the CT volume (volume[0]) as a grayscale image on the axes (ax). The imshow function is used to display the image data, and the cmap="gray" argument specifies that the image should be displayed using a grayscale colormap, which is appropriate for CT images where pixel intensity represents tissue density.
    ax.set_title("CT Viewer")

    # slider axis
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03]) # plt.axes([0.2, 0.1, 0.65, 0.03]): creates a new set of axes for the slider widget. The argument is a list of four values that specify the position and size of the axes in normalized figure coordinates (left, bottom, width, height). In this case, the slider will be positioned at 20% from the left, 10% from the bottom, and will have a width of 65% and a height of 3% of the total figure size.
    slider = Slider(ax_slider, "Slice", 0, z_max - 1, valinit=0, valstep=1) # slider = Slider(ax_slider, "Slice", 0, z_max - 1, valinit=0, valstep=1): creates a slider widget on the specified axes (ax_slider) with the label "Slice". The slider allows the user to select a slice index from 0 to z_max - 1 (the range of valid slice indices in the volume). The valinit=0 argument sets the initial value of the slider to 0 (the first slice), and valstep=1 ensures that the slider moves in increments of 1, allowing the user to select only integer slice indices.

    # update function
    def update(val): # update(val): defines a function that will be called whenever the slider value changes. The val parameter will receive the current value of the slider, which corresponds to the selected slice index.
        z = int(val) # z = int(val): converts the slider value (which may be a float) to an integer, since slice indices must be integers. This ensures that the correct slice is displayed when the slider is moved.
        img.set_data(volume[z]) # img.set_data(volume[z]): updates the image data displayed in the axes (img) to show the slice corresponding to the selected index (z). This function changes the pixel data of the displayed image without creating a new plot, allowing for efficient updating of the visualization as the user interacts with the slider.
        img.set_clim(vmin, vmax)
        fig.canvas.draw() # fig.canvas.draw_idle(): triggers a redraw of the figure canvas to update the displayed image with the new slice. The draw_idle() function is used to schedule a redraw of the figure at the next available opportunity, which helps to optimize performance by avoiding unnecessary redraws while the user is still interacting with the slider.

    slider.on_changed(update) # slider.on_changed(update): connects the update function to the slider's on_changed event, so that the update function will be called automatically whenever the user changes the slider value. This allows the displayed image to be updated in real-time as the user navigates through the slices of the CT volume.

    plt.show() # plt.show(): displays the figure with the CT viewer and the slider. This function starts the event loop for the interactive visualization, allowing the user to interact with the slider and see the corresponding slices of the CT volume. The figure will remain open until the user closes it.