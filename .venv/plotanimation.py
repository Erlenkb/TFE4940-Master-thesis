import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



def animate_heatmaps(npz_filename, output_filename):
    data = np.load(npz_filename)

    heatmap_keys = sorted([key for key in data.keys() if key.startswith('heatmap_')])
    n_z = len(heatmap_keys)
    n_rows, n_cols = data[heatmap_keys[0]].shape

    fig, ax = plt.subplots()

    def update(frame):
        ax.cla()  # Clear previous heatmap

        heatmap = data[heatmap_keys[frame]].reshape((n_rows, n_cols))
        ax.imshow(heatmap, cmap='hot')

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_title(f"Frame: {frame+1}/{n_z}")

    animation = FuncAnimation(fig, update, frames=n_z, interval=500)

    # Save the animation as a GIF file
    animation.save(output_filename, writer='pillow')

    plt.close(fig)







animate_heatmaps("press_arr.npz", "testing.gif")





def read_array_from_file(file_path):
    # Read the array from the CSV file
    data_frame = pd.read_csv(file_path, header=None)
    array = data_frame.values
    
    print(array)
    N, M = array.shape
    print(N, " ", M, " ")
    # Reshape the 2D array back to 3D
    N, M = array.shape
    L = int(M / N)
    reshaped_array = np.reshape(array, (N, N, L))

    # Return the reshaped array
    
    render_heatmap_animation(array)
    return reshaped_array



def render_heatmap_animation(heatmap_array):
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Create an empty plot
    heatmap = ax.imshow(heatmap_array[:,:,0], cmap='hot')

    def update(frame):
        # Update the heatmap data for each frame
        heatmap.set_array(heatmap_array[frame])
        return heatmap,

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(heatmap_array), interval=200)

    # Show the animation
    plt.show()


def read_3d_array_from_csv(file_path):
    df = pd.read_csv(file_path)
    matrix = df.values
    N, M = matrix.shape
    print(N, " ", M, " ")
    print(len(matrix))
    print(matrix)
    nx, nz = matrix.shape
    ny = 8
    nx = 8
    array = np.reshape(matrix, (8, 8, 25))
    
    render_heatmap_animation(array)
    return array





    
    
    
    
    

def generate_heatmap_animation(array):
    # Get the shape of the array
    N, M, L = array.shape
    print(N, " ", M, " ",L)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Define the animation update function
    def update(frame):
        # Clear the previous plot
        ax.cla()

        # Create a heatmap for the current frame
        heatmap = array[:, :, frame]

        # Plot the heatmap
        ax.imshow(heatmap, cmap='hot')

        # Set the title and labels
        ax.set_title(f'Frame {frame+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=L, interval=200)
    print(array)
    # Set up the writer for saving the animation as an MP4 file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    plt.show()
    # Save the animation as an MP4 file
    file_path = 'heatmap_animation.mp4'
    #anim.save(file_path, writer=writer)

    # Show the final plot (optional)

    # Return the file path
    return file_path



#read_array_from_file("press_arr.csv")
#read_3d_array_from_csv("press_arr.csv")