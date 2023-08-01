from PIL import Image
import os

def split_gif_into_frames(gif_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the GIF and get the number of frames
    with Image.open(gif_path) as gif:
        num_frames = gif.n_frames

        # Loop through each frame and save it in the output folder
        for frame_number in range(num_frames):
            gif.seek(frame_number)
            frame = gif.copy()
            frame.save(os.path.join(output_folder, f"frame_{frame_number}.png"))

if __name__ == "__main__":
    # Example usage:
    gif_path = "draw_Frames.gif"
    output_folder = "output_frames_folder_draw"
    split_gif_into_frames(gif_path, output_folder)