from PIL import Image
import math

def change_fps(input_file, output_file, new_fps):
    # Open the GIF file
    gif = Image.open(input_file)

    # Calculate the frame duration based on the desired fps
    frame_duration = math.ceil(1000 / new_fps)  # Duration in milliseconds

    # Modify the frame duration and disposal method for each frame
    frames = []
    for frame in range(gif.n_frames):
        gif.seek(frame)
        frame_copy = gif.copy()
        frame_copy.info['duration'] = frame_duration
        frame_copy.info['disposal'] = 2  # Set disposal method to "restore to background color"
        frames.append(frame_copy)

    # Save the modified GIF to the output file
    frames[0].save(output_file, save_all=True, append_images=frames[1:], loop=0, disposal=2)


change_fps("R=0.2 250 Hz  fs 22500 0.014 s  x plane.gif", "R=0.2 250 Hz fs 22500 0.014 s x plane new fps1.gif", 15)
change_fps("R=0.2 250 Hz  fs 22500 0.014 s  y plane.gif", "R=0.2 250 Hz fs 22500 0.014 s y plane new fps1.gif", 15)