from us_sweep_sim import VolumeSlicer

slicer = VolumeSlicer(
    "env/files/PHI078_2_R.npz"
)  # Replace with the actual path

# Extract a slice
image = slicer.extract_slice(
    z_pos=50, x_pos=75, y_pos=100, angle_x=10.0, angle_y=-5.0
)

# Save the slice to a file
image.save("env/files/slice.png")

movement_ranges = {
    "z": 30,
    "x": 12,
    "y": 6,
}

# Define rotation ranges for each axis
rotation_ranges = {
    "x": 0,
    "y": 15,
}

# Define position limits for each dimension
position_limits = {
    "z": (
        50,
        slicer.volume_dims[0] - 50,
    ),  # Clamp z positions between 10 and 90
    "x": (
        50,
        slicer.volume_dims[2] - 50,
    ),  # Clamp x positions between 5 and 45
    "y": (
        50,
        slicer.volume_dims[1] - 50,
    ),  # Clamp y positions between 5 and 45
}

rotation_limits = {
    "x": (0, 0),  # Clamp x-axis rotation between -20 and 20 degrees
    "y": (-30, 30),  # Clamp y-axis rotation between -10 and 10 degrees
}

# Create a synthetic video
frame_parameters = slicer.create_synthetic_video(
    "env/files/synthetic_video.mp4",
    num_frames=300,
    movement_ranges=movement_ranges,
    rotation_ranges=rotation_ranges,
    position_limits=position_limits,
    rotation_limits=rotation_limits,
    start_pos=(128, 220, 315, 0, 2),
)

slicer2 = VolumeSlicer(
    "env/files/PHI078_2_R-labels.npz"
)  # Replace with the actual path
slicer2.create_synthetic_video_from_params(
    "env/files/synthetic_video_labels.mp4", frame_parameters
)
