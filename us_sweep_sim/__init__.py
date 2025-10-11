import math

import numpy as np
from moviepy import ImageSequenceClip
from PIL import Image


class VolumeSlicer:
    def __init__(self, npz_file_path=None, np_data=None):
        """
        Initializes the VolumeSlicer with the path to an NPZ file.
        """
        self.volume = None
        self.volume_dims = None

        if (
            np_data is not None
            and np_data.size > 0
            and npz_file_path is not None
        ) or (np_data is None and npz_file_path is None):
            raise ValueError(
                "Either np_data or npz_file_path must be provided, but not both."
            )

        if np_data is not None:
            self.volume = np_data
            self.volume_dims = self.volume.shape[:3]
        if npz_file_path is not None:
            self.load_volume(npz_file_path)

    def load_volume(self, npz_file_path):
        """
        Loads the volume data from the NPZ file.
        """
        try:
            npz_file = np.load(npz_file_path)
            keys = list(npz_file.files)
            if not keys:
                raise ValueError("No arrays found in the NPZ file.")
            self.volume = npz_file[keys[0]]
            if self.volume.ndim not in (3, 4):
                raise ValueError(
                    "Expected a 3D or 4D array. Got shape:"
                    f" {self.volume.shape}"
                )

            if self.volume.ndim == 4:
                self.volume_dims = self.volume.shape[:3]
            else:
                self.volume_dims = self.volume.shape
        except Exception as e:
            raise ValueError(f"Error loading NPZ file: {e}")

    def extract_slice(self, z_pos, x_pos, y_pos, angle_x, angle_y):
        """
        Extracts a 2D slice from the 3D volume based on the given parameters.

        Args:
            z_pos (int): Z-coordinate of the slice.
            x_pos (int): X-coordinate of the slice.
            y_pos (int): Y-coordinate of the slice.
            angle_x (float): Rotation angle around the X-axis (in degrees).
            angle_y (float): Rotation angle around the Y-axis (in degrees).

        Returns:
            PIL.Image.Image: A Pillow Image object representing the extracted slice.
        """
        if self.volume is None:
            raise ValueError("Volume not loaded.  Call load_volume() first.")

        nZ, nY, nX = self.volume_dims[:3]

        # Convert angles to radians.
        ax = math.radians(angle_x)
        ay = math.radians(angle_y)

        # Compute rotation matrices for x and y.
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(ax), -math.sin(ax)],
                [0, math.sin(ax), math.cos(ax)],
            ]
        )
        Ry = np.array(
            [
                [math.cos(ay), 0, math.sin(ay)],
                [0, 1, 0],
                [-math.sin(ay), 0, math.cos(ay)],
            ]
        )
        # Compose rotations (first rotate about X, then Y).
        R = Ry.dot(Rx)

        # Determine the slicing plane.
        # The default slice has normal (0, 0, 1) with in-plane axes (1, 0, 0) and (0, 1, 0).
        normal = R.dot(np.array([0, 0, 1]))
        u_axis = R.dot(np.array([1, 0, 0]))
        v_axis = R.dot(np.array([0, 1, 0]))

        channels = 1
        if self.volume.ndim == 4:
            channels = self.volume.shape[3]

        # Define the center of the volume in (x, y, z) coordinates.
        center = np.array([(nX - 1) / 2.0, (nY - 1) / 2.0, (nZ - 1) / 2.0])

        # Set output slice parameters.
        out_size = 256  # Resolution of the output slice (256 x 256)
        L = max(nX, nY, nZ)
        # Create a grid in plane coordinates (u, v)
        u = np.linspace(-L / 2, L / 2, out_size)
        v = np.linspace(-L / 2, L / 2, out_size)
        U, V = np.meshgrid(
            u, v, indexing="xy"
        )  # each of shape (out_size, out_size)

        # For each output pixel compute the corresponding (x, y, z) in volume:
        # p = center + (slice_offset * normal) + (u * u_axis) + (v * v_axis)
        # The problem was that the x_pos and y_pos were not being used to offset the center.
        P = (
            np.array(
                [x_pos - center[0], y_pos - center[1], z_pos - center[2]]
            ).reshape(1, 1, 3)
            + center.reshape(1, 1, 3)
            + U[..., np.newaxis] * u_axis.reshape(1, 1, 3)
            + V[..., np.newaxis] * v_axis.reshape(1, 1, 3)
        )
        # P has shape (out_size, out_size, 3) with (x, y, z) coordinates.
        X = P[..., 0]
        Y = P[..., 1]
        Z = P[..., 2]

        # Convert to volume indices; volume indices are accessed as volume[z, y, x].
        x_idx = np.rint(X).astype(np.int32)
        y_idx = np.rint(Y).astype(np.int32)
        z_idx = np.rint(Z).astype(np.int32)

        # Clip indices so they lie within the volume bounds.
        x_idx = np.clip(x_idx, 0, nX - 1)
        y_idx = np.clip(y_idx, 0, nY - 1)
        z_idx = np.clip(z_idx, 0, nZ - 1)

        # Sample the volume using nearest-neighbor interpolation.
        if channels == 1:
            slice_img = self.volume[z_idx, y_idx, x_idx]
        else:
            slice_img = self.volume[z_idx, y_idx, x_idx, :]

        # Normalize to uint8 if needed.
        if slice_img.dtype != np.uint8:
            m = slice_img.min()
            M = slice_img.max()
            if M > m:
                slice_img = (255 * (slice_img - m) / (M - m)).astype(np.uint8)
            else:
                slice_img = slice_img.astype(np.uint8)

        # Convert the numpy slice to a Pillow image.
        if channels == 1:
            pil_im = Image.fromarray(slice_img, mode="L")
        elif channels == 3:
            pil_im = Image.fromarray(slice_img, mode="RGB")
        elif channels == 4:
            pil_im = Image.fromarray(slice_img, mode="RGBA")
        else:
            pil_im = Image.fromarray(slice_img.astype(np.uint8))

        return pil_im

    def generate_frame_parameters(
        self,
        num_frames=100,
        movement_ranges=None,
        rotation_ranges=None,
        position_limits=None,
        rotation_limits=None,
        start_pos=None,
        restrict_rotation_near_z_limits=True,
    ):
        """Generates a list of frame parameters for creating a synthetic video.

        Args:
            num_frames (int): Number of frames in the video.
            movement_ranges (dict): Maximum change in position per frame for
                                    each dimension.
                                    Example: {"z": 10, "x": 5, "y": 2}.
            rotation_ranges (dict): Maximum rotational change (degrees) per frame
                                    for each axis.
                                    Example: {"x": 5, "y": 3}.
            position_limits (dict): Limits for each dimension.
                                    Example: {"z": (0, 100), "x": (0, 50),
                                            "y": (0, 50)}.
            rotation_limits (dict): Limits for rotation angles (degrees).
                                    Example: {"x": (-30, 30), "y": (-20, 20)}.
            start_pos (tuple): Starting position and angles for the probe
            restrict_rotation_near_z_limits (bool): If True, rotation limits
                                                    become more restrictive as
                                                    z_pos approaches its limits.
                                                    Defaults to False.

        Returns:
            list: A list of tuples, where each tuple contains the (z_pos, x_pos,
                  y_pos, angle_x, angle_y) for a frame.
        """
        if self.volume is None:
            raise ValueError("Volume not loaded. Call load_volume() first.")

        # Set default movement ranges if not provided.
        if movement_ranges is None:
            movement_ranges = {"z": 10, "x": 5, "y": 2}

        # Set default rotation ranges if not provided.
        if rotation_ranges is None:
            rotation_ranges = {"x": 5, "y": 3}

        # Set default position limits if not provided.
        if position_limits is None:
            position_limits = {
                "z": (0, self.volume_dims[0] - 1),
                "x": (0, self.volume_dims[2] - 1),
                "y": (0, self.volume_dims[1] - 1),
            }

        # Set default rotation limits if not provided.
        if rotation_limits is None:
            rotation_limits = {"x": (-30, 30), "y": (-30, 30)}

        # Initial position and angles.
        if start_pos is not None:
            z_pos, x_pos, y_pos, angle_x, angle_y = start_pos
        else:
            z_pos = self.volume_dims[0] // 2
            x_pos = self.volume_dims[2] // 2
            y_pos = self.volume_dims[1] // 2
            angle_x = 0.0
            angle_y = 0.0

        # Initialize velocity terms for smoothing.
        velocity_z = 0.0
        velocity_x = 0.0
        velocity_y = 0.0
        rot_velocity_x = 0.0
        rot_velocity_y = 0.0

        # Smoothing factors. A value closer to 1 makes the movement more
        # “memoryful.”
        pos_smoothing = 0.9
        rot_smoothing = 0.9

        import random

        # List to store frame parameters.
        frame_parameters = []

        for _ in range(num_frames):
            # Generate target offsets for each dimension.
            target_delta_z = random.uniform(
                -movement_ranges["z"], movement_ranges["z"]
            )
            target_delta_x = random.uniform(
                -movement_ranges["x"], movement_ranges["x"]
            )
            target_delta_y = random.uniform(
                -movement_ranges["y"], movement_ranges["y"]
            )

            # Predicted next positions
            next_z_pos = z_pos + (
                pos_smoothing * velocity_z
                + (1 - pos_smoothing) * target_delta_z
            )
            next_x_pos = x_pos + (
                pos_smoothing * velocity_x
                + (1 - pos_smoothing) * target_delta_x
            )
            next_y_pos = y_pos + (
                pos_smoothing * velocity_y
                + (1 - pos_smoothing) * target_delta_y
            )

            # Check for clipping and reverse direction if needed
            if not (
                position_limits["z"][0]
                <= next_z_pos
                <= position_limits["z"][1]
            ):
                target_delta_z *= -1  # Reverse Z direction
            if not (
                position_limits["x"][0]
                <= next_x_pos
                <= position_limits["x"][1]
            ):
                target_delta_x *= -1  # Reverse X direction
            if not (
                position_limits["y"][0]
                <= next_y_pos
                <= position_limits["y"][1]
            ):
                target_delta_y *= -1  # Reverse Y direction

            # Update velocities using smoothing (this creates a momentum effect).
            velocity_z = (
                pos_smoothing * velocity_z
                + (1 - pos_smoothing) * target_delta_z
            )
            velocity_x = (
                pos_smoothing * velocity_x
                + (1 - pos_smoothing) * target_delta_x
            )
            velocity_y = (
                pos_smoothing * velocity_y
                + (1 - pos_smoothing) * target_delta_y
            )

            # Update positions.
            z_pos += velocity_z
            x_pos += velocity_x
            y_pos += velocity_y

            # Clamp positions to valid indices based on position limits.
            z_pos = int(
                np.clip(
                    z_pos, position_limits["z"][0], position_limits["z"][1]
                )
            )
            x_pos = int(
                np.clip(
                    x_pos, position_limits["x"][0], position_limits["x"][1]
                )
            )
            y_pos = int(
                np.clip(
                    y_pos, position_limits["y"][0], position_limits["y"][1]
                )
            )

            RESTRICT = 0.15
            if restrict_rotation_near_z_limits:
                # Scale rotation limits based on z position
                z_range = position_limits["z"][1] - position_limits["z"][0]
                z_proximity = min(
                    (z_pos - position_limits["z"][0]) / (z_range * RESTRICT),
                    (position_limits["z"][1] - z_pos) / (z_range * RESTRICT),
                )  # 0 when near boundary, 1 when far
                z_proximity = np.clip(z_proximity, 0, 1)

                scaled_rotation_limits = {
                    axis: (
                        rotation_limits[axis][0] * z_proximity,
                        rotation_limits[axis][1] * z_proximity,
                    )
                    for axis in rotation_limits
                }
            else:
                scaled_rotation_limits = rotation_limits

            # Generate target rotational offsets for each axis.
            target_rot_x = random.uniform(
                -rotation_ranges["x"], rotation_ranges["x"]
            )
            target_rot_y = random.uniform(
                -rotation_ranges["y"], rotation_ranges["y"]
            )

            # Update rotational velocities using smoothing.
            rot_velocity_x = (
                rot_smoothing * rot_velocity_x
                + (1 - rot_smoothing) * target_rot_x
            )
            rot_velocity_y = (
                rot_smoothing * rot_velocity_y
                + (1 - rot_smoothing) * target_rot_y
            )

            # Update rotation angles.
            angle_x += rot_velocity_x
            angle_y += rot_velocity_y

            # Clamp rotation angles to valid ranges based on scaled rotation
            # limits.
            angle_x = np.clip(
                angle_x,
                scaled_rotation_limits["x"][0],
                scaled_rotation_limits["x"][1],
            )
            angle_y = np.clip(
                angle_y,
                scaled_rotation_limits["y"][0],
                scaled_rotation_limits["y"][1],
            )

            # Append the frame parameters to the list.
            frame_parameters.append((z_pos, x_pos, y_pos, angle_x, angle_y))

        return frame_parameters

    def create_synthetic_video_from_params(
        self, output_path, frame_parameters
    ):
        """Creates a synthetic video from a list of frame parameters.

        Args:
            output_path (str): The path to save the video (e.g., "video.mp4").
            frame_parameters (list): A list of tuples, where each tuple
                                     contains the (z_pos, x_pos, y_pos,
                                     angle_x, angle_y) for a frame.
        """
        if self.volume is None:
            raise ValueError("Volume not loaded. Call load_volume() first.")

        # List to store frames as PIL images.
        frames = []

        for z_pos, x_pos, y_pos, angle_x, angle_y in frame_parameters:
            # Extract the slice.
            img = self.extract_slice(z_pos, x_pos, y_pos, angle_x, angle_y)

            # Append the frame (PIL image) to the list.
            frames.append(img)

        # Convert the list of PIL images to a video using moviepy.
        try:
            # Convert PIL images to numpy arrays for moviepy.
            frames_np = [np.array(frame) for frame in frames]

            # Reverse the frames to create a loop.
            frames_np = frames_np[::-1]

            # Create a video clip from the frames.
            clip = ImageSequenceClip(frames_np, fps=20)

            # Write the video to the output path.
            clip.write_videofile(output_path, codec="libx264", fps=20)
            print(f"Video saved to {output_path}")
        except Exception as e:
            raise ValueError(f"Error creating video: {e}")

    def create_synthetic_video(
        self,
        output_path,
        num_frames=100,
        movement_ranges=None,
        rotation_ranges=None,
        position_limits=None,
        rotation_limits=None,
        start_pos=None,
        restrict_rotation_near_z_limits=True,
    ):
        """Creates a synthetic video by generating frames that simulate an
        ultrasound probe moving over skin with smooth, natural-like motions.

        Args:
            output_path (str): The path to save the video (e.g., "video.mp4").
            num_frames (int): Number of frames in the video.
            movement_ranges (dict): Maximum change in position per frame for
                                    each dimension.
                                    Example: {"z": 10, "x": 5, "y": 2}.
            rotation_ranges (dict): Maximum rotational change (degrees) per frame
                                    for each axis.
                                    Example: {"x": 5, "y": 3}.
            position_limits (dict): Limits for each dimension.
                                    Example: {"z": (0, 100), "x": (0, 50),
                                            "y": (0, 50)}.
            rotation_limits (dict): Limits for rotation angles (degrees).
                                    Example: {"x": (-30, 30), "y": (-20, 20)}.
            start_pos (tuple): Starting position and angles for the probe.
            restrict_rotation_near_z_limits (bool): If True, rotation limits
                                                    become more restrictive as
                                                    z_pos approaches its limits.
                                                    Defaults to False.
        """
        frame_parameters = self.generate_frame_parameters(
            num_frames,
            movement_ranges,
            rotation_ranges,
            position_limits,
            rotation_limits,
            start_pos,
            restrict_rotation_near_z_limits,
        )
        self.create_synthetic_video_from_params(output_path, frame_parameters)

        return frame_parameters
