from PIL import Image
import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import List

import cuvslam

def color_from_id(identifier):
    """Generate a color from an identifier."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]

def get_distortion(cam):
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        coeffs = cam['distortion_coefficients']
        if cam['distortion_type'] == "radtan4":
            return has_dist, cuvslam.Distortion(
                cuvslam.Distortion.Model.Brown,
                [coeffs[0], coeffs[1], 0.0, coeffs[2], coeffs[3]]
            )
    
        if cam['distortion_type'] == "radtan5":
            return has_dist, cuvslam.Distortion(
                cuvslam.Distortion.Model.Brown,
                [coeffs[0], coeffs[1], coeffs[4], coeffs[2], coeffs[3]]
            )
        if cam['distortion_type'] == "equid4":
            return has_dist, cuvslam.Distortion(
                cuvslam.Distortion.Model.Fisheye,
                [coeffs[0], coeffs[1], coeffs[2], coeffs[3]]
            )
    return has_dist, None

def load_frame(image_path: str) -> np.ndarray:
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path)
    frame = np.array(image)

    if image.mode == 'L':
        # mono8
        if len(frame.shape) != 2:
            raise ValueError("Expected mono8 image to have 2 dimensions [H W].")
    elif image.mode == 'RGB':
        # rgb8 - convert to BGR for cuvslam compatibility
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected rgb8 image to have 3 dimensions with 3 channels [H W C].")
        # Convert RGB to BGR by reversing the channel order and ensure contiguous
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    elif image.mode == 'I;16':
        # uint16 depth image
        if len(frame.shape) != 2:
            raise ValueError("Expected uint16 depth image to have 2 dimensions [H W].")
        frame = frame.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    return frame

def transform_to_cam0_reference(cam0_transform: np.ndarray, 
                               sensor_transform: np.ndarray) -> np.ndarray:
    """Transform sensor pose to be relative to cam0 (cam0 becomes identity)."""
    
    cam0_body = np.linalg.inv(cam0_transform)
    return cam0_body @ sensor_transform

def transform_to_pose(transform_16: List[float]) -> cuvslam.Pose:
    """Convert a 4x4 transformation matrix to a cuvslam.Pose object."""
    transform = np.array(transform_16).reshape(4, 4)
    rotation_quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return cuvslam.Pose(rotation=rotation_quat, translation=transform[:3, 3])