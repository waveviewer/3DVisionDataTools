import os
from pathlib import Path
import math
import re
from typing import Dict

import numpy as np

"""
NHR Camera params format: https://github.com/wuminye/NHR

├──  CamPose.inf  Camera extrinsics. 
In each row, the 3x4 [R T] matrix is displayed in columns, 
with the third column followed by columns 1, 2, and 4, where R*X^{camera}+T=X^{world}.
│
└──  Intrinsic.inf  Camera intrinsics. 
The format of each intrinsics is: "idx \n fx 0 cx \n 0 fy cy \n 0 0 1 \n \n" (idx starts from 0)
"""


def extract_cameras_intrinsic(cameras_txt_path: Path) -> Dict:
    """
    Extract intrinsic camera parameters from a cameras.txt file.

    Args:
        cameras_txt_path (Path): Path to the cameras.txt file.

    Returns:
        dict: A dictionary containing the intrinsic camera parameters for each camera.
            The dictionary key is the camera ID, and the value is a dictionary containing the following keys:
            - 'w': Width of the camera image.
            - 'h': Height of the camera image.
            - 'fl_x': Focal length along the x-axis.
            - 'fl_y': Focal length along the y-axis.
            - 'k1': Radial distortion coefficient k1.
            - 'k2': Radial distortion coefficient k2.
            - 'k3': Radial distortion coefficient k3.
            - 'k4': Radial distortion coefficient k4.
            - 'p1': Tangential distortion coefficient p1.
            - 'p2': Tangential distortion coefficient p2.
            - 'cx': Principal point x-coordinate.
            - 'cy': Principal point y-coordinate.
            - 'is_fisheye': Indicates whether the camera model is fisheye (True/False).
            - 'camera_angle_x': Camera angle in the x-axis in radians.
            - 'camera_angle_y': Camera angle in the y-axis in radians.
            - 'fovx': Field of view in the x-axis in degrees.
            - 'fovy': Field of view in the y-axis in degrees.
    """
    cameras = {}
    with open(cameras_txt_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elements = line.split()
            camera = {}
            camera_id = int(elements[0])
            camera["w"] = float(elements[2])
            camera["h"] = float(elements[3])
            camera["fl_x"] = float(elements[4])
            camera["fl_y"] = float(elements[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False

            model_type = elements[1]
            if model_type in ["SIMPLE_PINHOLE", "PINHOLE"]:
                camera["fl_y"] = float(elements[5])
                camera["cx"] = float(elements[6])
                camera["cy"] = float(elements[7])
            elif model_type in ["SIMPLE_RADIAL", "RADIAL"]:
                camera["cx"] = float(elements[5])
                camera["cy"] = float(elements[6])
                camera["k1"] = float(elements[7])
                if model_type == "RADIAL":
                    camera["k2"] = float(elements[8])
            elif model_type == "OPENCV":
                camera["fl_y"] = float(elements[5])
                camera["cx"] = float(elements[6])
                camera["cy"] = float(elements[7])
                camera["k1"] = float(elements[8])
                camera["k2"] = float(elements[9])
                camera["p1"] = float(elements[10])
                camera["p2"] = float(elements[11])
            elif model_type.endswith("_FISHEYE"):
                camera["is_fisheye"] = True
                camera["cx"] = float(elements[5])
                camera["cy"] = float(elements[6])
                camera["k1"] = float(elements[7])
                if model_type == "RADIAL_FISHEYE":
                    camera["k2"] = float(elements[8])
                elif model_type == "OPENCV_FISHEYE":
                    camera["fl_y"] = float(elements[5])
                    camera["cx"] = float(elements[6])
                    camera["cy"] = float(elements[7])
                    camera["k1"] = float(elements[8])
                    camera["k2"] = float(elements[9])
                    camera["k3"] = float(elements[10])
                    camera["k4"] = float(elements[11])
            else:
                print("Unknown camera model:", model_type)

            camera["camera_angle_x"] = (
                math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            )
            camera["camera_angle_y"] = (
                math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            )
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

            cameras[camera_id] = camera

    return cameras


def qvec2rotmat(qvec):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        qvec (numpy.ndarray): Quaternion represented as a 4-element array.

    Returns:
        numpy.ndarray: 3x3 rotation matrix.

    Example:
        >>> qvec = np.array([0.5, 0.5, 0.5, 0.5])
        >>> result = qvec2rotmat(qvec)
        >>> print(result)
        [[-0.33333333  0.66666667  0.66666667]
         [ 0.66666667 -0.33333333  0.66666667]
         [ 0.66666667  0.66666667 -0.33333333]]
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def parse_images_txt(images_txt: Path):
    """
    Parse the contents of an images.txt file to extract camera pose information.

    Args:
        images_txt (Path): Path to the images.txt file.

    Returns:
        dict: A dictionary containing camera pose information for each camera ID.
            The dictionary keys are camera IDs, and the values are dictionaries with the following keys:
            - 'c2w': The 3x4 camera-to-world transformation matrix.
            - 'image_id': The ID of the corresponding image.

    Note:
        The function expects the images.txt file to be formatted as follows:
        - Each line represents a camera pose, alternating between camera parameters and image file information.
        - The camera parameters should contain the camera ID, quaternion, translation, and filename.
        - The quaternion should be specified as four space-separated values (w x y z).
        - The translation should be specified as three space-separated values (x y z).
        - The filename should be specified as the last element, potentially containing spaces.

    Example:
        # Example images.txt format:
        # 1 0.57735 0.57735 0.57735 0 0 0 example_image_1.png
        # 2 0.707107 -0.707107 0 0.5 0.5 0.5 example_image_2.png
        # ...

    """
    cameras_pose_info = {}
    with open(images_txt, "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 1:
                elems = line.split(
                    " "
                )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                image_id = int(Path(elems[9]).stem)
                camera_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                cameras_pose_info[camera_id] = {
                    "c2w": c2w[:3, :],
                    "image_id": image_id,
                }
    # print("length of parsed camera poses ", len(cameras_pose_info))
    return cameras_pose_info


if __name__ == "__main__":
    camera_intrinsic_dict = extract_cameras_intrinsic(
        Path("COLMAP_Cali/cameras.txt")
    )
    camera_pose_dict = parse_images_txt(Path("COLMAP_Cali/images.txt"))
    camera_pose_dict = dict(
        sorted(camera_pose_dict.items(), key=lambda x: x[1]["image_id"])
    )
    save_rt_list = []
    with open("Intrinsic.inf", "w") as output_k:
        for camera_id, value in camera_pose_dict.items():
            save_id = value["image_id"]
            if save_id < 50:
                save_id = save_id - 1
            else:
                save_id = save_id - 2
            camera_info = camera_intrinsic_dict[camera_id]
            fl_x = camera_info["fl_x"]
            fl_y = camera_info["fl_y"]
            cx = camera_info["cx"]
            cy = camera_info["cy"]
            rt_3x4 = value["c2w"]
            nhr_rt = rt_3x4[:, [2, 0, 1, 3]].T
            save_rt_list.append(nhr_rt.reshape(1, -1))
            write_k = (
                f"{save_id} \n {fl_x} 0 {cx} \n 0 {fl_y} {cy} \n 0 0 1 \n \n"
            )
            output_k.write(write_k)
    with open("CamPose.inf", "w") as output_rt:
        save_rt_array = np.array(save_rt_list).reshape(-1, 12)
        print(save_rt_array.shape)
        np.savetxt(output_rt, save_rt_array)
        pass
    pass
