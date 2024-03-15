import numpy as np
import os
from typing import Tuple
from copy import deepcopy
import cv2



def create_rectangle_boundary(resolution: int,
                              boundary_width: float,
                              boundary_height: float,
                              boundary_thickness: float,
                              center_shift: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    assert resolution > 1
    assert boundary_width > 0 and boundary_height > 0 , \
        "Boundary width, height, and thickness must be greater than 0"
    assert boundary_height < 1 and boundary_width < 1, \
        "Boundary width, height, and thickness must be less than 1"
    if center_shift:
        assert boundary_width//2 + np.abs(center_shift[0]) < 1, \
            "Boundary width is too large for the given center"
        assert boundary_height//2 + np.abs(center_shift[1]) < 1, \
            "Boundary height is too large for the given center"
        assert boundary_thickness//2 + np.abs(center_shift[2]) < 1, \
            "Boundary thickness is too large for the given center"

    boundary_array = np.zeros((resolution, resolution, resolution), dtype=np.float32) 
    boundary_array[int(resolution//2 - boundary_width/2 * resolution//2):int(resolution//2 + boundary_width/2 * resolution//2),  
                   int(resolution//2 - boundary_height/2 * resolution//2):int(resolution//2 + boundary_height/2 * resolution//2),
                   int(resolution//2 - boundary_thickness/2 * resolution//2):int(resolution//2 + boundary_thickness/2 * resolution//2)] = 1

    

    #shift the center
    if center_shift:
        pixel_center_shift = [int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]), int(resolution//2*center_shift[2])]
        i_index, j_index, k_index = np.where(boundary_array != 0)
        min_x, max_x = np.min(i_index), np.max(i_index)
        min_y, max_y = np.min(j_index), np.max(j_index)
        min_z, max_z = np.min(k_index), np.max(k_index)

        pixel_center_shift[0] = min(max(pixel_center_shift[0], -min_x), resolution - max_x)
        pixel_center_shift[1] = min(max(pixel_center_shift[1], -min_y), resolution - max_y)
        pixel_center_shift[2] = min(max(pixel_center_shift[2], -min_z), resolution - max_z)
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1,2))
    

    
    return boundary_array

def create_cylinder_boundary(resolution: int,
                             boundary_radius: float,
                             boundary_thickness: float,
                             center_shift: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    assert resolution > 1
    assert boundary_radius > 0 and boundary_thickness > 0, \
        "Boundary radius and thickness must be greater than 0"
    assert boundary_radius < 1, \
        "Boundary radius and thickness must be less than 0.5"
    if center_shift:
        assert boundary_radius + np.abs(center_shift[0]) < 1, \
            "Boundary width is too large for the given center"
        assert boundary_radius + np.abs(center_shift[1]) < 1, \
            "Boundary height is too large for the given center"
        assert boundary_thickness + np.abs(center_shift[2]) < 1, \
            "Boundary thickness is too large for the given center"

    boundary_array = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    x, y, z = np.ogrid[:resolution, :resolution, :resolution]
    mask = (x - resolution // 2) ** 2 + (y - resolution // 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    mask = np.logical_and(mask, z >= resolution//2 - boundary_thickness//2*resolution)
    mask = np.logical_and(mask, z <= resolution//2 + boundary_thickness//2*resolution)

    boundary_array[mask] = 1

    if center_shift:
        pixel_center_shift = [int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]), int(resolution//2*center_shift[2])]
        i_index, j_index, k_index = np.where(boundary_array != 0)
        min_x, max_x = np.min(i_index), np.max(i_index)
        min_y, max_y = np.min(j_index), np.max(j_index)
        min_z, max_z = np.min(k_index), np.max(k_index)

        pixel_center_shift[0] = min(max(pixel_center_shift[0], -min_x), resolution - max_x)
        pixel_center_shift[1] = min(max(pixel_center_shift[1], -min_y), resolution - max_y)
        pixel_center_shift[2] = min(max(pixel_center_shift[2], -min_z), resolution - max_z)
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1,2))

    return boundary_array


def create_sphere_boundary(resolution: int,
                           boundary_radius: float,
                           center_shift: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    assert resolution > 1
    assert boundary_radius > 0, "Boundary radius must be greater than 0"
    assert boundary_radius < 1, "Boundary radius must be less than 1"
    if center_shift:
        assert boundary_radius + np.abs(center_shift[0]) < 1, \
            "Boundary width is too large for the given center"
        assert boundary_radius + np.abs(center_shift[1]) < 1, \
            "Boundary height is too large for the given center"
        assert boundary_radius + np.abs(center_shift[2]) < 1, \
            "Boundary thickness is too large for the given center"

    boundary_array = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    # x, y, = np.ogrid[:resolution, :resolution]
    # mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    # X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    # h = (boundary_radius * resolution//2) ** 2 - (X[mask] - resolution / 2) ** 2 - (Y[mask] - resolution / 2) ** 2
    # z_1 = -np.sqrt(h)*2
    # z_2 = np.sqrt(h)*2
    # for single_z1, single_z2 in zip(z_1, z_2):
    #     assert single_z1 <= single_z2, "z_1 must be less than z_2"
    # # ensure 
    # normalized_z_1 = z_1 / resolution
    # normalized_z_2 = z_2 / resolution
    # boundary_array[mask, 0] = normalized_z_1
    # boundary_array[mask, 1] = normalized_z_2
    x, y, z = np.ogrid[:resolution, :resolution, :resolution]
    mask = (x - resolution // 2) ** 2 + (y - resolution // 2) ** 2 + (z - resolution // 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    boundary_array[mask] = 1

    if center_shift:
        pixel_center_shift = [int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]), int(resolution//2*center_shift[2])]
        i_index, j_index, k_index = np.where(boundary_array != 0)
        min_x, max_x = np.min(i_index), np.max(i_index)
        min_y, max_y = np.min(j_index), np.max(j_index)
        min_z, max_z = np.min(k_index), np.max(k_index)

        pixel_center_shift[0] = min(max(pixel_center_shift[0], -min_x), resolution - max_x)
        pixel_center_shift[1] = min(max(pixel_center_shift[1], -min_y), resolution - max_y)
        pixel_center_shift[2] = min(max(pixel_center_shift[2], -min_z), resolution - max_z)
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1,2))

    return boundary_array


# TODO: implement the cone boundary
def create_cone_boundary(resolution: int,
                         boundary_radius: float,
                         boundary_height: float,
                         center_shift: Tuple[float, float, float] = (0, 0, 0),
                         flip: bool = False) -> np.ndarray:
    assert resolution > 1
    assert boundary_radius > 0 and boundary_height > 0, \
        "Boundary radius and height must be greater than 0"
    assert boundary_radius < 1 and boundary_height < 1, \
        "Boundary radius and height must be less than 1"
    # assert boundary_radius + boundary_height < 1, \
    #     "Boundary radius and height must be less than 1"
    if center_shift:
        assert boundary_radius + np.abs(center_shift[0]) < 1, \
            "Boundary width is too large for the given center"
        assert boundary_radius + np.abs(center_shift[1]) < 1, \
            "Boundary height is too large for the given center"
        assert boundary_height//2 + np.abs(center_shift[2]) < 1, \
            "Boundary thickness is too large for the given center"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    h = (boundary_radius*resolution/2  - np.sqrt((X[mask] - resolution / 2) ** 2 + (Y[mask] - resolution / 2) ** 2)) * \
        boundary_height/resolution/boundary_radius*2

    top = h - boundary_height / 2
    bottom = -boundary_height/ 2
    boundary_array[mask, 0] = bottom
    boundary_array[mask, 1] = top
    if flip:
        boundary_array[:, :, 0], boundary_array[:, :, 1] = -boundary_array[:, :, 1], -boundary_array[:, :, 0]

    if center_shift:
        pixel_center_shift = (resolution//2*center_shift[0], resolution//2*center_shift[1])
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        mask = np.logical_and(boundary_array[:, :, 0] != 0, boundary_array[:, :, 1] != 0)
        boundary_array[mask] = boundary_array[mask] + center_shift[2]
    return boundary_array


def add_boundary(boundaries: list) -> np.ndarray:
    added_boundary = np.zeros_like(boundaries[0])
    for boundary in boundaries:
        added_boundary = np.maximum(added_boundary, boundary)
  
    return added_boundary


def subtract_boundary(boundaries: list) -> np.ndarray:
    """
    Subtract the second boundary from the first boundary
    """
    subtracted_boundary = boundaries[0] - boundaries[1]
    subtracted_boundary[subtracted_boundary < 0] = 0

    return subtracted_boundary



def combine_boundary(boundaries: list,
                     include_all: bool=False,
                     use_subtraction: bool=False,
                     igore_idx: list=None) -> np.ndarray:
    

    combined_boundary = add_boundary(boundaries)

    if use_subtraction:
        for i in range(1, len(boundaries)-1):
            if igore_idx and igore_idx[i]:
                continue
            boundaries[i] = subtract_boundary([boundaries[i], add_boundary(boundaries[:i])])
    if include_all:
        boundaries.append(combined_boundary)

    combined_boundary = np.array(boundaries)
    assert(combined_boundary.shape[0] == len(boundaries))
    assert(len(combined_boundary.shape) == 4), \
        "The combined boundary must be a 4D array"

    return combined_boundary


if __name__ == "__main__":
    save_dir = "custom/threestudio-mvdream/bounds"
    resolution = 512

    rectangle_width = 0.14 # lateral view dimension
    rectangle_height = 0.27 # front view dimension
    rectangle_thickness = 0.05 # z dimension
    rectangle_shift = (0, 0, 0.5)
    rectangle_boundary_1 = create_rectangle_boundary(resolution,
                                                   rectangle_width,
                                                   rectangle_height,
                                                   rectangle_thickness,
                                                   rectangle_shift)


    ball_radius = 0.3
    ball_center_shift = (0, 0, 0.3)
    ball_boundary = create_sphere_boundary(resolution, ball_radius, ball_center_shift)

    # np.save(os.path.join(save_dir, "ball_mask_boundary.npy"), ball_boundary[None, :, :, :])
    # rectangle_width = 0.2 # lateral view dimension
    # rectangle_height = 0.2 # front view dimension
    # rectangle_thickness = 0.21 # z dimension
    # rectangle_shift = (0, 0, 0.2)
    # rectangle_boundary_2 = create_rectangle_boundary(resolution,
    #                                                 rectangle_width,
    #                                                 rectangle_height,
    #                                                 rectangle_thickness,
    #                                                 rectangle_shift)

    rectangle_width = 0.6 # lateral view dimension
    rectangle_height = 0.6 # front view dimension
    rectangle_thickness = 0.8 # z dimension
    rectangle_shift = (0, 0, -0.299)
    rectangle_boundary_3 = create_rectangle_boundary(resolution,
                                                    rectangle_width,
                                                    rectangle_height,
                                                    rectangle_thickness,
                                                    rectangle_shift)
    
    # np.save(os.path.join(save_dir, "rectangle_mask_boundary.npy"), rectangle_boundary_3[None, :, :, :])

    # combined_boundary = combine_boundary([ball_boundary, rectangle_boundary_3],
    #                                      include_all=True,
    #                                      use_subtraction=True,
    #                                      igore_idx=[True, False, False])

    combined_boundary = combine_boundary([ball_boundary, rectangle_boundary_3],
                                         include_all=True,
                                         use_subtraction=True,
                                         igore_idx=[False, False, False])
    print(combined_boundary.shape)
    mask = np.logical_and(combined_boundary[-1, :, :, 0] != 0 , combined_boundary[-1, :, :, 1] != 0)
    np.save(os.path.join(save_dir, "combined_boundary_3body.npy"), combined_boundary)




    