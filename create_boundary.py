import numpy as np
import os
from typing import Tuple
from copy import deepcopy


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
        assert boundary_width + np.abs(center_shift[0]) < 1, \
            "Boundary width is too large for the given center"
        assert boundary_height + np.abs(center_shift[1]) < 1, \
            "Boundary height is too large for the given center"
        assert boundary_thickness + np.abs(center_shift[2]) < 1, \
            "Boundary thickness is too large for the given center"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32) 
    boundary_array[int(resolution//2 - boundary_width * resolution//2):int(resolution// 2 + boundary_width * resolution//2),  
                   int(resolution//2 - boundary_height * resolution//2):int(resolution//2 + boundary_height * resolution//2), 0] = -boundary_thickness
    boundary_array[int(resolution//2 - boundary_width * resolution//2):int(resolution// 2 + boundary_width * resolution//2),  
                   int(resolution//2 - boundary_height * resolution//2):int(resolution//2 + boundary_height * resolution//2), 1] = boundary_thickness
    

    #shift the center
    if center_shift:
        pixel_center_shift = (resolution//2*center_shift[0], resolution//2*center_shift[1])
        # make sure the shift is not out of boundary

        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        boundary_array = boundary_array + center_shift[2]


    
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

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    boundary_array[mask, 0] = -boundary_thickness
    boundary_array[mask, 1] = boundary_thickness

    if center_shift:
        pixel_center_shift = (resolution//2*center_shift[0], resolution//2*center_shift[1])
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        boundary_array = boundary_array + center_shift[2]

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

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y, = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    h = (boundary_radius * resolution//2) ** 2 - (X[mask] - resolution / 2) ** 2 - (Y[mask] - resolution / 2) ** 2
    z_1 = -np.sqrt(h)*2
    z_2 = np.sqrt(h)*2
    for single_z1, single_z2 in zip(z_1, z_2):
        assert single_z1 <= single_z2, "z_1 must be less than z_2"
    # ensure 
    normalized_z_1 = z_1 / resolution
    normalized_z_2 = z_2 / resolution
    boundary_array[mask, 0] = normalized_z_1
    boundary_array[mask, 1] = normalized_z_2

    if center_shift:
        pixel_center_shift = (resolution//2*center_shift[0], resolution//2*center_shift[1])
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        boundary_array = boundary_array + center_shift[2]

    return boundary_array


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
        boundary_array = boundary_array + center_shift[2]



    return boundary_array



def add_boundary(boundaries: list) -> np.ndarray:
    added_boundary = np.zeros_like(boundaries[0])
    for boundary in boundaries:
        mask = added_boundary[:, :, 0] != added_boundary[:, :, 1]
        added_boundary[mask, 1] = np.maximum(added_boundary[mask, 1], boundary[mask, 1])
        added_boundary[mask, 0] = np.minimum(added_boundary[mask, 0], boundary[mask, 0])
        print(np.sum(mask))

        inverse_mask = np.logical_not(mask)
        added_boundary[inverse_mask, 1]  = boundary[inverse_mask, 1]
        added_boundary[inverse_mask, 0]  = boundary[inverse_mask, 0]
        print(np.sum(inverse_mask))
    return added_boundary


def subtract_boundary(boundaries: list) -> np.ndarray:
    """
    Subtract the second boundary from the first boundary,
    the second boundary must be a subset of the first boundary
    """
    assert len(boundaries) == 2, "Only two boundaries can be subtracted"
    assert np.min(boundaries[0][:, :, 0]) <= np.min(boundaries[1][:, :, 0]) and\
          np.max(boundaries[0][:, :, 1]) >= np.max(boundaries[1][:, :, 1]), \
        "The second boundary must be a subset of the first boundary"
    subtracted_boundary = deepcopy(boundaries[0])
    # select the mask where the subtract boundary is defined
    small_mask = boundaries[1][:, :, 0] != boundaries[1][:, :, 1]
    big_mask = boundaries[0][:, :, 0] != boundaries[0][:, :, 1]
    mutual_mask = np.logical_and(small_mask, big_mask)
    overlap_mask = np.logical_and(mutual_mask, small_mask)
    between_mask = np.logical_and(overlap_mask, np.logical_not(small_mask))

    # if the subtracted boundary is at the bottom
    if np.max(boundaries[1][mutual_mask, 1]) < np.max(boundaries[0][mutual_mask, 1]): 
        subtracted_boundary[mutual_mask, 0] = boundaries[1][mutual_mask, 1]
    else: # if the subtracted boundary is at the top
        subtracted_boundary[mutual_mask, 1] = boundaries[1][mutual_mask, 0]
    if np.sum(between_mask) > 0:
        subtracted_boundary[between_mask, 1] = boundaries[1][between_mask, 0]
    
    filter_mask = np.logical_and(subtracted_boundary[:, :, 0] == \
                                 subtracted_boundary[:, :, 1], subtracted_boundary[:, :, 0] != 0)
    subtracted_boundary[filter_mask] = 0

    return subtracted_boundary



def combine_boundary(boundaries: list,
                     include_all: bool=False,
                     use_subtraction: bool=False) -> np.ndarray:
    
    if include_all:
        combined_boundary = add_boundary(boundaries)
     

    if use_subtraction:
        for i in range(len(boundaries)-1):
            boundaries[len(boundaries)-i-2] = \
                subtract_boundary([combined_boundary,
                                   boundaries[len(boundaries)-i-1]])
       
    boundaries.append(combined_boundary)
    combined_boundary = np.array(boundaries)
    assert(combined_boundary.shape[0] == len(boundaries))
    assert(len(combined_boundary.shape) == 4)

    return combined_boundary


if __name__ == "__main__":
    save_dir = "bounds"
    resolution = 1024
    # boundary_width = 0.4
    # boundary_height = 0.3
    # boundary_thickness = 0.5
    # rectangle_boundary = create_rectangle_boundary(resolution, boundary_width, boundary_height, boundary_thickness)
    # print(rectangle_boundary.shape)
    # np.save(os.path.join(save_dir, "rectangle_boundary.npy"), rectangle_boundary)

    # boundary_radius = 0.4
    # cylinder_boundary = create_cylinder_boundary(resolution, boundary_radius, boundary_thickness)
    # print(cylinder_boundary.shape)
    # np.save(os.path.join(save_dir, "cylinder_boundary.npy"), cylinder_boundary)

    # sphere_boundary = create_sphere_boundary(resolution, boundary_radius)
    # print(sphere_boundary.shape)
    # np.save(os.path.join(save_dir, "sphere_boundary.npy"), sphere_boundary)

    # boundary_height = 0.8
    # boundary_radius = 0.7
    # cone_boundary = create_cone_boundary(resolution, boundary_radius, boundary_height)
    # print(cone_boundary.shape)
    # np.save(os.path.join(save_dir, "cone_boundary.npy"), cone_boundary)

    rectangle_width = 0.15 # lateral view dimension
    rectangle_height = 0.3 # front view dimension
    rectangle_thickness = 0.1 # z dimension
    rectangle_shift = (0, 0, 0.6)
    rectangle_boundary = create_rectangle_boundary(resolution,
                                                   rectangle_width,
                                                   rectangle_height,
                                                   rectangle_thickness,
                                                   rectangle_shift)

    # cone_radius = 0.3
    # cone_height = 0.3
    # cone_center_shift_1 = (0, 0, 0.15)

    ball_radius = 0.4
    ball_center_shift = (0, 0, -0)
    ball_boundary = create_sphere_boundary(resolution, ball_radius, ball_center_shift)

    # cone_center_shift_2 = (0, 0, -0.6)
    # cone_boundary_2 = create_cone_boundary(resolution, cone_radius, cone_height, cone_center_shift_2, flip=True)

    combined_boundary = combine_boundary([rectangle_boundary, ball_boundary], include_all=True, use_subtraction=True)
    print(combined_boundary.shape)
    # print(np.sum(combined_boundary - cone_boundary_1))
    np.save(os.path.join(save_dir, "combined_boundary.npy"), combined_boundary)
    for i in range(combined_boundary.shape[0]):
        print(np.min(combined_boundary[i, :, :, 0][combined_boundary[i, :, :, 0]!=0]), np.max(combined_boundary[i, :, :, 1][combined_boundary[i, :, :, 1]!=0]))




    