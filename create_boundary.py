import numpy as np
import os
from typing import Tuple
from copy import deepcopy
import argparse
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
        pixel_center_shift = (int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]))
        # make sure the shift is not out of boundary

        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        mask = np.logical_and(boundary_array[:, :, 0] != 0, boundary_array[:, :, 1] != 0)
        boundary_array[mask] = boundary_array[mask] + center_shift[2]

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
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= \
        (boundary_radius * resolution//2) ** 2
    boundary_array[mask, 0] = -boundary_thickness
    boundary_array[mask, 1] = boundary_thickness

    if center_shift:
        pixel_center_shift = (int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]))
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        mask = np.logical_and(boundary_array[:, :, 0] != 0, boundary_array[:, :, 1] != 0)
        boundary_array[mask] = boundary_array[mask] + center_shift[2]

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
        pixel_center_shift = (int(resolution//2*center_shift[0]), int(resolution//2*center_shift[1]))
        boundary_array = np.roll(boundary_array, pixel_center_shift, axis=(0,1))
        mask = np.logical_and(boundary_array[:, :, 0] != 0, boundary_array[:, :, 1] != 0)
        boundary_array[mask] = boundary_array[mask] + center_shift[2]

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
        mask = np.logical_and(boundary_array[:, :, 0] != 0, boundary_array[:, :, 1] != 0)
        boundary_array[mask] = boundary_array[mask] + center_shift[2]
    return boundary_array

def add_boundary(boundaries: list) -> np.ndarray:
    added_boundary = boundaries[0]
    added_mask = np.logical_and(boundaries[0][:, :, 0] != 0, boundaries[0][:, :, 1] != 0)
    for idx, boundary in enumerate(boundaries):
        # get the mask for the current boundary
        mask = np.logical_and(boundary[:, :, 0] != 0, boundary[:, :, 1] != 0)
        mutual_mask = np.logical_and(mask, added_mask)
        
        cv2.imwrite(f"mask_{idx}.png", mask.astype(np.uint8)*255)
        cv2.imwrite(f"mutual_mask_{idx}.png", mutual_mask.astype(np.uint8)*255)
   
        added_boundary[mutual_mask, 1] = np.maximum(added_boundary[mutual_mask, 1],
                                                    boundary[mutual_mask, 1])
        added_boundary[mutual_mask, 0] = np.minimum(added_boundary[mutual_mask, 0],
                                                    boundary[mutual_mask, 0])
        # print(np.sum(mask))
        inverse_mask = np.logical_not(added_mask)

        # save the inverse mask as png
        cv2.imwrite(f"inverse_mask_{idx}.png", inverse_mask.astype(np.uint8)*255)

        added_boundary[inverse_mask, 1]  = boundary[inverse_mask, 1]
        added_boundary[inverse_mask, 0]  = boundary[inverse_mask, 0]
        cv2.imwrite(f"added_mask_{idx}.png", added_mask.astype(np.uint8)*255)
        added_mask = np.logical_or(added_mask, mask)
        # print(np.sum(inverse_mask))
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
                     use_subtraction: bool=False,
                     igore_idx: list=None) -> np.ndarray:
    

    combined_boundary = add_boundary(boundaries)
    sub_combined_boundary_list = []
    for i in range(len(boundaries)-1):
        # add boundaries except for the ith boundary
        sub_combined_boundary = add_boundary(boundaries[:i] + boundaries[i+1:])
        sub_combined_boundary_list.append(sub_combined_boundary)
    if use_subtraction:
        for i in range(len(boundaries)-1):
            if igore_idx[i]:
                continue
            boundaries[i] = subtract_boundary(
                [combined_boundary, sub_combined_boundary_list[i]])
    if include_all:
        boundaries.append(combined_boundary)

    combined_boundary = np.array(boundaries)
    assert(combined_boundary.shape[0] == len(boundaries))
    assert(len(combined_boundary.shape) == 4), \
        "The combined boundary must be a 4D array"

    return combined_boundary


if __name__ == "__main__":
    save_dir = "bounds"
    resolution = 1024

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
    ball_center_shift = (0, 0, 0.2)
    ball_boundary = create_sphere_boundary(resolution, ball_radius, ball_center_shift)
    # rectangle_width = 0.2 # lateral view dimension
    # rectangle_height = 0.2 # front view dimension
    # rectangle_thickness = 0.21 # z dimension
    # rectangle_shift = (0, 0, 0.2)
    # rectangle_boundary_2 = create_rectangle_boundary(resolution,
    #                                                 rectangle_width,
    #                                                 rectangle_height,
    #                                                 rectangle_thickness,
    #                                                 rectangle_shift)

    rectangle_width = 0.3 # lateral view dimension
    rectangle_height = 0.2 # front view dimension
    rectangle_thickness = 0.4 # z dimension
    rectangle_shift = (0, 0, -0.399)
    rectangle_boundary_3 = create_rectangle_boundary(resolution,
                                                    rectangle_width,
                                                    rectangle_height,
                                                    rectangle_thickness,
                                                    rectangle_shift)

    # combined_boundary = combine_boundary([ball_boundary, rectangle_boundary_3],
    #                                      include_all=True,
    #                                      use_subtraction=True,
    #                                      igore_idx=[True, False, False])

    combined_boundary = combine_boundary([ball_boundary, rectangle_boundary_3, rectangle_boundary_1],
                                         include_all=True,
                                         use_subtraction=True,
                                         igore_idx=[True, False, False])
    print(combined_boundary.shape)
    mask = np.logical_and(combined_boundary[-1, :, :, 0] != 0 , combined_boundary[-1, :, :, 1] != 0)
    cv2.imwrite("final_mask.png", mask.astype(np.uint8)*255)
    np.save(os.path.join(save_dir, "combined_boundary_3body.npy"), combined_boundary)
    for i in range(combined_boundary.shape[0]):
        print(np.min(combined_boundary[i, :, :, 0][combined_boundary[i, :, :, 0]!=0]),
              np.max(combined_boundary[i, :, :, 1][combined_boundary[i, :, :, 1]!=0]))




    