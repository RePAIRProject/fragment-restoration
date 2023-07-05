#-------------------------------------------------------------------------------
# EXAMPLER-BASED INPAINTING METHOD [inpainting_Criminisi et al. 2004] USED IN
# P. Harel and O. Ben-Shahar, Crossing cuts polygonal puzzles: Models and Solvers ,
# In the Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), June 2021.
#-------------------------------------------------------------------------------
# pylint: disable=missing-docstring,invalid-name
import time
import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

def inpaint(image, mask, patch_size=9):
    """ Compute the new image and return it """

    height, width = image.shape[:2]

    confidence = (1 - mask).astype(float)
    data = np.zeros([height, width])

    working_image = np.copy(image)
    working_mask = np.copy(mask.round().astype('uint8'))


    start_time = time.time()
    keep_going = True
    while keep_going:
        front = (laplace(working_mask) > 0).astype('uint8')

        confidence = update_confidence(
            confidence,
            patch_size,
            front,
            (height, width))

        data = update_data(working_image, working_mask, patch_size, front)
        priority = confidence * data * front

        target_pixel = np.unravel_index(priority.argmax(), priority.shape)

        find_start_time = time.time()
        source_patch = find_source_patch(working_image,
                                         working_mask,
                                         target_pixel,
                                         patch_size)

        #print('Time to find best: %f seconds' % (time.time()-find_start_time))

        update_image(working_image,
                     working_mask,
                     confidence,
                     target_pixel,
                     source_patch,
                     patch_size)

        remaining = working_mask.sum()
        total = height * width

        keep_going = remaining != 0

    return working_image

def find_source_patch(working_image, working_mask, target_pixel, patch_size):

    working_rgb_mask = to_rgb(working_mask)

    target_patch = get_patch(target_pixel, patch_size, working_image.shape)

    (t_i1, t_i2), (t_j1, t_j2) = target_patch
    patch_height, patch_width = ((t_i2 - t_i1), (t_j2 - t_j1))


    lab_image = rgb2lab(working_image)

    ker = np.ones((patch_height, patch_width))
    visit_mask = convolve2d(working_mask, ker, mode='valid')
    to_visit = np.argwhere(visit_mask == 0)


    rgb_mask = 1 - working_rgb_mask[t_i1:t_i2, t_j1:t_j2]

    target_data = lab_image[t_i1:t_i2, t_j1:t_j2] * rgb_mask


    height, width = lab_image.shape[:2]

    YY, XX = np.mgrid[:height - patch_height + 1, :width - patch_width + 1]
    YY_sub, XX_sub = np.mgrid[:patch_height, :patch_width]

    YY_ind = YY[:, :, None, None] + YY_sub
    XX_ind = XX[:, :, None, None] + XX_sub
    source_data = lab_image[YY_ind, XX_ind] * rgb_mask

    diffs = np.sum((source_data - target_data) **2, axis=(2, 3, 4))

    ret_i1, ret_j1 = to_visit[
        np.argmin(diffs[to_visit[:, 0], to_visit[:, 1]])]

    best_match = [
        [ret_i1, ret_i1 + patch_height],
        [ret_j1, ret_j1 + patch_width]]

    return best_match

def update_image(
        working_image,
        working_mask,
        confidence,
        target_pixel,
        source_patch,
        patch_size):
    target_patch = get_patch(target_pixel, patch_size, working_image.shape)

    (s_i1, s_i2), (s_j1, s_j2) = source_patch
    (t_i1, t_i2), (t_j1, t_j2) = target_patch

    pixels_positions = (np.argwhere(working_mask[t_i1: t_i2, t_j1:t_j2] == 1) +
                        [t_i1, t_j1])

    patch_confidence = confidence[target_pixel[0], target_pixel[1]]

    confidence[pixels_positions[:, 0], pixels_positions[:, 1]] = patch_confidence

    mask = working_mask[t_i1: t_i2, t_j1:t_j2]

    rgb_mask = to_rgb(mask)
    source_data = working_image[s_i1:s_i2, s_j1:s_j2]
    target_data = working_image[t_i1:t_i2, t_j1:t_j2]

    new_data = source_data*rgb_mask + target_data*(1-rgb_mask)


    (i1, i2), (j1, j2) = target_patch

    working_image[i1:i2, j1:j2] = new_data
    working_mask[i1:i2, j1:j2] = 0

def to_rgb(image):
    height, width = image.shape
    return image.reshape(height, width, 1).repeat(3, axis=2)

def get_patch(point, patch_size, shape):
    half_patch_size = (patch_size-1)//2
    height, width = shape[:2]
    patch = [
        [
            max(0, point[0] - half_patch_size),
            min(point[0] + half_patch_size, height-1) + 1
        ],
        [
            max(0, point[1] - half_patch_size),
            min(point[1] + half_patch_size, width-1) + 1
        ]
    ]
    return patch

def update_confidence(confidence, patch_size, front, shape):
    new_confidence = np.copy(confidence)
    front_positions = np.argwhere(front == 1)
    for point in front_positions:
        patch = get_patch(point, patch_size, shape)
        (i1, i2), (j1, j2) = patch
        patch_area = ((i2 - i1)) * (j2 - j1)

        new_confidence[point[0], point[1]] = (
            np.sum(confidence[i1:i2, j1:j2])/patch_area)
    return new_confidence

def update_data(working_image, working_mask, patch_size, front):
    normal = calc_normal_matrix(working_mask)
    gradient = calc_gradient_matrix(
        working_image,
        working_mask,
        patch_size,
        front)

    normal_gradient = normal * gradient
    data = np.sqrt(
        normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2
    ) + 0.001  # To be sure to have a greater than 0 data
    return data

def calc_normal_matrix(working_mask):
    x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
    y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

    x_normal = convolve(working_mask.astype(float), x_kernel)
    y_normal = convolve(working_mask.astype(float), y_kernel)
    normal = np.dstack((x_normal, y_normal))

    height, width = normal.shape[:2]
    norm = (np
            .sqrt(y_normal**2 + x_normal**2)
            .reshape(height, width, 1)
            .repeat(2, axis=2))
    norm[norm == 0] = 1

    unit_normal = normal/norm
    return unit_normal

def calc_gradient_matrix(working_image, working_mask, patch_size, front):
    height, width = working_image.shape[:2]

    grey_image = rgb2gray(working_image)
    '''if len(working_image.shape) == 3:
        grey_image = rgb2gray(working_image)
    else:
        grey_image = working_image'''
    grey_image[working_mask == 1] = None

    gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
    gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
    max_gradient = np.zeros([height, width, 2])

    front_positions = np.argwhere(front == 1)
    for point in front_positions:
        patch = get_patch(point, patch_size, working_image.shape)
        (i1, i2), (j1, j2) = patch

        (patch_y_gradient, patch_x_gradient, patch_gradient_val) = (
            mat[i1:i2, j1:j2]
            for mat in (
                gradient[0], gradient[1], gradient_val))

        patch_max_pos = np.unravel_index(
            patch_gradient_val.argmax(),
            patch_gradient_val.shape
        )

        max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]
        max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]

    return max_gradient
