
# These are the functions written by Aref

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from PIL import Image


import paths_inpainting



def sort_key(s):
    # Extract numeric part from the string
    numeric_part = re.findall(r'\d+', s)

    # If numeric part is found, return it as integer, else print the filename and return 0
    if numeric_part:
        return int(numeric_part[0])
    else:
        print(f"No numeric part in filename: {s}")
        return 0

#Check if masks correctly overlapped with correspondant images
'''def visualize_random_image(RGB_images):
    # Generate a random index
    index = random.choice(range(len(RGB_images)))

    # Add weighted images to create a single image with bounding boxes
    dst = cv2.addWeighted(RGB_images[index], 0.5, mask_images_3channels[index], 0.5, 0)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(dst)
    plt.axis('off')
    plt.show()'''

def make_background_white(image, foreground_mask):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #_, foreground_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    background_mask = cv2.bitwise_not(foreground_mask)
    image_with_white_background = image.copy()
    image_with_white_background[np.where(background_mask == 255)] = [255, 255, 255]

    white_background_image = Image.fromarray(image_with_white_background)
    return white_background_image

def make_background_black(image, fg, edge_threshold=100):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, edge_threshold, edge_threshold * 3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    # Fill the largest contour (assumed to be the background) with white
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    image_with_black_background = image.copy()
    image_with_black_background[mask == 0] = [0, 0, 0]

    image_with_black_background = cv2.cvtColor(image_with_black_background, cv2.COLOR_RGB2BGR)
    black_background_image = Image.fromarray(image_with_black_background)

    return black_background_image

def keep_white_pixels(image, lower_threshold=(200, 200, 200), upper_threshold=(255, 255, 255)):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    mask = cv2.inRange(image, lower_threshold, upper_threshold)

    black_image = np.zeros_like(image)
    black_image[mask == 255] = image[mask == 255]
    gray_image = cv2.cvtColor(black_image, cv2.COLOR_RGB2GRAY)
    white_pixels_image = Image.fromarray(gray_image)

    return white_pixels_image

def make_background_black_updated(image, edge_threshold=100, frame_size=10):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Add frame around the image
    height, width, _ = image.shape
    framed_image = cv2.copyMakeBorder(image, frame_size, frame_size, frame_size, frame_size, cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])

    gray_image = cv2.cvtColor(framed_image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, edge_threshold, edge_threshold * 3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    # Fill the largest contour (assumed to be the background) with white
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    framed_image_with_black_background = framed_image.copy()
    framed_image_with_black_background[mask == 0] = [0, 0, 0]

    # Remove the frame
    image_with_black_background = framed_image_with_black_background[frame_size:-frame_size, frame_size:-frame_size]

    image_with_black_background = cv2.cvtColor(image_with_black_background, cv2.COLOR_RGB2BGR)
    black_background_image = Image.fromarray(image_with_black_background)

    return black_background_image

def make_background_black_updated_3(image):
    edge_threshold = 100
    frame_size = 10
    min_contour_area = 500
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Add frame around the image
    height, width, _ = image.shape
    framed_image = cv2.copyMakeBorder(image, frame_size, frame_size, frame_size, frame_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    gray_image = cv2.cvtColor(framed_image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, edge_threshold, edge_threshold * 3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    # Fill the contours with white, but only if they are large enough
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], 0, 255, -1)

    framed_image_with_black_background = framed_image.copy()
    # Apply the mask to the color image
    for i in range(3):  # For each color channel
        framed_image_with_black_background[:, :, i][mask == 0] = 0

    # Remove the frame
    image_with_black_background = framed_image_with_black_background[frame_size:-frame_size, frame_size:-frame_size]

    return image_with_black_background
