import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from imageio import imread, imsave, imwrite

sigma_s = 3
sigma_r = 30


def geometric_distance(x, y, i, j):
    return math.sqrt(np.power((x - i), 2) + np.power((y - j), 2))

def photometric_distance(original_image, x, y, i, j):
    original_value = original_image[x, y]
    compared_value = original_image[i, j]

    return abs(int(original_value) - int(compared_value))

def pixel_weight(original_image, x, y, i, j):

    return geometric_gaussian(geometric_distance(x, y, i, j)) * photometric_gaussian(photometric_distance(original_image, x, y, i, j))

def geometric_gaussian(distance):

    return (1 / (sigma_s * math.sqrt(2 * math.pi))) * np.exp(-np.power(distance, 2) / (2 * np.power(sigma_s, 2)))

def photometric_gaussian(distance):

    return (1 / (sigma_r * math.sqrt(2 * math.pi))) * math.exp(-np.power(distance, 2) / (2 * np.power(sigma_r, 2)))

def calc_new_pixel_value(x, y, height, width, original_image, box_size):

    normalization_factor = 0
    new_pixel_value = 0
    d = round(box_size / 2)

    for i in range(x - d, x + d):
        for j in range(y - d, y + d):

            if (i >= 0 and i < height and j >= 0 and j < width):

                weight = pixel_weight(original_image, x, y, i, j)

                neighbor_pixel = original_image[i, j]

                new_pixel_value += neighbor_pixel * weight

                normalization_factor += weight

    new_pixel_value = new_pixel_value / normalization_factor

    return new_pixel_value

def bilateral_filter(original_image):

    height, width = original_image.shape
    filtered_image = np.empty(original_image.shape)  # initialize empty photo

    size_of_box = 3

    for x in range(height):
        for y in range(width):
            new_pixel_value = calc_new_pixel_value(x, y, height, width, original_image, size_of_box)
            filtered_image[x,y] = int(round(new_pixel_value))

    return filtered_image

def color_bilateral_filter(original_image):

    r, g, b = [original_image[:,:,i] for i in range(3)]

    filtered_image = np.empty(original_image.shape)

    filtered_image[:,:,0] = bilateral_filter(r)
    filtered_image[:,:,1] = bilateral_filter(g)
    filtered_image[:,:,2] = bilateral_filter(b)

    return filtered_image

if __name__ == '__main__':

    #input_name = str(sys.argv[1])
   # output_name = sys.argv[2] + '.jpeg'

    original_image = imread('in_img.jpg')

    if original_image.shape[2] == 3:                    # RBG image
        print("RGB")
        color_filtered_image = color_bilateral_filter(original_image)
        print('yea')
        imwrite('bew.png', color_filtered_image.astype(np.uint8))

    else:                                               #black and white image
        print("black and white")
        BW_filtered_image = bilateral_filter(original_image)
        imwrite('bew.png', BW_filtered_image)
