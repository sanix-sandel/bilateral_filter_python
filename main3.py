import numpy as np
import cv2
import sys
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1 / (2 * np.pi * (sigma ** 2))) * np.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    source = np.zeros(source.shape)

    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
           # print('neighbour_x', np.int(np.abs(neighbour_x)))
           # print('neighbour_y', np.int(np.abs(neighbour_y)))
           # print('x', x)
           # print('y', y)
            source[int(neighbour_x)][int(neighbour_y)]
           # print(source[int(neighbour_x)][int(neighbour_y)])
           # print(source[x][y])
            gi = gaussian(source[np.int(neighbour_x)][np.int(neighbour_y)] - source[x][y], np.int(sigma_i))

            #print('gi', gi)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[np.int(neighbour_x)][np.int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered // Wp
    filtered_image[x][y] = (np.round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":
    image = cv2.imread("lions.jpg")

    width, heigth = image.shape[:2]
    final_wide = int(width / 4)
    r = float(final_wide) / image.shape[1]
    dim = (final_wide, int(image.shape[0] * r))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    filtered_image_OpenCV = cv2.bilateralFilter(image, 5, 12, 16)
    cv2.imwrite("original_image_grayscale.png", image)
    cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    filtered_image_own = bilateral_filter_own(image, 5, 12, 16)
    cv2.imwrite("filtered_image_own.png", filtered_image_own)







