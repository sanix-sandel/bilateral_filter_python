import cv2
import numpy

numpy.seterr(over='ignore')

def gaussian(x,sigma):
    print(type(x))
    return (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    print('function called')
    new_image = numpy.zeros(image.shape)

    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0
            for k in range(diameter):
                for l in range(diameter):
                    n_x =row - (diameter/2 - k)
                    n_y =col - (diameter/2 - l)
                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])

                    gi = gaussian(image[int(n_x)][int(n_y)] - image[int(row)][int(col)], int(sigma_i))
                    gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                    wp = gi * gs
                    filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp
            filtered_image = filtered_image // wp_total
            new_image[row][col] = int(numpy.round(filtered_image))
    print('function ended')
    return new_image


image = cv2.imread("lions.jpg", 0)
filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
print('yo1')
image_own = bilateral_filter(image, 7, 20.0, 20.0)
print('yo')
cv2.imwrite("filtered_image_own.png", image_own)
cv2.imwrite('diff.png', filtered_image_OpenCV - image_own)
print('image ',image[0])
print('image ',image_own[0])
print('image ',filtered_image_OpenCV[0])

