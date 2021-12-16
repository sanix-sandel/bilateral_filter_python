import cv2
import numpy

numpy.seterr(over='ignore')

def f(x):
    return numpy.int(x)

def gaussian(x,sigma):
    print('x', x)
    return (1.0/(2*numpy.pi*(sigma**2))) * numpy.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s):
    print('function called')


    print(image.shape)


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
            new_image[row][col] = numpy.round(filtered_image)
            new_image=new_image.astype(int)
    print('function ended')
    return new_image


image = cv2.imread("lions.jpg", cv2.IMREAD_COLOR)
width, heigth = image.shape[:2]
final_wide = int(width / 4)
r = float(final_wide) / image.shape[1]
dim = (final_wide, int(image.shape[0] * r))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
print('yo1')
image_own = bilateral_filter(image, 7, 20.0, 20.0)
print('yo')
cv2.imwrite("filtered_image_own.png", image_own)
cv2.imwrite('diff.png', image_own-filtered_image_OpenCV)
print('original ',image[0])
print('image_own ',image_own[0])
print('image_open_cv ',filtered_image_OpenCV[0])

