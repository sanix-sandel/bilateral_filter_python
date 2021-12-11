import numpy
import cv2


def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):


    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises:
        ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
    """

    # check the input
   # if not isinstance( img_in, numpy.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
   #     raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (numpy.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = numpy.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = numpy.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum

if __name__=='__main__':
    image = cv2.imread("in_img.jpg")
    filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
    cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    print('yo1')
    image_own = filter_bilateral(image, 7, 20.0, 20.0)
    print('yo')
    cv2.imwrite("filtered_image_own.png", image_own)
    cv2.imwrite('diff.png', filtered_image_OpenCV-image_own)

