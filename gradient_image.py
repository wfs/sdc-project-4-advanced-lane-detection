import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pickle
#%matplotlib inline


class BinaryMask:
    """ Static class applies a threshold range to filter Sobel x, y gradient directions and generate binary image. """
    # Read in an image
    TEST_IMAGE = mpimg.imread('signs_vehicles_xygrad.png')

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """
        Define a function that applies Sobel x and y, then computes the direction of the gradient
        and applies a threshold.

        :param img: to generate binary mask from
        :param sobel_kernel: window size for calculating x and y gradients
        :param thresh: range of radians that we want to include
        :return: binary image mask
        """
        # NOTE : thresh = (0.7, 1.3) will result in this function roughly identifying the
        # lane lines but with lots of noise

        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Calculate the gradient in x and y separately
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobel_y, abs_sobel_x) to calculate the direction of the gradient
        abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))  # returns radians aka approx. angles

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(abs_grad_dir)
        binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    @staticmethod
    def view_mask_test(img, sobel_kernel, thresh):
        """
        Creates binary image by applying gradient, direction, threshold range.
        Displays the binary image for visual testing of different kernel sizes and threshold ranges.
        :param img: to test params on
        :param sobel_kernel: window size
        :param thresh: gradient direction range
        """
        bin_mask = BinaryMask.dir_threshold(img, sobel_kernel, thresh)

        # # Plot the result
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(bm.TEST_IMAGE)
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(dir_binary, cmap='gray')
        # ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        cv2.imshow('bin_mask', bin_mask)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
