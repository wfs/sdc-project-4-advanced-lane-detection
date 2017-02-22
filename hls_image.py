import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# used in Jupyter notebook from video
# %matplotlib inline
# %matplotlib qt

class IntenseColoursMask:
    """ Represents most intense colours as a binary mask. """
    # Read in an image, you can also try test1.jpg or test4.jpg
    TEST_IMAGE = mpimg.imread('bridge_shadow.jpg')

    @staticmethod
    def hls_select(img, thresh=(0, 255)):
        """

        :param img:
        :param thresh:
        :return:
        """
        # Define a function that thresholds the S-channel of HLS
        # Use exclusive lower bound (>) and inclusive upper (<=)
        # Define a function that thresholds the S-channel of HLS

        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # 2) Apply a threshold to the S channel
        # s_channel = hls[:, :, 2]  # GLITCH IN VIDEO CODE !!!
        s_channel = hls[:, :, -1]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        print(binary_output)

        # 3) Return a binary image of threshold result
        # binary_output = np.copy(img)  # placeholder line
        return binary_output

    @staticmethod
    def view_hls_binary_mask(image, thresh):
        # # Plot the result
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(image)
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(hls_binary, cmap='gray')
        # ax2.set_title('Thresholded S', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        binary_image = IntenseColoursMask.hls_select(image, thresh)

        cv2.imshow('binary_image', binary_image)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

    @staticmethod
    def view_hls_test():
        test_image = IntenseColoursMask.TEST_IMAGE
        tst_img = cv2.cvtColor(test_image, cv2.COLOR_RGB2HLS)
        cv2.imshow('test_image', test_image)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
