""" Imports """
import cv2
import numpy as np
import matplotlib.pyplot as plt
# uncomment next line when running code in jupyter notebook
# %matplotlib inline
import glob
import matplotlib.cm as cm


# ------------------
class Points:
    """ Static class """
    objpoints = []
    imgpoints = []


# ------------------
def find_corner_coordinates_and_map_to_reference_chessboard():
    """
    Create 9 col x 6 row grid reference coordinates aka 3D objpoints.
    Locate corners surrounded by 4 squares in calibration images and store coordinates aka 2D imgpoints
    """
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    images = glob.glob('camera_cal/calibration*.jpg')
    for img in images:
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            Points.imgpoints.append(corners)
            Points.objpoints.append(objp)
    print("Grid setup done.")


# ------------------
def calculate_camera_distortion_coefficients(img, objpts, imgpts):
    """
    aka calibrate camera
    :param img: to calculate against
    :param objpts: 3D grid coordinates from Points class
    :param imgpts: 2D image corner coordinates from Points class
    :return: camera_matrix, distortion_coefficients
    """
    img_size = (img.shape[1], img.shape[0])
    return_value, camera_matrix, distortion_coefficients, rotation_vectors, \
    translation_vectors = cv2.calibrateCamera(objpts, imgpts, img_size, None, None)
    print("Distortion coefficients calculated.")
    return camera_matrix, distortion_coefficients


# ------------------
def undistort_image(img, mtx, dist):
    """
    Undistort the image.
    :param img: image
    :param mtx: camera attributes matrix
    :param dist: distortion coefficients
    :return: undistorted image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    print("Image undistorted.")
    return undist


# ------------------
def compare_raw_distorted_against_undistorted_images(raw_1, undist_1, raw_2, undist_2):
    """
    Side-by-side visual comparison test.
    :param raw_1: distorted image
    :param raw_2: 2nd distorted image
    :param undist_1: undistorted image
    :param undist_2: 2nd undistorted image
    """
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(15)

    fig.add_subplot(2, 2, 1)
    plt.title("Distorted Images")
    plt.imshow(raw_1, cmap='gray')
    fig.add_subplot(2, 2, 2)
    plt.title("Undistorted Images")
    plt.imshow(undist_1, cmap='gray')
    fig.add_subplot(2, 2, 3)
    plt.imshow(raw_2)
    fig.add_subplot(2, 2, 4)
    plt.imshow(undist_2)

    # IMPORTANT : uncomment when running from Jupyter Notebook
    plt.show()

    print("Side-by-side views done.")


# ------------------
""" 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. """
find_corner_coordinates_and_map_to_reference_chessboard()

calibration_img = cv2.imread("camera_cal/calibration1.jpg")
c_matrix, dist_coeff = calculate_camera_distortion_coefficients(calibration_img, Points.objpoints,
                                                                Points.imgpoints)

# ------------------
""" 2. Apply a distortion correction to raw images. """
# calibration images
undistorted_calibration_img = undistort_image(calibration_img, c_matrix, dist_coeff)
cv2.imwrite('camera_cal/undist_calibration1.jpg', undistorted_calibration_img)

# straight lane line images
straight_lines_image_2 = cv2.imread('test_images/straight_lines2.jpg')
undistorted_straight_lines_2 = undistort_image(straight_lines_image_2, c_matrix, dist_coeff)
cv2.imwrite('output_images/undistorted_straight_lines2.jpg', undistorted_straight_lines_2)

# View results
compare_raw_distorted_against_undistorted_images(calibration_img, undistorted_calibration_img,
                                                 straight_lines_image_2,
                                                 undistorted_straight_lines_2)

# ------------------
""" 3. Use color transforms, gradients, etc., to create a thresholded binary image. """


def extract_saturation_channel(colour_img):
    """
    Convert colour image into HLS channels and extract saturation channel.
    :param colour_img: image to extract s channel ndarray from
    :return: image saturation channel ndarray
    """
    hls = cv2.cvtColor(colour_img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]  # aka 3rd channel
    return s_channel


def sobel_gradient_direction(copy_of_colour_image, sobel_kernel=3):
    """
     Define a function that applies Sobel x and y, then computes the direction (radians) of the gradient.
    :param copy_of_colour_image: to process
    :param sobel_kernel: window size for calculating x and y gradients
    :return:
    """
    gray = cv2.cvtColor(copy_of_colour_image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))  # radians aka approx. angles
    return abs_grad_dir


def view_transformed_images(gradient_saturation_binary_image, combined_saturation_gradient_colour_binary_image):
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(15)

    fig.add_subplot(2, 2, 1)
    plt.title("Gradient Saturation Binary")
    plt.imshow(gradient_saturation_binary_image)
    fig.add_subplot(2, 2, 2)
    plt.title("Gradient Saturation Colour Binary")
    plt.imshow(combined_saturation_gradient_colour_binary_image, cmap='gray')

    # IMPORTANT : uncomment when running from Jupyter Notebook
    plt.show()

    print("Binary side-by-side views done.")


def colour_image_transformation_pipeline(colour_image, s_thresh=(180, 255), sobel_threshold_range=(40, 100),
                                         rgb_thresh=(200, 255)):
    copy_of_colour_image = np.copy(colour_image)

    # Extract S channel
    saturation_channel = extract_saturation_channel(copy_of_colour_image)

    # Calculate line directions
    sobel_gradient_directions = sobel_gradient_direction(copy_of_colour_image)

    # Threshold gradient
    sobel_binary = np.zeros_like(sobel_gradient_directions)
    sobel_binary[(sobel_gradient_directions >= sobel_threshold_range[0]) & (
        sobel_gradient_directions <= sobel_threshold_range[1])] = 1

    # Threshold RGB channel for range (200, 255) aka yellows
    yellow = copy_of_colour_image[:, :, 0]
    yellow_binary = np.zeros_like(yellow)
    yellow_binary[(yellow > rgb_thresh[0]) & (yellow <= rgb_thresh[1])] = 1

    # Threshold colour intensity
    saturation_binary = np.zeros_like(saturation_channel)
    saturation_binary[(saturation_channel >= s_thresh[0]) & (saturation_channel <= s_thresh[1])] = 1

    # Stack dimensions
    grad_sat_bin_image = np.dstack((np.zeros_like(sobel_binary), sobel_binary, saturation_binary))
    stacked_sat_grad_colour_bin_image = np.zeros_like(sobel_binary)
    stacked_sat_grad_colour_bin_image[(saturation_binary == 1) | (sobel_binary == 1) | (yellow_binary == 1)] = 1
    print("colour_image_transformation_pipeline processing done")
    return grad_sat_bin_image, stacked_sat_grad_colour_bin_image


grad_sat_bin_image, stacked_sat_grad_colour_bin_image = colour_image_transformation_pipeline(
    undistorted_straight_lines_2)
plt.imsave('output_images/colour_binary_straight_lines2.png', np.array(grad_sat_bin_image))
plt.imsave('output_images/stacked_binary_straight_lines2.png', np.array(stacked_sat_grad_colour_bin_image),
           cmap=cm.gray)

view_transformed_images(grad_sat_bin_image, stacked_sat_grad_colour_bin_image)

# ------------------

""" 4. Apply a perspective transform to rectify binary image ("birds-eye view"). """

# ------------------

""" 5. Detect lane pixels and fit to find the lane boundary. """

# ------------------
""" 6. Determine the curvature of the lane and vehicle position with respect to center. """

# ------------------

""" 7. Warp the detected lane boundaries back onto the original image. """

# ------------------

""" 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. """

# ------------------
