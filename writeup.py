# ---
# # Advanced Lane Finding Project
#
# The goals / steps of this project are the following:
#
# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# 2. Apply a distortion correction to raw images.
# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# 5. Detect lane pixels and fit to find the lane boundary.
# 6. Determine the curvature of the lane and vehicle position with respect to center.
# 7. Warp the detected lane boundaries back onto the original image.
# 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#
# ---
""" Imports """
import cv2
import numpy as np
import matplotlib.pyplot as plt
# uncomment next line when running code in jupyter notebook
% matplotlib
inline
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
    print("Calibration grid setup done.")


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


def sobel_gradient_direction(copy_of_colour_image, sobel_kernel=3):
    """
     Define a function that applies Sobel x and y, then computes the direction (radians) of the gradient.
    :param copy_of_colour_image: to process
    :param sobel_kernel: window size for calculating x and y gradients, cv2.Sobel default = 3 if no param
    :return:
    """
    gray = cv2.cvtColor(copy_of_colour_image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # Note : from video code, couldn't get np.arctan2() to produce usable results, so simply scaling instead.
    # sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))  # radians aka approx. angles
    abs_grad_dir = np.uint8(
        255 * (np.absolute(sobel_x)) / np.max(np.absolute(sobel_x)))  # rescale down to 8 bit integer
    # print("np.absolute(sobel_x) :", np.absolute(sobel_x))
    # print("np.max(np.absolute(sobel_x) :", np.max(np.absolute(sobel_x)))
    # print("abs_grad_dir :", abs_grad_dir)
    return abs_grad_dir


def extract_saturation_channel(colour_img):
    """
    Convert colour image into HLS channels and extract saturation channel.
    :param colour_img: image to extract s channel ndarray from
    :return: image saturation channel ndarray
    """
    hls = cv2.cvtColor(colour_img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]  # aka 3rd channel
    # print("saturation channel extracted :", s_channel)
    return s_channel


def view_transformed_images(gradient_saturation_binary_image, combined_saturation_gradient_colour_binary_image):
    """
    View the images output from the transformation pipeline.
    :param gradient_saturation_binary_image: Directional, S channel, binary thresholded image.
    :param combined_saturation_gradient_colour_binary_image: S channel, Directional, Yellows, binary
        thresholded image.
    """
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


def colour_image_transformation_pipeline(colour_image, sat_thresh=(150, 255), sobel_threshold_range=(40, 100),
                                         rgb_thresh=(200, 255)):
    """
    Image information extraction pipeline.
    :param colour_image: input image
    :param sat_thresh: saturation threshold range
    :param sobel_threshold_range: gradient direction threshold range
    :param rgb_thresh: colour intensity threshold range
    :return: grad_sat_bin_image, stacked_s_g_c_bin_image
    """
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
    saturation_binary[(saturation_channel >= sat_thresh[0]) & (saturation_channel <= sat_thresh[1])] = 1

    # Stack dimensions
    grad_sat_bin_img = np.dstack((np.zeros_like(sobel_binary), sobel_binary, saturation_binary))
    stacked_s_g_c_bin_img = np.zeros_like(sobel_binary)
    stacked_s_g_c_bin_img[(saturation_binary == 1) | (sobel_binary == 1) | (yellow_binary == 1)] = 1
    print("colour_image_transformation_pipeline processing done")
    return grad_sat_bin_img, stacked_s_g_c_bin_img


# ------------------
# Extract information from image
grad_sat_bin_image, stacked_sat_grad_colour_bin_image = colour_image_transformation_pipeline(
    undistorted_straight_lines_2)

# Save outputs
plt.imsave('output_images/colour_binary_straight_lines2.png', np.array(grad_sat_bin_image))
plt.imsave('output_images/stacked_binary_straight_lines2.png', np.array(stacked_sat_grad_colour_bin_image),
           cmap=cm.gray)

# View outputs
view_transformed_images(grad_sat_bin_image, stacked_sat_grad_colour_bin_image)

# ------------------
""" 4. Apply a perspective transform to rectify binary image ("birds-eye view"). """


def view_polygon_and_warped_images(polygon_overlay_image, warped_polygon_overlay_image):
    """
    View the undistored image with polygon lane line and warped image from overhead viewpoint with
        destination points drawn.
    :param polygon_overlay_image: image with overlay
    :param warped_polygon_overlay_image: overhead view of image with overlay
    """
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(15)

    fig.add_subplot(2, 2, 1)
    plt.title("Undistorted image with polygon overlay")
    plt.imshow(polygon_overlay_image)
    fig.add_subplot(2, 2, 2)
    plt.title("Warped image from overhead viewpoint")
    plt.imshow(warped_polygon_overlay_image)

    # IMPORTANT : uncomment when running from Jupyter Notebook
    plt.show()

    print("Polygon and warped side-by-side views done.")


def define_source_polygon():
    """
    Define 4 corners of polygon region of undistorted image.
    """
    global source_transformation
    source_transformation = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    # print("source_transformation polygon defined :", source_transformation)


def define_destination_polygon():
    """
    Define 4 corners of polygon region to warp transform onto.
    """
    global destination_transformation
    destination_transformation = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    # print("destination_transformation polygon defined :", destination_transformation)


def warp_perspective_to_top_down(src, img, dst):
    """
    Calculate the inversed transformation matrix and
    :param src: source points where warp transforms from
    :param img: stacked binary thresholded image that includes saturation, gradient direction, colour intensity
    :param dst: destination points where warp transforms to
    :return: the transformation matrix inversed, warped top-down binary image
    """
    transformation_matrix = cv2.getPerspectiveTransform(src, dst)  # the transform matrix
    transformation_matrix_inverse = cv2.getPerspectiveTransform(dst, src)  # the transform matrix inverse
    warped_to_top_down = cv2.warpPerspective(img, transformation_matrix,
                                             img_size)  # warp image to a top-down view
    print("Perspective warp done.")
    return transformation_matrix_inverse, warped_to_top_down


def polygon():
    """
    Create green polygon and overlay on undistorted image.
    """
    global polygon_undistored_image
    polygon_undistored_image = cv2.line(undistorted_straight_lines_2, (240, 700), (610, 440), [0, 255, 0], 3)
    polygon_undistored_image = cv2.line(undistorted_straight_lines_2, (240, 700), (1080, 700), [0, 255, 0], 3)
    polygon_undistored_image = cv2.line(undistorted_straight_lines_2, (1080, 700), (670, 440), [0, 255, 0], 3)
    polygon_undistored_image = cv2.line(undistorted_straight_lines_2, (610, 440), (670, 440), [0, 255, 0], 3)
    # print("Green polygon defined : ", polygon_undistored_image)


# ------------------
""" 5. Detect lane pixels and fit to find the lane boundary. """
# Get image 2D size
print("undistorted_straight_lines_2.shape : ", undistorted_straight_lines_2.shape)
img_size = (undistorted_straight_lines_2.shape[1], undistorted_straight_lines_2.shape[0])

# Set 4 corners of warp source polygon
define_source_polygon()

# Set 4 corners of warp destination polygon
define_destination_polygon()

# Create polygon overlayed onto undistorted image
polygon()

# Engage warp drive :)
matrix_transform_inversed, top_down_warped_binary_polygon = warp_perspective_to_top_down(source_transformation,
                                                                                         polygon_undistored_image,
                                                                                         destination_transformation)

matrix_transform_inversed_stacked, top_down_warped_binary_stacked = warp_perspective_to_top_down(
    source_transformation,
    stacked_sat_grad_colour_bin_image,
    destination_transformation)

# Store output
cv2.imwrite('output_images/lined_image_straight_lines2.jpg', polygon_undistored_image)
cv2.imwrite('output_images/warped_straight_lines2.jpg', top_down_warped_binary_polygon)

# View output
view_polygon_and_warped_images(polygon_undistored_image, top_down_warped_binary_polygon)

# ------------------
histogram = np.sum(top_down_warped_binary_stacked[top_down_warped_binary_stacked.shape[0] // 2:, :], axis=0)
print("top_down_warped_binary_stacked.shape : ", top_down_warped_binary_stacked.shape)
plt.plot(histogram)

# ------------------
"""
6. Determine the curvature of the lane and vehicle position with respect to center.
"""


class Line:
    def __init__(self):
        """
        A class to receive the characteristics of each line detection

        A Udacity provided class definition.
        """
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_x_fitted = []
        # average x values of the fitted line over the last n iterations
        self.best_x = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.all_x = None
        # y values for detected line pixels
        self.all_y = None
        # unable to detect left / right lanes
        self.lost_sight_of_lanes_count = None


# ------------------
def get_left_mid_right_pts_from_warp_histogram(top_down_bin_warp):
    """
    Located the peaks in the histogram and calculate the mid-lane point.
    :param top_down_bin_warp: image with polygon overlay of lanes
    :return: left_x_point, midpoint, right_x_point
    """
    histogram = np.sum(top_down_bin_warp[top_down_bin_warp.shape[0] // 2:, :], axis=0)  # 360
    # print("histogram :", histogram)
    midpoint = np.int(histogram.shape[0] / 2)  # 180
    print("midpoint :", midpoint)
    left_x_point = np.argmax(histogram[:midpoint])
    print("left_x_point :", left_x_point)
    right_x_point = np.argmax(histogram[midpoint:]) + midpoint
    print("right_x_point :", right_x_point)
    return left_x_point, midpoint, right_x_point


# ------------------
def get_offset_from_center(right_x, left_x, midpoint):
    """
    Using the warp histogram x points, calculate the camera's (aka car's) position, relative to lane centre.
    Note 1 : minimum USA lane width for interstate highways is 3.7 metres.
    See https://en.wikipedia.org/wiki/Interstate_Highway_standards
    Note 2 : image 2D is 780 (y) x 1280 (x) pixels and  lane is approx. 800 (x) pixels wide at base
    :param right_x: right lane spike max x value
    :param left_x: right lane spike max x value
    :param midpoint: calculated mid-point value between left_x and right_x
    :return: car offset from lane centre
    """

    return (3.7 / 700) * abs(((right_x - left_x) / 2) + left_x - midpoint)  # 3.7 m / 700 px == metres per pixel


# ------------------
class SlidingWindow:
    """ Sliding windows that track the lane curvature. """

    def __init__(self):
        self.left_x_pt = 0
        self.right_x_pt = 0
        self.top_down_bin_warp = []
        self.number_of_windows = 0
        self.window_height = 0
        self.non_zero_warp_elements = []
        self.non_zero_y_warp_elements = []
        self.non_zero_x_warp_elements = []
        self.left_x_current = 0
        self.right_x_current = 0
        self.window_width_margin = 0
        self.recentre_window_min_pixels = 0
        self.left_lane_pixel_index = []
        self.right_lane_pixel_index = []
        # print("SlidingWindow object created.")

    def calc_window_height(self, top_down_bin_warp_polygon):
        """
        Split y size into proportional number of windows.
        :param top_down_bin_warp_polygon: perspective to operate on
        """
        self.window_height = np.int(top_down_bin_warp_polygon.shape[0] / self.number_of_windows)
        # print("SlidingWindow.calc_window_height set to :", self.window_height)

    def locate_nonzero_pixels_by_axis(self, top_down_bin_warp_polygon):
        self.non_zero_warp_elements = top_down_bin_warp_polygon.nonzero()
        self.non_zero_y_warp_elements = np.array(self.non_zero_warp_elements[0])
        self.non_zero_x_warp_elements = np.array(self.non_zero_warp_elements[1])
        # print("SlidingWindow.locate_nonzero_pixels_by_axis() done.")


# ------------------
def setup_sliding_window(left_x_pt, right_x_pt, sliding_window, top_down_bin_warp):
    """
    Set key SlidingWindow attributes, some calculated, some defined.
    :param left_x_pt: current position of SlidingWindow updated for each new window instance
    :param right_x_pt: current position of SlidingWindow updated for each new window instance
    :param sliding_window: object to update attributes of
    :param top_down_bin_warp: perspective to operate on
    """
    sliding_window.number_of_windows = 9
    sliding_window.calc_window_height(top_down_bin_warp)
    sliding_window.locate_nonzero_pixels_by_axis(top_down_binary_warp)
    sliding_window.left_x_current = left_x_pt
    sliding_window.right_x_current = right_x_pt
    sliding_window.window_width_margin = 50
    sliding_window.recentre_window_min_pixels = 30
    # print("SlidingWindow object key attributes now set.")


# ------------------
def extract_and_fit_sliding_window_lines(left_line, right_line, sliding_window):
    """

    :param left_line: update object attribute with non_zero pixel locations
    :param right_line: update object attribute with non_zero pixel locations
    :param sliding_window: current SlidingWindow object to operate on
    :return: best fit lines for left and right SlidingWindows
    """
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(sliding_window.left_lane_pixel_index)
    right_lane_inds = np.concatenate(sliding_window.right_lane_pixel_index)
    # Extract left and right line pixel positions
    left_x = sliding_window.non_zero_x_warp_elements[left_lane_inds]
    left_y = sliding_window.non_zero_y_warp_elements[left_lane_inds]
    right_x = sliding_window.non_zero_x_warp_elements[right_lane_inds]
    right_y = sliding_window.non_zero_y_warp_elements[right_lane_inds]
    left_line.all_x = left_x  # values for detected line pixels
    left_line.all_y = left_y  # values for detected line pixels
    right_line.all_x = right_x  # values for detected line pixels
    right_line.all_y = right_y  # values for detected line pixels
    # Fit a second order polynomial
    left_fit = np.polyfit(left_y, left_x,
                          2)  # Least squared-error best fit of polynomial of degree 2 to points in left_y, left_x
    right_fit = np.polyfit(right_y, right_x, 2)
    print("extract_and_fit_sliding_window_lines done.")
    return left_fit, right_fit


# ------------------
def track_lane_line_recent_and_best_fits(left_fit, left_line, right_fit, right_line, top_down_bin_warp):
    """
    Update history of recently fitted lines for SlidingWindows.
    :param left_fit: best fit line for SlidingWindow
    :param left_line: current line object
    :param right_fit: best fit line for SlidingWindow
    :param right_line: current line object
    :param top_down_bin_warp: perspective to operate on, Shape == (720, 1280)
    :return: y_values ; 720 evenly spaced numbers from 0 to 719.
    """
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    y_values = np.linspace(0, top_down_bin_warp.shape[0] - 1, top_down_bin_warp.shape[0])

    # Create a 2nd order polynomial using left_fit coefficients ...
    #     x[0] = p[0]        * y[0]     ** n + p[1]        * y[0]     + p[2]
    left_fit_x = left_fit[0] * y_values ** 2 + left_fit[1] * y_values + left_fit[2]
    # print("left_fit_x :", left_fit_x)
    right_fit_x = right_fit[0] * y_values ** 2 + right_fit[1] * y_values + right_fit[2]
    # print("right_fit_x :", right_fit_x)

    left_line.recent_x_fitted.append(left_fit_x)
    right_line.recent_x_fitted.append(right_fit_x)

    left_line.best_x = left_fit_x
    right_line.best_x = right_fit_x
    # print("y_values :", y_values)
    print("track_lane_line_recent_and_best_fits done.")
    return y_values


# ------------------
def draw_and_identify_good_sliding_windows(out_img, sliding_window, top_down_bin_warp, window):
    """
    Draw thin ('2') SlidingWindow rectangle, identify non-zero pixels within left & right SlidingWindow.
    :param out_img: output image that will have shaded polygon projected onto
    :param sliding_window: current SlidingWindow object to operate on
    :param top_down_bin_warp: perspective to operate on, Shape == (720, 1280)
    :param window: window item from for loop iteration over range of 9 total SlidingWindows
    """
    # Identify window boundaries in y for both right and left windows on x
    win_y_low = top_down_bin_warp.shape[0] - (window + 1) * sliding_window.window_height
    win_y_high = top_down_bin_warp.shape[0] - window * sliding_window.window_height
    win_x_left_low = sliding_window.left_x_current - sliding_window.window_width_margin
    win_x_left_high = sliding_window.left_x_current + sliding_window.window_width_margin
    win_x_right_low = sliding_window.right_x_current - sliding_window.window_width_margin
    win_x_right_high = sliding_window.right_x_current + sliding_window.window_width_margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_index = ((sliding_window.non_zero_y_warp_elements >= win_y_low) & (
        sliding_window.non_zero_y_warp_elements < win_y_high) & (
                           sliding_window.non_zero_x_warp_elements >= win_x_left_low) & (
                           sliding_window.non_zero_x_warp_elements < win_x_left_high)).nonzero()[0]
    good_right_index = ((sliding_window.non_zero_y_warp_elements >= win_y_low) & (
        sliding_window.non_zero_y_warp_elements < win_y_high) & (
                            sliding_window.non_zero_x_warp_elements >= win_x_right_low) & (
                            sliding_window.non_zero_x_warp_elements < win_x_right_high)).nonzero()[0]
    # print("good_left_index :", good_left_index)
    # print("good_right_index :", good_right_index)
    print("draw_and_identify_good_sliding_windows done.")
    return good_left_index, good_right_index


# ------------------
def fit_sliding_windows_lines_to_lanes(top_down_bin_warp, left_line, right_line):
    """
    Create fitted lines to left and right lane lines using SlidingWindow technique.
    :param top_down_bin_warp: perspective to operate on, Shape == (720, 1280)
    :param left_line: newly instantiated object
    :param right_line: newly instantiated object
    :return: offset_from_center, updated left_line, updated right_line, best fit y_values
    """
    # Create white, empty image 3D array to later overwrite with shaded projection
    out_img = np.dstack((top_down_bin_warp, top_down_bin_warp,
                         top_down_bin_warp)) * 255  # fit_sliding_windows_lines_to_lanes(top_down_binary_warp, left_line, right_line)
    # print("out_img.shape :", out_img.shape)  # (720, 1280, 3)

    # Locate the lane x points and calculate the mid-point.
    left_x_pt, mid_pt, right_x_pt = get_left_mid_right_pts_from_warp_histogram(top_down_bin_warp)

    # Calculate car offset from lane centre.
    offset_from_center = get_offset_from_center(right_x_pt, left_x_pt, mid_pt)
    print("offset_from_center :", offset_from_center)

    # Instantiate and initialise SlidingWindow
    sliding_window = SlidingWindow()
    setup_sliding_window(left_x_pt, right_x_pt, sliding_window, top_down_bin_warp)

    # Step through the windows one by one
    for window in range(sliding_window.number_of_windows):
        good_left_index, good_right_index = draw_and_identify_good_sliding_windows(out_img, sliding_window,
                                                                                   top_down_bin_warp, window)

        # Append these indices to the lists
        sliding_window.left_lane_pixel_index.append(good_left_index)
        sliding_window.right_lane_pixel_index.append(good_right_index)

        if len(good_left_index) > sliding_window.recentre_window_min_pixels:
            sliding_window.left_x_current = np.int(
                np.mean(sliding_window.non_zero_x_warp_elements[good_left_index]))
        if len(good_right_index) > sliding_window.recentre_window_min_pixels:
            sliding_window.right_x_current = np.int(
                np.mean(sliding_window.non_zero_x_warp_elements[good_right_index]))

    left_fit, right_fit = extract_and_fit_sliding_window_lines(left_line, right_line, sliding_window)

    y_values = track_lane_line_recent_and_best_fits(left_fit, left_line, right_fit, right_line,
                                                    top_down_bin_warp)
    return offset_from_center, left_line, right_line, y_values


# ------------------
def best_fit_left_and_right_x_polynomials(y_values):
    # Create a 2nd order polynomial using left_fit coefficients ...
    #     x[0] = p[0]                     * y[0]     ** n + p[1]                     * y[0]     + p[2]
    left_fit_x = left_line.current_fit[0] * y_values ** 2 + left_line.current_fit[1] * y_values + \
                 left_line.current_fit[2]
    # print("right_line.current_fit.shape :", right_line.current_fit.shape)  # (3,)
    right_fit_x = right_line.current_fit[0] * y_values ** 2 + right_line.current_fit[1] * y_values + \
                  right_line.current_fit[2]
    return y_values, left_fit_x, right_fit_x


# ------------------
def draw_fitted_line(points):
    """
    Iteratively generate the fitted line from the supplied points.
    :param points: [left | right] line points
    """
    previous = points[0]
    for point in points:
        cv2.line(out_img, (int(previous[0]), int(previous[1])), (int(point[0]), int(point[1])), [255, 255, 0],
                 10)
        previous = point


# ------------------
# Create Line objects : left and right
left_line = Line()
right_line = Line()

# Load test image
img = cv2.imread('test_images/test5.jpg')

# Undistort the test image, using the calibration matrix and distortion coefficients for this camera lens.
undistorted = undistort_image(img, c_matrix, dist_coeff)

# Generate the gradient binary and stacked saturation, colour, gradient binary.
gradient_binary, stacked_binary = colour_image_transformation_pipeline(undistorted)

# Warp perspective to top down view point.
transform_matrix_inverse, top_down_binary_warp = warp_perspective_to_top_down(source_transformation,
                                                                              stacked_binary,
                                                                              destination_transformation)

# Calculate left, right line attributes and distance from lane centre.
# 'y_axis' is output from track_lane_line_recent_and_best_fits()
offset_from_center, left_line, right_line, y_axis = fit_sliding_windows_lines_to_lanes(top_down_binary_warp,
                                                                                       left_line, right_line)

# Create y-axis numbers aka 720 evenly spaced numbers from 0 to 719.
# Note : top_down_warped_binary_stacked.shape : (720, 1280)
y_axis = np.linspace(0, top_down_binary_warp.shape[0] - 1, top_down_binary_warp.shape[0])
best_fit_left_and_right_x_polynomials(y_axis)

# out_img will have red coloured left SlidingWindows, blue coloured right SlidingWindows
out_img = np.dstack((top_down_binary_warp, top_down_binary_warp, top_down_binary_warp)) * 255
out_img[left_line.all_y, left_line.all_x] = [255, 0, 0]
out_img[right_line.all_y, right_line.all_x] = [0, 0, 255]

left_points = np.array(np.column_stack((left_line.recent_x_fitted[0], y_axis)))
print("left_points :", left_points)
right_points = np.array(np.column_stack((right_line.recent_x_fitted[0], y_axis)))
print("right_points :", right_points)

draw_fitted_line(left_points)
draw_fitted_line(right_points)
plt.imsave('output_images/fitted_line.jpg', out_img)
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.imshow(out_img)


# ------------------
def check_lane_lines_are_parallel(left_line_fit_x, right_line_fit_x):
    """
    Check that the difference in line absolute values is less than 100.
    :param left_line_fit_x: Fitted left line.
    :param right_line_fit_x: Fitted right line.
    :return: True if check passes, else False.
    """
    if abs((right_line_fit_x[0] - left_line_fit_x[0]) - (right_line_fit_x[-1] - left_line_fit_x[-1])) < 100:
        return True
    else:
        return False


# ------------------
def check_lane_lines_have_expected_distance(left_line_fit_x, right_line_fit_x):
    """
    Check distance between left and right top-down projected lines is within range.
    Range gives allowance for fanning out of top of lines.
    :param left_line_fit_x: Fitted left line.
    :param right_line_fit_x: Fitted right line.
    :return: True if check passes, else False.
    """
    if (570 < (right_line_fit_x[0] - left_line_fit_x[0]) < 820) and (
                    570 < (right_line_fit_x[-1] - left_line_fit_x[-1]) < 820):
        return True
    else:
        return False


# ------------------
def check_we_have_not_lost_sight_of_lines(left_line_to_check, right_line_to_check, top_down_bin_warp):
    """
    If have lost sight of lines, re-fit sliding windows.
    :param left_line_to_check: left line object
    :param right_line_object: right line object
    :param top_down_bin_warp: perspective to operate on, Shape == (720, 1280)
    :return: existing, else updated lines
    """
    if len(left_line_to_check.recent_x_fitted) or len(right_line_to_check.recent_x_fitted) == 0:
        centre_offset, left_line_to_check, right_line_to_check, y_values = fit_sliding_windows_lines_to_lanes(
            top_down_bin_warp, left_line_to_check,
            right_line_to_check)
    return left_line_to_check, right_line_to_check


# ------------------
def find_initial_non_zero_element_indices(top_down_bin_warp):
    """
    Find non-zero pixel positions.
    :param top_down_bin_warp: perspective to operate on, Shape == (720, 1280)
    :return: x, y position indices
    """
    nonzero = top_down_bin_warp.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    return nonzero_x, nonzero_y


# ------------------
def find_left_lane_pixels_positions_within_margin(left_fit, margin, nonzero_x, nonzero_y):
    """
    Extend search for pixels around margins.
    :param left_fit: polynomial coefficients for the most recent fit
    :param margin: number of pixels to extend search boundary
    :param nonzero_x: all non-zero pixel locations in the x dimension across the warped image
    :param nonzero_y: all non-zero pixel locations in the y axis across the warped image
    :return:
    """
    left_lane_index = (
        (nonzero_x > (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] - margin)) & (
            nonzero_x < (left_fit[0] * (nonzero_y ** 2) + left_fit[1] * nonzero_y + left_fit[2] + margin)))
    return left_lane_index


# ------------------
def find_right_lane_pixels_positions_within_margins(margin, nonzero_x, nonzero_y, right_fit):
    right_lane_index = (
        (nonzero_x > (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] - margin)) & (
            nonzero_x < (right_fit[0] * (nonzero_y ** 2) + right_fit[1] * nonzero_y + right_fit[2] + margin)))
    return right_lane_index


# ------------------
def extract_left_non_zero_pixel_positions(left_lane_index, nonzero_x, nonzero_y):
    """

    :param left_lane_index:
    :param nonzero_x:
    :param nonzero_y:
    :return:
    """
    left_x = nonzero_x[left_lane_index]
    left_y = nonzero_y[left_lane_index]
    return left_x, left_y


# ------------------
def extract_right_non_zero_pixel_positions(nonzero_x, nonzero_y, right_lane_index):
    right_x = nonzero_x[right_lane_index]
    right_y = nonzero_y[right_lane_index]
    return right_x, right_y


# ------------------
def best_fit_to_detected_lines(left_fit, left_x, left_y, right_fit, right_x, right_y):
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    return left_fit, right_fit


# ------------------
def generate_plotting_values(left_fit, right_fit, top_down_bin_warp):
    y_values = np.linspace(0, top_down_bin_warp.shape[0] - 1, top_down_bin_warp.shape[0])
    left_fit_x_to_plot = left_fit[0] * y_values ** 2 + left_fit[1] * y_values + left_fit[2]
    right_fit_x_to_plot = right_fit[0] * y_values ** 2 + right_fit[1] * y_values + right_fit[2]
    return left_fit_x_to_plot, right_fit_x_to_plot, y_values


# ------------------
def limit_accrued_fitted_line_history(left_line_obj, max_stored_lines, right_line_obj):
    if len(left_line_obj.recent_x_fitted) > max_stored_lines:  # max_stored_lines = 60
        np.delete(left_line_obj.recent_x_fitted, 0)
        np.delete(right_line_obj.recent_x_fitted, 0)


# ------------------
def search_for_and_update_lines(top_down_bin_warp, left_line_obj, right_line_obj,
                                max_allowed_stored_lines_history):
    """
    Search for lines in new warped image and update line objects.
    :param top_down_bin_warp:
    :param left_line_obj:
    :param right_line_obj:
    :param max_allowed_stored_lines_history:
    :return:
    """
    left_line_obj, right_line_obj = check_we_have_not_lost_sight_of_lines(left_line_obj, right_line_obj,
                                                                          top_down_bin_warp)

    left_fit = left_line_obj.current_fit
    # print("left_fit[0]: ", left_fit[0])  # e.g. -0.000299906383635
    # print("left_fit[1] : ", left_fit[1])  # e.g. 0.272966182225
    # print("left_fit[2] : ", left_fit[2])  # e.g. 243.371212179
    right_fit = right_line_obj.current_fit

    nonzero_x, nonzero_y = find_initial_non_zero_element_indices(top_down_bin_warp)

    margin = 20

    left_lane_index = find_left_lane_pixels_positions_within_margin(left_fit, margin, nonzero_x, nonzero_y)
    right_lane_index = find_right_lane_pixels_positions_within_margins(margin, nonzero_x, nonzero_y, right_fit)

    left_x, left_y = extract_left_non_zero_pixel_positions(left_lane_index, nonzero_x, nonzero_y)
    right_x, right_y = extract_right_non_zero_pixel_positions(nonzero_x, nonzero_y, right_lane_index)

    left_fit, right_fit = best_fit_to_detected_lines(left_fit, left_x, left_y, right_fit, right_x, right_y)

    left_fit_x_to_plot, right_fit_x_to_plot, y_values = generate_plotting_values(left_fit, right_fit,
                                                                                 top_down_bin_warp)

    if (check_lane_lines_are_parallel(left_fit_x_to_plot,
                                      right_fit_x_to_plot) and check_lane_lines_have_expected_distance(
        left_fit_x_to_plot, right_fit_x_to_plot)):

        left_line_obj.recent_x_fitted.append(left_fit_x_to_plot)
        right_line_obj.recent_x_fitted.append(right_fit_x_to_plot)

        # trim the line history
        limit_accrued_fitted_line_history(left_line_obj, max_allowed_stored_lines_history, right_line_obj)

        # smooth fitted line
        left_line_obj.best_x = np.average(left_line_obj.recent_x_fitted, axis=0)
        right_line_obj.best_x = np.average(right_line_obj.recent_x_fitted, axis=0)

        # update all values for detected line pixels
        left_line_obj.all_x = left_x
        left_line_obj.all_y = left_y
        right_line_obj.all_x = right_x
        right_line_obj.all_y = right_y

        # update polynomial coefficients for the most recent fit
        left_line_obj.current_fit = left_fit
        right_line_obj.current_fit = right_fit
    else:  # lane lines not parallel OR invalid width between them
        left_line_obj.lost_sight_of_lanes_count = + 1

        if left_line_obj.lost_sight_of_lanes_count > max_allowed_stored_lines_history:
            centre_offset, left_line_obj, right_line_obj, y_values = fit_sliding_windows_lines_to_lanes(
                top_down_bin_warp, left_line_obj,
                right_line_obj)
            left_line_obj.lost_sight_of_lanes_count = 0
    centre_offset = get_offset_from_center(right_fit_x_to_plot[0], left_fit_x_to_plot[0],
                                           top_down_bin_warp.shape[1] / 2)
    return centre_offset, left_line_obj, right_line_obj, y_values


# ------------------
# Quick check that nothing is broken ...
# offset_from_center, left_line, right_line, y_axis = search_for_and_update_lines(top_down_binary_warp, left_line,
#                                                                                 right_line, 20)

# ------------------
# TODO REFACTOR FROM HERE DOWN !
def find_curv_pix(y_values, left_line, right_line):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    y_eval = np.max(y_values)
    left_curve_radius = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curve_radius = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])
    print(left_curve_radius, right_curve_radius)


# ------------------
find_curv_pix(y_axis, left_line, right_line)  # e.g. 1730.24402261 1349.2995529


# ------------------
def find_curv_real(left_line, right_line, y_values):
    # Define conversions in x and y from pixels space to meters
    left_line_x = left_line.all_x
    left_line_y = left_line.all_y
    right_x = right_line.all_x
    right_y = right_line.all_y
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(y_values)
    left_fit_cr = np.polyfit(left_line_y * ym_per_pix, left_line_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curve_radius = ((1 + (
        2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (
        2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curve_radius, 'm', right_curverad, 'm')
    # print (len(rightx), ' ', len(righty))
    return left_curve_radius, right_curverad
    # Example values: 632.1 m    626.2 m


# ------------------
find_curv_real(left_line, right_line, y_axis)  # e.g. (547.92457013247042, 336.99267665542919)

# ------------------
# Quick check that nothing is broken ...
offset_from_center, left_line, right_line, y_axis = fit_sliding_windows_lines_to_lanes(top_down_binary_warp,
                                                                                       left_line, right_line)

# ------------------
""" 6.2 vehicle position with respect to center. """

# def get_offset_from_center(right_x, left_x, midpoint):
#     return (3.7 / 700) * abs(((right_x - left_x) / 2) + left_x - midpoint)


# def are_roughly_parallel(left_fitx, right_fitx):
#     if abs((right_fitx[0] - left_fitx[0]) - (right_fitx[-1] - left_fitx[-1])) < 100:
#         return True
#     else:
#         return False


# def have_proper_distance(left_fitx, right_fitx):
#     if (570 < (right_fitx[0] - left_fitx[0]) < 820) and (570 < (right_fitx[-1] - left_fitx[-1]) < 820):
#         return True
#     else:
#         return False


# ------------------

""" 7. Warp the detected lane boundaries back onto the original image. """

# ------------------

""" 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. """


def draw_lane(img, mtx, dist, src, top_down_polygon, dst, y_values, left_line, right_line):
    # Create an image to draw the lines on
    left_fitx = left_line.recent_x_fitted[-1]
    right_fitx = right_line.recent_x_fitted[-1]
    undistorted = undistort_image(img, mtx, dist)
    Minv, top_down_bin_warp = warp_perspective_to_top_down(src, top_down_polygon, dst)
    warp_zero = np.zeros_like(top_down_bin_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    # print(y_values, '  ', left_fitx)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, y_values]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_values])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # if (cv2.matchShapes(newwarp, prev_warp, 1, 0.0) > 0.01):
    #    newwarp = prev_warp
    # else:
    #    prev_warp = new_warp
    # Combine the top_down_polygon with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    return result


font = cv2.FONT_HERSHEY_SIMPLEX


def print_curvature(img, left_curverad, right_curverad, offset_from_center, left_line, right_line):
    text1 = "Radius of left Curvature= " + str(left_curverad) + ' m , right: ' + str(right_curverad)
    text2 = "Vehicle is: " + str(offset_from_center) + ' m from the center of the lane'
    text3 = "bottom: " + str(
        right_line.recent_x_fitted[-1][0] - left_line.recent_x_fitted[-1][0]) + ' top: ' + str(
        right_line.recent_x_fitted[-1][-1] - left_line.recent_x_fitted[-1][-1])
    cv2.putText(img, text1, (10, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(img, text2, (10, 130), font, 1, (255, 255, 255), 2)
    # cv2.putText(img, text3,(10,160), font, 1,(255,255,255),2)


# ------------------
left_curverad, right_curverad = find_curv_real(left_line, right_line, y_axis)

laned_image = draw_lane(straight_lines_image_2, c_matrix, dist_coeff, source_transformation,
                        top_down_warped_binary_polygon,
                        destination_transformation, y_axis,
                        left_line, right_line)
print_curvature(laned_image, left_curverad, right_curverad, offset_from_center, left_line, right_line)

cv2.imwrite('output_images/laned_image.jpg', laned_image)

plt.imshow(laned_image)

# ------------------

img = cv2.imread('test_images/straight_lines1.jpg')
prev_warp = None
left_line = Line()
right_line = Line()


def Pipeline(img, mtx, dist, left_line, right_line):
    undistorted = undistort_image(img, mtx, dist)
    matrix_trans_inv, top_down_warp_bin_poly = colour_image_transformation_pipeline(undistorted)
    Minv, warped = warp_perspective_to_top_down(source_transformation, top_down_warp_bin_poly,
                                                destination_transformation)
    offset_from_center, left_line, right_line, y_values = search_for_and_update_lines(warped, left_line,
                                                                                      right_line, 60)
    laned_image = draw_lane(undistorted, mtx, dist, source_transformation, top_down_warp_bin_poly,
                            destination_transformation,
                            y_values, left_line, right_line)
    left_curverad, right_curverad = find_curv_real(left_line, right_line, y_values)
    print_curvature(laned_image, left_curverad, right_curverad, offset_from_center, left_line, right_line)
    return laned_image


# Pipeline(img, c_matrix, dist_coeff, left_fit, right_fit)
Pipeline(img, c_matrix, dist_coeff, left_line.current_fit, right_line.current_fit)


# ------------------
# TODO : THIS COULD MEAN I NEED TO CREATE NEW FUNCTION SIGNATURE TO RETURN A TUPLE OF NDARRAYS.
def process_image(img):
    result = Pipeline(img, c_matrix, dist_coeff, left_line, right_line)
    return result


# ------------------
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# ------------------

white_output = 'processed.mp4'
clip1 = VideoFileClip("CarND-Advanced-Lane-Lines-master/project_video.mp4")
# TODO test assumption that 'process_image' is calling function 'process_image(img)' above. If so, create new signature function to fix
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
% time
white_clip.write_videofile(white_output, audio=False)

# ------------------
