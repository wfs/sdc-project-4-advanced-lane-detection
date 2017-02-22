# ---------------
""" 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. """
# ---------------
""" Imports """
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import glob

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
    :return: ret [True | False], coefficients
    """
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img_size, None, None)
    print("Distortion coefficients calculated.")
    return mtx, dist

# ------------------
def undistort_calibration_image(img, mtx, dist):
    """
    Undistort the calibration image.
    :param img: calibration image
    :param mtx: calibration coefficient
    :param dist: calibration coefficient
    :return: undistorted calibration image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    print("Image undistorted.")
    return undist

# ------------------
def visual_comparison_of_distorted_undistorted_calibration_images(img1, img2, title1, title2, cmap):
    """
    Side-by-side visual comparison test.
    :param img1: calibration image
    :param img2: undistorted calibration image
    :param title1: Distorted
    :param title2: Undistorted
    :param cmap: grayscale, else handle colour
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.set_title(title2, fontsize=50)
    if cmap=='gray':
        ax2.imshow(img2, cmap='gray')
    else:
        ax2.imshow(img2)
        ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    print("Side-by-side view done.")

# ------------------
find_corner_coordinates_and_map_to_reference_chessboard()

calibration_img = cv2.imread("camera_cal/calibration1.jpg")
mtx, dist = calculate_camera_distortion_coefficients(calibration_img, Points.objpoints, Points.imgpoints)

undistorted_calibration_img = undistort_calibration_image(calibration_img, mtx, dist)
cv2.imwrite('camera_cal/undist_calibration1.jpg', undistorted_calibration_img)

visual_comparison_of_distorted_undistorted_calibration_images(calibration_img, undistorted_calibration_img, "Distorted", "Undistorted", 'gray')

# ------------------

""" 2. Apply a distortion correction to raw images. """

# ------------------

""" 3. Use color transforms, gradients, etc., to create a thresholded binary image. """

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
