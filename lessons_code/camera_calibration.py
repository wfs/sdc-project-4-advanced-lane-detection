import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping
# used in Jupyter notebook from video
# %matplotlib inline
# %matplotlib qt

# use at least 20 images of skewed chessboard examples (aka photos taken at different angles and distances)
# to reliably calibrate your camera
# have a separate test image to test the calibration has worked correctly

# Read in a calibration image
img = mping.imread('../camera_cal/calibration1.jpg')
plt.imshow(img)

# next, i'll map the coordinates of the corners in this 2D image (which i'll call image_points),
# to the 3D coordinates of the real, undistorted chessboard corners (which i'll call object_points).


# Arrays to store object points and image points from all the images

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object inner corner points, like (0,0,0), (1,0,0), (2,0,0) ..., (7,5,0)
objp = np.zeros([6*8, 3], np.float32)  # note the [rows*cols, 3D] (where rows == y, cols == x) in the numpy array of lists initialisation

# Create an x, y grid with coordinates
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
# print(objp)

# Next, to create the image points, i want to look at
# look at the distorted calibration image and
# detect the corners of the board
# in a grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)  # 'None' flags

# if corners are found, add object points, image points
if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    # draw and display the corners
    img = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
    plt.imshow(img)

