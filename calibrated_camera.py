import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib qt

class Points:
    """A class containing mapped 3D to 2D points used for calibrating any camera."""
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        # 3d points in real world space
        self.objpoints = []
        # 2d points in image plane
        self.imgpoints = []

    def map_3d_object_to_2d_image_points(self, store=False, display=False):
        """
        Creates ndarray representing reference 9 col x 6 x row chessboard 3D coordinates,
        find the chessboard corner coordinates in each calibration (aka skewed) example image,
        and collect in imgpoints list.

        :param store: image with corner points overlay
        :param display: image with corner points overlay, for visual testing.
        """
        # prepare x,y,z object points of reference chessboard,
        # like [[ 0.  0.  0.] [ 1.  0.  0.] [ 2.  0.  0.] ... [ 8.  5.  0.]]
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        # print(objp)
        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            # img = cv2.imread(fname)
            img = mpimg.imread(fname)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                #self.draw_chess_board(corners, display, idx, img, ret, store)

        # return self.objpoints, self.imgpoints


def draw_chess_board(self, corners, display, idx, img, ret, store):
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    if store:
        write_name = 'corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)
    if display:
        cv2.imshow('img', img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()


def calibrate_and_undistort(img, show_undistored_img=False):
    # Generate point mapping
    points = Points()
    points.map_3d_object_to_2d_image_points(store=False, display=False)

    # Test undistortion on an image
    # img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points.objpoints, points.imgpoints, img_size, None, None)

    destination_img = cv2.undistort(img, mtx, dist, None, mtx)
    # cv2.imwrite('camera_cal/undistorted_calibration1.jpg', destination_transformation)

    # save_calibration_result(dist, mtx)

    if show_undistored_img:
        # Visualize undistortion
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        # ax1.imshow(img)
        # ax1.set_title('Original Image', fontsize=30)
        # ax2.imshow(destination_transformation)
        # ax2.set_title('Undistorted Image', fontsize=30)
        cv2.imshow('destination_img', destination_img)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

    return destination_img


def save_calibration_result(dist, mtx):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_cal/camera_calibration_wide_pickle.p", "wb"))
    # destination_transformation = cv2.cvtColor(destination_transformation, cv2.COLOR_BGR2RGB)
