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

"""
notes :
1. objp[:,:2] assigns values to x,y dimensions only and leaves z as zeros
2. np.mgrid[0:8, 0:6] == [cols == x, rows == y]
3. mgrid() returns 1 row of 3D for each single generated spaced value aka a Transpose of what is returned by np.arange()
    see http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/ for examples.
4. "np.reshape(-1", where -1  == a, aka the array to be shaped.
    One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    see https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
5. "np.reshape(-1, 2" where 2 == newshape, aka the new shape that is compatible with the original shape
"""

# objp = np.mgrid[0:8, 0:6]
# objp = np.meshgrid(np.arange(0, 8), np.arange(0, 6))
"""
[
array([[0, 1, 2, 3, 4, 5, 6, 7],
       [0, 1, 2, 3, 4, 5, 6, 7],
       [0, 1, 2, 3, 4, 5, 6, 7],
       [0, 1, 2, 3, 4, 5, 6, 7],
       [0, 1, 2, 3, 4, 5, 6, 7],
       [0, 1, 2, 3, 4, 5, 6, 7]]),
array([[0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4, 4, 4],
       [5, 5, 5, 5, 5, 5, 5, 5]])
]
"""

# objp = np.mgrid[0:8, 0:6]  # 0:8 == cols == x, 0:6 == rows == y
"""
objp = np.mgrid[0:8, 0:6]
[[[0 0 0 0 0 0]
  [1 1 1 1 1 1]
  [2 2 2 2 2 2]
  [3 3 3 3 3 3]
  [4 4 4 4 4 4]
  [5 5 5 5 5 5]
  [6 6 6 6 6 6]
  [7 7 7 7 7 7]]

 [[0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]
  [0 1 2 3 4 5]]]
"""

# objp = np.mgrid[0:8, 0:6].T
"""
[[[0 0]
  [1 0]
  [2 0]
  [3 0]
  [4 0]
  [5 0]
  [6 0]
  [7 0]]

 [[0 1]
  [1 1]
  [2 1]
  [3 1]
  [4 1]
  [5 1]
  [6 1]
  [7 1]]

 [[0 2]
  [1 2]
  [2 2]
  [3 2]
  [4 2]
  [5 2]
  [6 2]
  [7 2]]

 [[0 3]
  [1 3]
  [2 3]
  [3 3]
  [4 3]
  [5 3]
  [6 3]
  [7 3]]

 [[0 4]
  [1 4]
  [2 4]
  [3 4]
  [4 4]
  [5 4]
  [6 4]
  [7 4]]

 [[0 5]
  [1 5]
  [2 5]
  [3 5]
  [4 5]
  [5 5]
  [6 5]
  [7 5]]]

"""

# objp = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # reshape back into 2 cols, 1 for x and 1 for  y coordinates

"""
[[0 0]
 [1 0]
 [2 0]
 [3 0]
 [4 0]
 [5 0]
 [6 0]
 [7 0]
 [0 1]
 [1 1]
 [2 1]
 [3 1]
 [4 1]
 [5 1]
 [6 1]
 [7 1]
 [0 2]
 [1 2]
 [2 2]
 [3 2]
 [4 2]
 [5 2]
 [6 2]
 [7 2]
 [0 3]
 [1 3]
 [2 3]
 [3 3]
 [4 3]
 [5 3]
 [6 3]
 [7 3]
 [0 4]
 [1 4]
 [2 4]
 [3 4]
 [4 4]
 [5 4]
 [6 4]
 [7 4]
 [0 5]
 [1 5]
 [2 5]
 [3 5]
 [4 5]
 [5 5]
 [6 5]
 [7 5]]
"""

objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # reshape back into 2 cols, 1 for x and 1 for  y coordinates
"""
[[ 0.  0.  0.]
 [ 1.  0.  0.]
 [ 2.  0.  0.]
 [ 3.  0.  0.]
 [ 4.  0.  0.]
 [ 5.  0.  0.]
 [ 6.  0.  0.]
 [ 7.  0.  0.]
 [ 0.  1.  0.]
 [ 1.  1.  0.]
 [ 2.  1.  0.]
 [ 3.  1.  0.]
 [ 4.  1.  0.]
 [ 5.  1.  0.]
 [ 6.  1.  0.]
 [ 7.  1.  0.]
 [ 0.  2.  0.]
 [ 1.  2.  0.]
 [ 2.  2.  0.]
 [ 3.  2.  0.]
 [ 4.  2.  0.]
 [ 5.  2.  0.]
 [ 6.  2.  0.]
 [ 7.  2.  0.]
 [ 0.  3.  0.]
 [ 1.  3.  0.]
 [ 2.  3.  0.]
 [ 3.  3.  0.]
 [ 4.  3.  0.]
 [ 5.  3.  0.]
 [ 6.  3.  0.]
 [ 7.  3.  0.]
 [ 0.  4.  0.]
 [ 1.  4.  0.]
 [ 2.  4.  0.]
 [ 3.  4.  0.]
 [ 4.  4.  0.]
 [ 5.  4.  0.]
 [ 6.  4.  0.]
 [ 7.  4.  0.]
 [ 0.  5.  0.]
 [ 1.  5.  0.]
 [ 2.  5.  0.]
 [ 3.  5.  0.]
 [ 4.  5.  0.]
 [ 5.  5.  0.]
 [ 6.  5.  0.]
 [ 7.  5.  0.]]
"""


print(objp)