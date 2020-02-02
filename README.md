#Advanced Lane Finding

This project covers camera calibration, colour and gradient thresholding, applied geometry (calculating the radii of left and right lane ‘circles’) / algebra (degree-2 polynomials fit to left and right lanes) and tracking detected lanes using the Sliding Window technique.

The result is a coloured (green), filled polygon to highlight the detected road lane.


##The Project

The approach of this project is the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


##How I computed the camera matrix and distortion coefficients.

The code for this step is contained in section 1. of the Jupyter notebook located in "./writeup.ipynb" (note : the file called writeup.py will run the entire notebook code).

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world - see find_corner_coordinates_and_map_to_reference_chessboard function. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and Points.objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. Points.imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output Points.objpoints and Points.imgpoints to compute the camera calibration and distortion coefficients using cv2.calibrateCamera() - see calculate_camera_distortion_coefficients function. I applied this distortion correction to the test image using cv2.undistort_image() - see section 2.

##Pipeline (single images)

- I used color transforms, gradients or other methods to create a thresholded binary image.
- I used a combination of color and gradient thresholds to generate a binary image.
- The code for my perspective transform includes a function called warp_perspective_to_top_down - see section 4. and lines 325-328 in the file writeup.py.

```python
	def warp_perspective_to_top_down(src, img, dst):
	    """
	    Calculate the inversed transformation matrix and warped top down transform image
	    :param src: source points where warp transforms from
	    :param img: stacked binary thresholded image that includes saturation, gradient direction, colour intensity
	    :param dst: destination points where warp transforms to
	    :return: the transformation matrix inversed, warped top-down binary image
	    """
	    transformation_matrix = cv2.getPerspectiveTransform(src, dst)  # the transform matrix
	    transformation_matrix_inverse = cv2.getPerspectiveTransform(dst, src)  # the transform matrix inverse
	    warped_to_top_down = cv2.warpPerspective(img, transformation_matrix,
	                                             img_size)  # warp image to a top-down view
	    # print("Perspective warp done.")
	    return transformation_matrix_inverse, warped_to_top_down
```

The warp_perspective_to_top_down() function takes as inputs an image (img), as well as source (src) and destination (dst) points. I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:



|Source|Destination|
|------|-----------|
|585, 460|320, 0|
|203, 720|320, 720|
|1127, 720|960, 720|
|695, 460|960, 0|

I verified that my perspective transform was working as expected by drawing the source_transformation and destination_transformation points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

## Indentifying lane-line pixels and fit their positions with a polynomial

Then I setup_lines by applying the iterative sliding window technique :

- Created top-down output image template.
- Located the spiked points from the histogram of the binary warped image on the x-axis and calculated the lane midpoint.
- Captured the active, non-zero pixels in the x and y directions.
- Set a margin around the sliding windows to locate new active pixels.
- Set a trigger threshold that would recentre the next sliding in order to follow the lane curvature.
- Fit my lane lines with a degree-2 polynomial.

>see section "6.1 Determine the curvature of the lane and ..." for details.

## Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the following functions in my code in write.py :

Radii
```python
def convert_and_calc_curvature_radii(left_line_obj, right_line_obj, y_axis_values):
    """
    Convert x, y from pixels to metres, calc new left and right lane radii
    :param left_line_obj: access the Line object attributes
    :param right_line_obj: access the Line object attributes
    :param y_axis_values: number series from 0 - 719 representing image y-axis
    :return: curve radii
    """
    # Get latest line attributes
    left_x = left_line_obj.allx
    left_y = left_line_obj.ally
    right_x = right_line_obj.allx
    right_y = right_line_obj.ally

    # Calc metres per pixel of lane length and width
    y_metres_per_pix = 30 / 720  # metres per pixel calc
    x_metres_per_pix = 3.7 / 700  # metres per pixel calc

    # Get bottom of image from y values aka 719
    y_max = np.max(y_axis_values)

    # Calc and least squares error fit the degree-2 polynomials
    left_fit_metres_per_pixel = np.polyfit(left_y * y_metres_per_pix, left_x * x_metres_per_pix, 2)
    right_fit_metres_per_pixel = np.polyfit(right_y * y_metres_per_pix, right_x * x_metres_per_pix, 2)

    # Calc new curvature radius for left and right
    left_curve_radius = ((1 + (
        2 * left_fit_metres_per_pixel[0] * y_max * y_metres_per_pix + left_fit_metres_per_pixel[
            1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_metres_per_pixel[0])
    right_curve_radius = ((1 + (
        2 * right_fit_metres_per_pixel[0] * y_max * y_metres_per_pix + right_fit_metres_per_pixel[
            1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_metres_per_pixel[0])

    print("left_curve_radius :", "{0:.0f}".format(left_curve_radius), "metres")
    print("right_curve_radius :", "{0:.0f}".format(right_curve_radius), "metres")
    return left_curve_radius, right_curve_radius
```

Vehicle offset from lane centre ; Called find_key_lane_points_along_x function to calculate the lanes midpoint, then passed this to get_offset_from_center to complete the calculation.

```python
def find_key_lane_points_along_x(binary_warped):
    """
    Use max points from histogram spikes, calculate left, mid and right x points.
    :param binary_warped: top-down perspective of lane lines to operate on
    :return: left, mid and right x points
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # print("histogram.shape : ", histogram.shape)  # (1280,)
    midpoint = np.int(histogram.shape[0] / 2)
    # print("midpoint :", midpoint)  # 640
    left_x_base = np.argmax(histogram[:midpoint])  # get max point between left and middle
    # print("left_x_base :", left_x_base)  # e.g. 290 and 312
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint  # get max point between middle and right
    # print("right_x_base :", right_x_base)  # e.g. 983 and 963
    return left_x_base, midpoint, right_x_base
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
####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
```

I implemented this step in lines 1077-1079 in my code in writeup.py in the function draw_filled_polygon(). Here is an example of my result on a test image:
> see section "7. Warp the detected lane boundaries back onto the original image.".

## Problems and issues

1. Code from lesson video showed application of np.arctan2() to absolute sobel gradients however I couldn't get this to work in my code - see lines 155, 162-164 in sobel_gradient_direction function in file writeup.py - and resorted to rescaling the gradients in the x direction instead.

2. Impact of variation in colour saturation (see 1st image below) evidenced by lowering threshold parameter (150 vs 180) and tree shade on the road (see 2nd image below) caused momentary glitches in the projected polygon. This scenario (see 2nd image below) may result in the car veering suddenly of course at speed if the lane detection was the highest priority input to the controller at that instant. This shows how brittle computer vision using 'traditional' tools / techniques is.

3. If I had more time I'd experiment with training a Deep Learning agent to improve robustness in varied lighting conditions.
