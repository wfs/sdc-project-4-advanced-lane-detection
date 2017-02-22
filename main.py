import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

""" Structure """
import calibrated_camera as camera
import gradient_image
import lane_lines

""" Computer Vision Setup """
# 1. Calibrate Camera and Correct Distortion

# points = camera.Points()
# points.map_3d_object_to_2d_image_points(store=False, display=False)  # To test, set params (store=False, display=True)
# print("points.imgpoints : ", points.imgpoints)  # 9x6 = 54 x,y coordinate lists per image array.
camera.calibrate(show_undistored_img=False)


# TODO 2. Iterate over each frame in the video, applying all steps below until detect lane lines
video_frame = image_to_test_on = gradient_image.BinaryMask.TEST_IMAGE


# TODO 3. Colour and Gradient Thresholds
# 3.2 Apply colour threshold

# 3.2. Generate gradient direction binary mask
sobel_kernel_size = 15
threshold = (0.7, 1.3)  # param tuple
binary_image = gradient_image.BinaryMask.dir_threshold(video_frame, sobel_kernel_size, threshold)
# image_to_test_on = gradient_image.BinaryMask.TEST_IMAGE
gradient_image.BinaryMask.view_mask_test(video_frame, sobel_kernel_size, threshold)


# TODO 4. Transform Perspective


""" Find Real Lane Lines : Base Measurement """
# TODO 5. Detect Lane Lines


# TODO 6. Determine Lane Curvature
# TODO TEST : Automatically determining if your detected lines are the real thing :
# TODO 6.1. Check curvature : that both left and right lines have similar curvature.
# TODO 6.2. Check separation : that they are separated by approx the correct distance horizontally
# TODO 6.3. Check parallel : that they are roughly parallel.
# TODO 6.4. Finding your offset from lane centre : assume camera is mounted at centre of car, so lane centre is at midpoint at bottom of image.


# TODO Keeping track of recent measurements : instantiate lane_lines.Line for both left and right lane lines to keep track of recent values from previously processed images.


""" Find Real Lane Lines : Next Measurement """
# TODO 7. search within a window around the previous detection, then perform TEST
"""
After determining you found the lines, where to look in the next frame :
simply search within a window around the previous detection

For example, if you fit a polynomial (WARN : I'd like to apply finding_the_lines_convolution.py instead!!!), then
for each y position, you have an x position that represents the lane center from the last frame.
Search for the new line within +/- some margin around the old line center.
See finding_the_lines.py, Step 5 for details.

Then do automatic test if detected lines are real aka check curvature, separation, parallel.
"""

"""
If you lose track of the lines : assume just a bad image for 3-5 consecutive frames and use previous positions from
last good frame.

Else, go back to the blind search method using a histogram and sliding window, or other method,
to re-establish your measurement.

"""

"""
Smoothing (average) your measurements over last n frames.

"""

"""
Drawing the line back down onto the road

Once you have a good measurement of the line positions in warped space,
it's time to project your measurement back down onto the road!

Let's suppose, as in the previous example, you have a warped binary image called warped, and
you have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which
represent the x and y pixel values of the lines.
You can then project those lines onto the original image as follows:
"""
# # Create an image to draw the lines on
# warp_zero = np.zeros_like(warped).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
# # Recast the x and y points into usable format for cv2.fillPoly()
# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# pts = np.hstack((pts_left, pts_right))
#
# # Draw the lane onto the warped blank image
# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#
# # Warp the blank back to original image space using inverse perspective matrix (Minv)
# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# # Combine the result with the original image
# result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
# plt.imshow(result)

