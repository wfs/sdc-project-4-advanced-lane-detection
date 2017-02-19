import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

"""
Tips and tricks for the project
"""

"""
Camera calibration : set chessboard size to 9x6, not 8x6 as in the lesson.
"""

"""
Do your curvature values make sense?
"""

"""
Finding your offset from lane centre : assume camera is mounted at centre of car, so lane centre is at
midpoint at bottom of image.
"""

"""
Keeping track of recent measurements : instantiate lane_lines.Line for both left and right lane lines
to keep track of recent values from previously processed images.
"""

"""
Automatically determining if your detected lines are the real thing :

1. Check curvature : that both left and right lines have similar curvature.
2. Check separation : that they are separated by approx the correct distance horizontally
3. Check parallel : that they are roughly parallel.
"""

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
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)

