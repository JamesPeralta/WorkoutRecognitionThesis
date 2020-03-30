import cv2
import numpy as np

cap = cv2.VideoCapture("./squat6.mp4")
video_size = (500, 500)
# cap = cv2.VideoCapture(0)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variable for color to draw optical flow track
color = (0, 255, 0)

ret, first_frame = cap.read()
first_frame = cv2.resize(first_frame, video_size)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

mask = np.zeros_like(first_frame)

while cap.isOpened():
    ret, next_frame = cap.read()
    next_frame = cv2.resize(next_frame, video_size)

    # Only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1]
    # Selects good feature points for next position
    good_new = next[status == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        next_frame = cv2.circle(next_frame, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    output = cv2.add(next_frame, mask)

    # Display the resulting frame
    cv2.imshow('frame', output)

    # Sets the new previous frames
    prev_gray = gray

    # Updates the features to ones that can still be found
    prev = good_new.reshape(-1, 1, 2)

    key_pressed = cv2.waitKey(15)
    if key_pressed & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

