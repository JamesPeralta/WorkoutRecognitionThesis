import cv2
import numpy as np


video_size = (500, 500)


def generate_sparse_features():
    x_length = video_size[0]
    y_length = video_size[1]
    point_stride = 10

    x_vector = np.arange(point_stride, x_length, point_stride, dtype=np.float32)

    y_vector = np.arange(point_stride, y_length, point_stride, dtype=np.float32)

    return np.array(np.meshgrid(x_vector, y_vector)).T.reshape(-1, 1, 2)


def find_displaced_features(prev, next):
    # Find displaced features with respect to y
    diff_arr = np.subtract(prev, next)
    diff_arr_y = diff_arr[:, 1]

    increasing = np.argwhere(diff_arr_y > 2)[:, 0]
    if len(increasing) > 20:
        print("Increasing")
        return increasing

    decreasing = np.argwhere(diff_arr_y < -2)[:, 0]
    if len(decreasing) > 20:
        print("Decreasing")
        return decreasing

    print("Stationary")
    return []


cap = cv2.VideoCapture("./ohp6.mp4")
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

prev = None
# print(prev[:10].dtype)
# print(" _________________ ")
# prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# print(prev[:10].dtype)

mask = np.zeros_like(first_frame)

frame_on = 0
while cap.isOpened():
    prev = generate_sparse_features()

    frame_on = (frame_on + 1) % 5

    ret, next_frame = cap.read()
    next_frame = cv2.resize(next_frame, video_size)

    # Only need the luminance channel for detecting edges - less computationally expensive
    gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Take in prev features and compute where they are next which is return in "next"
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    # Selects features that have moved
    good_old = prev[status == 1]
    good_new = next[status == 1]

    moved_indices = find_displaced_features(good_old, good_new)

    # Only draw features that have moved
    for i, (new, old) in enumerate(zip(good_new[moved_indices], good_old[moved_indices])):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        # mask = cv2.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        next_frame = cv2.circle(next_frame, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    # output = cv2.add(next_frame)

    # Display the resulting frame
    cv2.imshow('frame', next_frame)

    # Sets the new previous frames
    prev_gray = gray.copy()

    # Updates the features to ones that can still be found
    prev = good_new.reshape(-1, 1, 2)

    key_pressed = cv2.waitKey(30)
    if key_pressed & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

