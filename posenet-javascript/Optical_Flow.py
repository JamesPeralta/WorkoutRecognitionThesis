import cv2
import numpy as np


video_size = (500, 500)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def generate_sparse_features():
    x_length = video_size[0]
    y_length = video_size[1]
    point_stride = 10

    x_vector = np.arange(point_stride, x_length, point_stride, dtype=np.float32)

    y_vector = np.arange(point_stride, y_length, point_stride, dtype=np.float32)

    return np.array(np.meshgrid(x_vector, y_vector)).T.reshape(-1, 1, 2)


def find_displaced_features(prev_f, next_f):
    # Find displaced features with respect to y
    diff_arr = np.subtract(prev_f, next_f)
    diff_arr_y = diff_arr[:, 1]

    increasing = np.argwhere(diff_arr_y > 5)[:, 0]
    if len(increasing) < 75:
        increasing = np.array([])

    decreasing = np.argwhere(diff_arr_y < -5)[:, 0]
    if len(decreasing) < 75:
        decreasing = np.array([])

    increasing_size = increasing.size
    decreasing_size = decreasing.size

    if increasing_size == decreasing_size:
        # print("Stationary")
        return []
    elif increasing_size > decreasing_size:
        print("Increasing")
        return increasing
    else:
        print("Decreasing")
        return decreasing


def calculate_optical_flow(prev_t, next_t, frame=None):
    # Convert these to gray because we only need luminisence layer
    prev_gray = cv2.cvtColor(prev_t, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_t, cv2.COLOR_BGR2GRAY)

    # Generate a dense feature set F(t-1) for prev frame
    prev_features = generate_sparse_features()

    # Compute the Optical flow for frame t using F(t-1), this generates
    # F(t) and status tells us if a feature couldn't be found in frame t
    next_features, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_features, None, **lk_params)

    # Selects features that have moved
    good_features_t_1 = prev_features[status == 1]  # Features at t-1
    good_features_t = next_features[status == 1]  # Features a t

    moved_indices = find_displaced_features(good_features_t_1, good_features_t)

    if frame is None:
        return

    # Only draw features that have moved
    for i, (new, old) in enumerate(zip(good_features_t_1[moved_indices], good_features_t[moved_indices])):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()

        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv2.circle(frame, (a, b), 3, color, -1)

    return frame


cap = cv2.VideoCapture("./ohp35.mp4")
# cap = cv2.VideoCapture(1)
video_size = (500, 500)
# cap = cv2.VideoCapture(0)

# Variable for color to draw optical flow track
color = (0, 255, 0)

ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, video_size)
while cap.isOpened():
    ret, next_frame = cap.read()
    next_frame = cv2.resize(next_frame, video_size)
    raw_next_frame = next_frame  # Store the raw next frame for later

    next_frame = calculate_optical_flow(prev_frame, next_frame, raw_next_frame)

    # Display the resulting frame
    cv2.imshow('frame', next_frame)

    prev_frame = raw_next_frame
    key_pressed = cv2.waitKey(50)
    if key_pressed & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

