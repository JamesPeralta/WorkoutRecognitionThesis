import cv2
import numpy as np

import posenet.constants
from skimage import data, img_as_float
from skimage import exposure
from joblib import dump, load

keypoints_list = [
    "nose", #1
    "leftEye", #2
    "rightEye", #3
    "leftEar", #4
    "rightEar", #5
    "leftShoulder", #6
    "rightShoulder", #7
    "leftElbow", #8
    "rightElbow", #9
    "leftWrist", #10
    "rightWrist", #11
    "leftHip", #12
    "rightHip", #13
    "leftKnee", #14
    "rightKnee", #15
    "leftAnkle", #16
    "rightAnkle", #17
]

model = load('./_models/svm_workout_detector.joblib')
x_scaler = load('./_models/x_scaler.save')
y_scaler = load('./_models/y_scaler.save')
resize_factor = 0.4


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img / 255.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def read_cap(cap, scale_factor=1.0, output_stride=16):
    res, img = cap.read()

    if not res:
        raise IOError("webcam failure")

    # img = img[70:, 200: 500, :]
    # img = img[120:660, 120: 560, :]
    # img = img[:, 160: 500, :]
    # img = img[70:, 200: 500, :]
    height = int(img.shape[1] * resize_factor)
    width = int(img.shape[0] * resize_factor)
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
    img = img[:, 280: 480, :]

    return _process_input(img, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    """
    This function takes in possible detections of poses and filters the poses
    based on the min_pose_score and filters each part based on the min_part_score

    :return out_img: The image with key points drawn on
    :return num_key_points_detected: Number of keypoints detected from frame (17 per person)
    :return people_location: Coordinates of people after filter
    """
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    num_keypoints_detected = 0
    people_location = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        people_location.append(keypoint_coords[ii, :, ::-1])
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
            num_keypoints_detected += 1

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

    # Need to know the number of people in the image to filter
    # for difficulty based on the amount of people detected
    num_of_people_detected = len(people_location)
    return out_img, num_keypoints_detected, people_location, num_of_people_detected


def draw_bounding_box(overlay_image, person):
    """
    This function takes as input the co-ordinates of a person after
    the keypoint extractor is applied and then returns a bounding box
    around the user detected
    """
    # Grab all x-coordinates of the user
    x_axis = person[:, 0]
    # Grab all y-coordinates of the user
    y_axis = person[:, 1]

    # obtain max and mins for each axis
    x_min = int(x_axis.min())
    x_max = int(x_axis.max())
    y_min = int(y_axis.min())
    y_max = int(y_axis.max())

    overlay_image = cv2.line(overlay_image, (x_min, y_min), (x_max, y_min), (255, 0, 0), 2)
    # Right
    overlay_image = cv2.line(overlay_image, (x_max, y_min), (x_max, y_max), (255, 0, 0), 2)
    # Bottom
    overlay_image = cv2.line(overlay_image, (x_min, y_max), (x_max, y_max), (255, 0, 0), 2)
    # Left
    overlay_image = cv2.line(overlay_image, (x_min, y_min), (x_min, y_max), (255, 0, 0), 2)

    return overlay_image


def label_and_return_keypoints(people_location):
    keypoint_coord_dict = {}
    for part, coords in list(zip(keypoints_list, people_location)):
        keypoint_coord_dict[part] = coords
    return keypoint_coord_dict


def draw_coord_grid(overlay_image):
    """
    Draws coordinate grid on the image
    :return:
    """
    GRID_SIZE = 20
    height, width, channels = overlay_image.shape

    # Print x-axis lines
    for x in range(0, width - 1, GRID_SIZE):
        cv2.line(overlay_image, (x, 0), (x, height), (255, 0, 0), 1, 1)
        number = str(x)
        for ind, char in enumerate(number):
            cv2.putText(overlay_image, char, (x, 12 + (ind * 14)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),
                        2, cv2.LINE_AA)
    # Print y-axis lines
    for y in range(0, height - 1, GRID_SIZE):
        cv2.line(overlay_image, (0, y), (width, y), (255, 0, 0), 1, 1)
        cv2.putText(overlay_image, str(y), (0, y + 3), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0),
                    2, cv2.LINE_AA)

    return overlay_image


def detect_workout_type(person_keypoints):
    if str(person_keypoints.shape) == "(17, 2)":
        # Add the batch dimension
        person_keypoints = np.expand_dims(person_keypoints, axis=0)

        # Scale the x axis keypoints
        person_keypoints[:, :, 0] = x_scaler.transform(person_keypoints[:, :, 0])

        # Scale the y axis keypoints
        person_keypoints[:, :, 1] = y_scaler.transform(person_keypoints[:, :, 1])

        # Reshape the keypoints
        person_keypoints = person_keypoints.reshape((person_keypoints.shape[0], person_keypoints.shape[1] * person_keypoints.shape[2]))

        # Predict on the Keypoint
        prediction = model.predict(person_keypoints)[0]

        if prediction == 0:
            return "ohp"
        elif prediction == 1:
            return "squats"

    return "Don't know"


def rate_of_change(past_keypoints, present_keypoints):
    increasing = False
    decreasing = False
    for index in range(0, len(keypoints_list)):
        # Retrieve datapoint for a body part
        past = past_keypoints[index, 1]
        present = present_keypoints[index, 1]

        # Calculate growth rates
        growth_rate = (present - past) / past
        if growth_rate > 0.3:
            increasing = True
            break

        # Calculate decay rates
        decay_rate = (past - present) / present
        if decay_rate > 0.3:
            decreasing = True
            break

    if increasing is False and decreasing is False:
        return "Still"
    elif increasing is True:
        return "Up"
    else:
        return "Down"
