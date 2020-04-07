import json
import os

# Accessing Annotation file constants
ANNOTATIONS = "subject_blocks"
ANNOTATION_TYPE = "name"
START = "enter_frame"
EXIT = "exit_frame"
LABEL_ARR = "events"
LABEL = "data"
METADATA = "video_metadata"
DURATION = "duration"

# Creating annotation arr constants
ANNOTATION_KEY = "annotation_type"
START_FRAME_KEY = "start_frame"
END_FRAME_KEY = "end_frame"
LABEL_KEY = "label"


def parse_exercise_labels(annotation_locations, video_name):
    """
    Labels each frame with the label from the annotion file.

    Parameters:

    Returns:
    :return: {0: Squat, 1: Squat, 2: Nothing}: For Frame 0 and 1 the person is squating,
    at Frame 2 they are doing nothing
    """
    # Locate annotation file
    possible_files = os.listdir("{}/labels".format(annotation_locations))
    annotation_file = list(filter(lambda x: video_name in str(x), possible_files))[0]

    json_loc = "{}/labels/{}".format(annotation_locations, annotation_file)
    # print("Retrieving labels for: " + json_loc)
    with open(json_loc) as json_file:
        data = json.load(json_file)

        # Retrieving the annotations
        annotations = data[ANNOTATIONS]
        labeled_duration = data[METADATA][DURATION]
        annotation_arr = []
        for annotation in annotations:
            annotation_type = annotation[ANNOTATION_TYPE]
            start_frame = annotation[START]
            end_frame = annotation[EXIT]
            label = annotation[LABEL_ARR][0][LABEL]
            annotation_arr.append({ANNOTATION_KEY: annotation_type,
                                   START_FRAME_KEY: start_frame,
                                   END_FRAME_KEY: end_frame,
                                   LABEL_KEY: label})

        frame_labels = {}
        duration = data[METADATA][DURATION]
        for frame in range(0, duration):
            labels = []
            # Check which labels this frame has
            for annotation in annotation_arr:
                start_frame = annotation[START_FRAME_KEY]
                end_frame = annotation[END_FRAME_KEY]
                if frame >= start_frame and frame < end_frame:
                    labels.append(annotation[LABEL_KEY])

            frame_labels[str(frame)] = labels

        return labeled_duration, frame_labels


def parse_inc_dec_labels(annotation_locations, video_name):
    """
    Labels each frame with the label from the annotion file.

    Parameters:
    annotation_locations (string): Directory where labels are
    (ex. ~/video_dataset/new-train/Alex-Labels)

    video_name (string): Video name without suffixes
    (ex. ohp0)

    Returns:
    :return: {0: Increasing, 1: Increasing, 2: Stationary}: For Frame 0 and 1 the person is increasing,
    at Frame 2 they are stationary
    """
    # Locate annotation file
    possible_files = os.listdir("{}/".format(annotation_locations))
    annotation_file = list(filter(lambda x: video_name in str(x), possible_files))[0]

    json_loc = "{}/{}".format(annotation_locations, annotation_file)
    print("Retrieving labels for: " + json_loc)
    with open(json_loc) as json_file:
        data = json.load(json_file)

        # Retrieving the annotations
        annotations = data[ANNOTATIONS]
        labeled_duration = data[METADATA][DURATION]
        annotation_arr = []
        for annotation in annotations:
            annotation_type = annotation[ANNOTATION_TYPE]
            start_frame = annotation[START]
            end_frame = annotation[EXIT]
            annotation_arr.append({ANNOTATION_KEY: annotation_type,
                                   START_FRAME_KEY: start_frame,
                                   END_FRAME_KEY: end_frame})

        frame_labels = {}
        for frame in range(0, labeled_duration):
            labels = []
            # One label per frame
            for annotation in annotation_arr:
                start_frame = annotation[START_FRAME_KEY]
                end_frame = annotation[END_FRAME_KEY]
                if frame >= start_frame and frame < end_frame:
                    current_label = annotation[ANNOTATION_KEY]
                    if current_label in ["Increasing", "Decreasing", "Stationary"]:
                        labels.append(current_label.lower())
                    else:
                        labels.append("Stationary")
                    break

            if labels == []:
                labels = ["Stationary"]

            frame_labels[str(frame)] = labels

        return labeled_duration, frame_labels
