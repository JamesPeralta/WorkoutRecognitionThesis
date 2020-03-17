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


def parse_annotation_file(annotation_locations, video_name):
    """
    Labels each from with the label from the annotion file.

    :return: {0: Squat, 1: Squat, 2: Nothing}: For Frame 0 and 1 the person is squating,
    at Frame 2 they are doing nothing
    """
    # Locate annotation file
    possible_files = os.listdir("{}/labels".format(annotation_locations))
    annotation_file = list(filter(lambda x: video_name in str(x), possible_files))[0]

    with open("{}/labels/{}".format(annotation_locations, annotation_file)) as json_file:
        data = json.load(json_file)

        # Retrieving the annotations
        annotations = data[ANNOTATIONS]
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

        return frame_labels
