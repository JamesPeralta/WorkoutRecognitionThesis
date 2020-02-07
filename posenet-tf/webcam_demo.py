import cv2
import time
import argparse
import statistics
import pandas as pd
import configparser
import os
import Janus_v1_0

# Load in configuration
config = configparser.ConfigParser()
config.read('configurations.cfg')

# CONSTANTS
MIN_POSE_SCORE = 0.45
MIN_KEYPOINT_SCORE = 0

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--csv_loc', type=str, default=None)
args = parser.parse_args()


def main():

    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    janus = Janus_v1_0.Janus(cap)

    start = time.time()
    frame_count = 0
    rep_count = 0
    keypoints_detected = []
    people_detected = []
    all_keypoints_detected = []

    try:
        up_rep = False
        down_rep = False
        while True:

            # Run PoseNet
            overlay_image, num_keypoints_detected, people_location, num_of_people_detected = janus.get_poses()

            # Use PoseNet outout to detect repetitons
            rep_count = janus.count_reps(people_location)

            # Reporting workout and rep count
            overlay_image = cv2.rectangle(overlay_image, (20 - 15, 380 + 35), (145 - 15, 440 + 35), (255, 255, 255), -1)
            prediction = "Lateral Raises"#posenet.detect_workout_type(i)
            cv2.putText(overlay_image, "{}".format(prediction), (8, 395 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(overlay_image, 'Rep count: {}'.format(rep_count), (8, 420 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('', overlay_image)

            # Store data needed to print final summaries
            frame_count += 1
            keypoints_detected.append(num_keypoints_detected)
            people_detected.append(num_of_people_detected)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)

    # Report all of the statistics
    print("Average number of key-points detected throughout the video: {}".format(statistics.mean(keypoints_detected)))
    missing_keypoints = len([i for i in keypoints_detected if i < 17])
    extra_keypoints = len([i for i in keypoints_detected if i > 17])
    print("{} of the {} frames are missing atleast one keypoint".format(missing_keypoints,
                                                                        len(keypoints_detected)))
    print("{} of the {} frames have atleast one extra keypoint".format(extra_keypoints,
                                                                        len(keypoints_detected)))
    print("There were at most {} person(s) detected in this video".format(max(people_detected)))
    print("Final rep count is {}".format(rep_count))
    print("The video contained {} frames".format(frame_count))
    print('Average FPS: ', int(frame_count / (time.time() - start)))

    # TODO: Compare rep count vs actual
    # TODO: Compare rep count vs actual

    # writing to file
    print("Writing to file")
    df = pd.DataFrame(all_keypoints_detected, columns=['keypoints'])
    df.to_csv(args.csv_loc, header=['keypoints'])


if __name__ == "__main__":
    main()
