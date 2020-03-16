import cv2
import time
import argparse
import statistics
import pandas as pd
import configparser
import re
import os
import Janus_v2_0

# Load in configuration
config = configparser.ConfigParser()
config.read('configurations.cfg')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--labels', type=str, default=None, help="Give the location of your labels file")
parser.add_argument('--csv_loc', type=str, default=None)
args = parser.parse_args()

video = "../video_dataset/" + args.file

stop_program = False
statistics_df = []


def main():
    name_search = re.search("^.*/(.+)\\.", video)
    file_name = name_search.group(1)

    if video is not None:
        cap = cv2.VideoCapture(video)
    else:
        cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    janus = Janus_v2_0.Janus(cap)

    start = time.time()
    frame_count = 0
    rep_count = 0
    keypoints_detected = []
    people_detected = []
    all_keypoints_detected = []
    try:
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
                return False

    except Exception as e:
        print(e)

    print("_________ {} Statistics _________".format(file_name))
    # Report all of the statistics
    print("Average number of key-points detected throughout the video: {}".format(statistics.mean(keypoints_detected)))
    missing_keypoints = len([i for i in keypoints_detected if i < 17])
    extra_keypoints = len([i for i in keypoints_detected if i > 17])
    print("{} of the {} frames are missing atleast one keypoint".format(missing_keypoints,
                                                                        len(keypoints_detected)))
    print("{} of the {} frames have atleast one extra keypoint".format(extra_keypoints,
                                                                        len(keypoints_detected)))
    print("There were at most {} person(s) detected in this video".format(max(people_detected)))
    print("The video contained {} frames".format(frame_count))
    print('Average FPS: ', int(frame_count / (time.time() - start)))

    labels = pd.read_csv(args.labels)
    video_labels = labels.loc[labels["file_name"] == file_name]
    actual_reps = video_labels["reps"].values
    if len(actual_reps) > 0:
        rep_search = re.search('(^\\d+)', actual_reps[0])
        actual_reps = rep_search.group(1)
    print("The expected amount of reps was:", actual_reps)
    print("The predicted rep count is {}".format(rep_count))

    # TODO: Compare workout prediction vs actual

    # writing to file
    # print("Writing to file")
    # df = pd.DataFrame(all_keypoints_detected, columns=['keypoints'])
    # df.to_csv(args.csv_loc, header=['keypoints'])

    model_config = janus.get_model_config()
    stats_dict = {"file_name": file_name ,
                  "fps": int(frame_count / (time.time() - start)),
                  "expected_reps": actual_reps,
                  "predicted_reps": rep_count}
    for param in model_config.keys():
        stats_dict[param] = str(model_config[param])

    statistics_df.append(stats_dict)

    return True


if __name__ == "__main__":
    label_loc = "../video_dataset/test/"
    for i in os.listdir(label_loc):
        if ".mp4" in i:
            video = label_loc + i
            if main() is False:
                break
            break

    eval_df = pd.DataFrame(statistics_df)
    eval_df.to_csv("./eval_results_2.csv", index=False)
