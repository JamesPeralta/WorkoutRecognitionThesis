import tensorflow as tf
import cv2
import time
import argparse
import statistics
import sys
import posenet
import pandas as pd
import os

# CONSTANTS
MIN_POSE_SCORE = 0.40
MIN_KEYPOINT_SCORE = 0

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--csv_loc', type=str, default=None, help="Location of the csv you want to create")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        rep_count = 0
        keypoints_detected = []
        people_detected = []
        all_keypoints_detected = []
        try:
            while True:
                # Read image
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride)

                # Retrieve the heatmaps from the image
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                # Decode the heatmaps into poses and keypoints
                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=MIN_POSE_SCORE)

                # Draw the poses onto the image
                keypoint_coords *= output_scale
                overlay_image, num_keypoints_detected, people_location, num_of_people_detected = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=MIN_POSE_SCORE, min_part_score=MIN_KEYPOINT_SCORE)

                # Store data needed to print final summaries
                frame_count += 1
                keypoints_detected.append(num_keypoints_detected)
                people_detected.append(num_of_people_detected)
                for i in people_location:
                    all_keypoints_detected.append([i, "Overhead Press"])

                cv2.putText(overlay_image, 'Rep count: {}'.format(rep_count), (53, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('', overlay_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            print("Pose accuracy was an average of {}".format(statistics.mean(keypoints_detected)))
            missing_keypoints = len([i for i in keypoints_detected if i < 17])
            extra_keypoints = len([i for i in keypoints_detected if i > 17])
            print("{} of the {} frames are missing atleast one keypoint".format(missing_keypoints,
                                                                                len(keypoints_detected)))
            print("{} of the {} frames have atleast one extra keypoint".format(extra_keypoints,
                                                                                len(keypoints_detected)))
            print("There were at most {} person(s) detected in this video".format(max(people_detected)))

        print('Average FPS: ', frame_count / (time.time() - start))

        # Create the pandas DataFrame
        df = pd.DataFrame(all_keypoints_detected, columns=['keypoints', 'label'])
        # Write all of the keypoints into a csv
        # if file does not exist write header
        if not os.path.isfile('filename.csv'):
            df.to_csv(args.csv_loc, header=['keypoints', 'label'])
        else:  # else it exists so append without writing the header
            df.to_csv(args.csv_loc, mode='a', header=False)


if __name__ == "__main__":
    main()

