import tensorflow as tf
import cv2
import time
import argparse
import statistics
import posenet
import pandas as pd
import os

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

        # These variables are used to calculate
        # the rate of change for counting reps
        count = 0
        past = None
        present = None
        try:
            up_rep = False
            down_rep = False
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

                # Create prediction display area
                overlay = cv2.imread('chat.png')
                # rows, cols, channels = overlay.shape
                # overlay = cv2.addWeighted(overlay_image[250:250 + rows, 0:0 + cols], 0.5, overlay, 0.5, 0)
                # overlay_image[250:250 + rows, 0:0 + cols] = overlay

                # Display workout type
                prediction = ""
                for i in people_location:
                    all_keypoints_detected.append([i])  # Remove
                    if past is None:
                        past = i
                    elif count % 20 == 0:
                        present = i

                        # Calculate the rate of change
                        result = posenet.rate_of_change(past, present)

                        if result == "Up":
                            up_rep = True
                        elif result == "Down":
                            down_rep = True
                        else:
                            if up_rep is True and down_rep is True:
                                rep_count += 1
                                up_rep = False
                                down_rep = False

                        # Reset variables
                        past = None
                        present = None
                        count = -1

                overlay_image = cv2.rectangle(overlay_image, (20 - 15, 380), (145 - 15, 440), (255, 255, 255), -1)
                prediction = "One-hand OHP"#posenet.detect_workout_type(i)
                cv2.putText(overlay_image, "{}".format(prediction), (8, 395 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Display rep count
                cv2.putText(overlay_image, 'Rep count: {}'.format(rep_count), (8, 420 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)
                # overlay_image = posenet.draw_coord_grid(overlay_image)
                cv2.imshow('', overlay_image)

                # Store data needed to print final summaries
                count += 1
                frame_count += 1
                keypoints_detected.append(num_keypoints_detected)
                people_detected.append(num_of_people_detected)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            print("Average number of key-points detected throughout the video: {}".format(statistics.mean(keypoints_detected)))
            missing_keypoints = len([i for i in keypoints_detected if i < 17])
            extra_keypoints = len([i for i in keypoints_detected if i > 17])
            print("{} of the {} frames are missing atleast one keypoint".format(missing_keypoints,
                                                                                len(keypoints_detected)))
            print("{} of the {} frames have atleast one extra keypoint".format(extra_keypoints,
                                                                                len(keypoints_detected)))
            print("There were at most {} person(s) detected in this video".format(max(people_detected)))

        print("The video contained {} frames".format(frame_count))
        print('Average FPS: ', frame_count / (time.time() - start))

        # writing to file
        print("Writing to file")
        df = pd.DataFrame(all_keypoints_detected, columns=['keypoints'])
        df.to_csv(args.csv_loc, header=['keypoints'])


if __name__ == "__main__":
    main()
