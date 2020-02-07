import tensorflow as tf
import posenet
import cv2
import configparser

# Load in configuration
config = configparser.ConfigParser()
config.read('configurations.cfg')

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

# CONSTANTS
MIN_POSE_SCORE = float(config.get("Algorithm", "Min_Pose_Score"))
MIN_KEYPOINT_SCORE = float(config.get("Algorithm", "Min_Keypoint_Score"))


class Janus:
    def __init__(self, stream, model=101, scale_factor=0.7125):
        self.sess = tf.compat.v1.Session()
        self.model_cfg, self.model_outputs = posenet.load_model(model, self.sess)
        self.output_stride = self.model_cfg['output_stride']
        self.cap = stream
        self.scale_factor = scale_factor

        # These variables are used to calculate
        # the rate of change for counting reps
        self.count = 0
        self.past = None
        self.roc_sampling = int(config.get("Algorithm", "ROC_Sampling"))
        self.up_rep = False
        self.down_rep = False
        self.rep_count = 0

        # Remebering what Janus has seen so far
        self.keypoints_detected = []
        self.people_detected = []
        self.all_keypoints_detected = []

    def get_poses(self):
        # Read image
        input_image, display_image, output_scale = posenet.read_cap(
            self.cap, scale_factor=self.scale_factor, output_stride=self.output_stride)

        # Retrieve the heatmaps from the image
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
            self.model_outputs,
            feed_dict={'image:0': input_image}
        )

        # Decode the heatmaps into poses and keypoints
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=self.output_stride,
            max_pose_detections=10,
            min_pose_score=MIN_POSE_SCORE)

        # Draw the poses onto the image
        keypoint_coords *= output_scale
        overlay_image, num_keypoints_detected, people_location, num_of_people_detected = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=MIN_POSE_SCORE, min_part_score=MIN_KEYPOINT_SCORE)

        return overlay_image, num_keypoints_detected, people_location, num_of_people_detected

    def count_reps(self, people_location):
        for i in people_location:
            self.all_keypoints_detected.append([i])
            if self.past is None:
                self.past = i
            elif self.count % self.roc_sampling == 0:
                present = i

                # Calculate the rate of change
                result = self.rate_of_change(self.past, present)

                if result == "Up":
                    self.up_rep = True
                elif result == "Down":
                    self.down_rep = True
                else:
                    if self.up_rep is True and self.down_rep is True:
                        self.rep_count += 1
                        self.up_rep = False
                        self.down_rep = False

                # Reset variables
                self.past = None
                self.count = -1

        self.count += 1

        return self.rep_count

    def rate_of_change(self, past_keypoints, present_keypoints):
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

