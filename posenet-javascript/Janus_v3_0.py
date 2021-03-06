import cv2
import configparser
import requests
import json
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
from collections import deque

# API Endpoint
URL = "http://localhost:3000/python/posenet"
headers = {'content-type': "image/jpeg"}

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

workout_map = {
    "0": "nothing",
    "1": "ohp",
    "2": "squat"
}

radius = 3
color = (255, 0, 0)
thickness = 3

class Janus:
    def __init__(self, stream):
        self.cap = stream

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
        # self.all_keypoints_detected = []

        # Model helpers
        self.model = load_model('./Jahnus_v3_0_Configs/jahnus_v1_0.h5')
        self.scaler = load('./Jahnus_v3_0_Configs/standard_scaler_jahnus_v1_0.save')
        self.detection_queue = deque()

    def get_poses(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        if ret is False:
            return False, None, None, None, None

        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        height = int(config.get("VideoSize", "Height"))
        width = int(config.get("VideoSize", "Width"))
        frame = cv2.resize(frame, (height, width))
        en_ret, en_frame = cv2.imencode('.jpg', frame)

        keypoints, people_location = self.get_keypoints(en_frame)

        frame = self.draw_coordinates(keypoints, frame)

        # TODO: ACTUALLY IMPLEMENT THIS VARIABLES
        num_keypoints_detected = 16
        num_of_people_detected = 1

        return ret, frame, num_keypoints_detected, people_location, num_of_people_detected

    def detect_workout(self, keypoints):
        flattened_keypoints = keypoints.flatten().reshape(1, -1)
        scaled_keypoints = self.scaler.transform(flattened_keypoints)

        pred = self.model.predict(scaled_keypoints)
        pred = np.argmax(pred, axis=1)[0]

        self.detection_queue.append(pred)
        if len(self.detection_queue) < 10:
            return "LOADING MODEL"
        self.detection_queue.popleft()

        # Get Max of the last 10 predictions
        last_detections = np.array(list(self.detection_queue))
        pred = np.bincount(last_detections)
        pred = str(np.argmax(pred))
        label = workout_map[pred]

        return label

    def count_reps(self, people_location):
        if self.past is None:
            self.past = people_location
        elif self.count % self.roc_sampling == 0:
            present = people_location

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
            past = past_keypoints[index][1]
            present = present_keypoints[index][1]

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

    def get_keypoints(self, img):
        r = requests.post(url=URL, data=img.tostring(), headers=headers)
        json_file = json.loads(r.text)

        # Get num of keypoints
        body_point_map = {}
        people_location = []
        for i in json_file[0]['keypoints']:
            part = i["part"]
            x_coord = i["position"]["x"]
            y_coord = i["position"]["y"]

            body_point_map[part] = (x_coord, y_coord)
            people_location.append([x_coord, y_coord])

        return body_point_map, people_location

    def draw_coordinates(self, keypoints, img):
        # Draw a circle with blue line borders of thickness of 2 px
        for i in keypoints.keys():
            x_coord = int(keypoints[i][0])
            y_coord = int(keypoints[i][1])
            img = cv2.circle(img, (x_coord, y_coord), radius, color, thickness)

        return img

    def get_model_config(self):
        response = requests.get(url=URL + "/modelconfig")
        response_json = json.loads(response.text)["config"]

        return response_json

    def reset_rep_count(self):
        self.count = 0
        self.up_rep = False
        self.down_rep = False
        self.rep_count = 0
