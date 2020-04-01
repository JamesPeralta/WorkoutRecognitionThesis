import cv2
import configparser
import requests
import json
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
from collections import deque
import Optical_Flow as flow

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

# Store it front and back to allow reverse look ups
rep_phase_map = {
    "0": "stationary",
    "1": "increasing",
    "2": "decreasing",
    "stationary": "0",
    "increasing": "1",
    "decreasing": "2"
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
        self.increasing = False
        self.decreasing = False
        self.rep_count = 0

        # Remebering what Janus has seen so far
        self.keypoints_detected = []
        self.people_detected = []
        # self.all_keypoints_detected = []

        # Model helpers
        self.model = load_model('./Jahnus_v3_0_Configs/jahnus_v1_0.h5')
        self.scaler = load('./Jahnus_v3_0_Configs/standard_scaler_jahnus_v1_0.save')
        self.detection_queue = deque()
        self.rep_state_queue = deque()
        self.last_phase = "stationary"

        # Variables for resizing video
        self.vid_size = (int(config.get("VideoSize", "Height")), int(config.get("VideoSize", "Width")))
        self.prev_frame = None

    def get_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, self.vid_size)

        if ret is False:
            return False, None

        return ret, frame

    def set_prev_frame(self, frame):
        self.prev_frame = frame

    def get_poses(self):
        # Capture frame-by-frame
        ret, raw_frame = self.get_frame()

        if ret is False:
            return None, None, None

        en_ret, en_frame = cv2.imencode('.jpg', raw_frame)

        keypoints, people_location = self.get_keypoints(en_frame)

        frame = self.draw_coordinates(keypoints, raw_frame)

        # TODO: ACTUALLY IMPLEMENT THIS VARIABLES
        num_keypoints_detected = 16
        num_of_people_detected = 1

        return raw_frame, frame, num_keypoints_detected, people_location, num_of_people_detected

    def detect_workout(self, keypoints):
        flattened_keypoints = keypoints.flatten().reshape(1, -1)
        scaled_keypoints = self.scaler.transform(flattened_keypoints)

        pred = self.model.predict(scaled_keypoints)
        pred = np.argmax(pred, axis=1)[0]

        self.detection_queue.append(pred)
        if len(self.detection_queue) < 10:
            print("INITIALIZING WORKOUT DETECTION")
            return "nothing"

        self.detection_queue.popleft()

        # Get Max of the last 10 predictions
        last_detections = np.array(list(self.detection_queue))
        pred = np.bincount(last_detections)
        pred = str(np.argmax(pred))
        label = workout_map[pred]

        return label

    def count_reps(self, next_frame, frame):
        # Calculate the rate of change
        result, frame = flow.calculate_optical_flow(self.prev_frame, next_frame, frame)

        self.rep_state_queue.append(rep_phase_map[result])
        if len(self.rep_state_queue) < 6:
            print("INITIALIZING REP COUNT")
            return self.rep_count, frame

        self.rep_state_queue.popleft()

        # Get Max of the last 10 rep predictions
        last_detections = np.array(list(map(lambda x: int(x), list(self.rep_state_queue))))
        pred = np.bincount(last_detections)
        pred = rep_phase_map[str(np.argmax(pred))]

        if self.last_phase != pred:
            if self.increasing is True and self.decreasing is True:
                self.rep_count += 1
                self.increasing = False
                self.decreasing = False

                self.last_phase = pred
                self.set_prev_frame(next_frame)

                return self.rep_count, frame

        if pred == "increasing":
            self.increasing = True
        elif pred == "decreasing":
            self.decreasing = True

        self.last_phase = pred
        self.set_prev_frame(next_frame)

        return self.rep_count, frame

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
        self.increasing = False
        self.decreasing = False
        self.rep_count = 0
