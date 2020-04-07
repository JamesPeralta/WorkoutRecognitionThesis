import cv2
import os
from HelperFunctions.AnnotationParser import parse_inc_dec_labels
import pandas as pd
import numpy as np
import Optical_Flow as flow

# API Endpoint
URL = "http://localhost:3000/python/posenet"
headers = {'content-type': "image/jpeg"}

vid_path = "../video_dataset/new-train/"
annotation_path = "../video_dataset/new-train/James-Labels"
video_dir = os.listdir(vid_path)


for video in video_dir:
    if ".mp4" not in video:
        continue

    frame_count = 0
    frame_on = 0
    no_extension = video.split(".")[0]

    # Retrieve labels
    print("Processing: " + no_extension)
    labeled_duration, labels = parse_inc_dec_labels(os.path.abspath(annotation_path), no_extension)
    cap = cv2.VideoCapture(vid_path + video)

    modulo = 4
    if video in ["squat0.mp4", "squat1.mp4", "squat2.mp4", "squat3.mp4", "squat4.mp4", "squat5.mp4"]:
        modulo = 2

    inc_dec_label_arr = []
    vid_num = 1
    # Collect a frame right away
    status, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (300, 300))
    frame_count = 0
    frame_on = 0
    total_inc = []
    total_dec = []
    while True:
        status, next_frame = cap.read()
        if status is False:
            break
        next_frame = cv2.resize(next_frame, (300, 300))

        result, frame, total_found = flow.calculate_optical_flow(prev_frame, next_frame, next_frame)
        if result == "increasing":
            total_inc.append(total_found)
        elif result == "decreasing":
            total_dec.append(total_found)

        # Get Ground truth
        label = labels.get(str(frame_on), [''])[0]

        inc_dec_label_arr.append({
            "predicted phase": result,
            "label": label
        })

        cv2.putText(frame, result, (8, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(video, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = next_frame

        if modulo == 2:
            if frame_count % modulo == 0:
                frame_on += 1
        else:
            if frame_count % modulo != 0:
                frame_on += 1

        frame_count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # Dump what's left
    inc_dec_label_df = pd.DataFrame(inc_dec_label_arr)
    inc_dec_label_df.to_csv("./phase_detected_{}.csv".format(no_extension)) #, index=False)
    inc_dec_label_arr = []

    print("{} has avg of {} keypoints when increasing and {} keypoints when decreasing".format(no_extension, sum(total_inc) / len(total_inc), sum(total_dec) / len(total_dec)))

    total_inc = []
    total_dec = []
