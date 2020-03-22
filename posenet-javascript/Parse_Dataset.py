import cv2
import Janus_v2_0
import os
from HelperFunctions.AnnotationParser import parse_annotation_file
import pandas as pd

# API Endpoint
URL = "http://localhost:3000/python/posenet"
headers = {'content-type': "image/jpeg"}

dir_path = "../video_dataset/new-train/"
video_dir = os.listdir(dir_path)

completed_videos = ['squat4.mp4', 'squat5.mp4', 'squat7.mp4', 'ohp8.mp4', 'squat6.mp4', 'squat2.mp4', 'nothing1.mp4',
                    'nothing0.mp4', 'squat3.mp4', 'squat1.mp4', 'nothing2.mp4', 'squat0.mp4', 'ohp3.mp4', 'ohp2.mp4',
                    'ohp0.mp4', 'ohp1.mp4', 'ohp5.mp4', 'ohp4.mp4', 'squat8.mp4']

for video in video_dir:
    if ".mp4" not in video:
        continue

    if video in completed_videos:
        continue

    frame_count = 0
    frame_on = 0
    no_extension = video.split(".")[0]

    # Retrieve labels
    print(no_extension)
    labeled_duration, labels = parse_annotation_file(os.path.abspath(dir_path), no_extension)

    cap = cv2.VideoCapture(dir_path + video)
    janus = Janus_v2_0.Janus(cap)
    # While there are frames in the video

    modulo = 4
    if video in ["squat0.mp4", "squat1.mp4", "squat2.mp4", "squat3.mp4", "squat4.mp4", "squat5.mp4"]:
        modulo = 2

    keypoint_label_arr = []
    vid_num = 1
    while(True):
        status, frame, num_keypoints_detected, people_location, num_of_people_detected = janus.get_poses()

        # Get Label
        label = labels.get(str(frame_on), [''])[0]
        if status is False:
            break

        keypoint_label_arr.append({
            "keypoints": people_location,
            "label": label
        })

        frame = cv2.resize(frame, (500, 500))
        cv2.putText(frame, label, (8, 395 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(video, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if modulo == 2:
            if frame_count % modulo == 0:
                frame_on += 1
        else:
            if frame_count % modulo != 0:
                frame_on += 1

        if frame_count % 150 == 0:
            # Every 150 frames dump this array to avoid memory issues
            keypoint_label_df = pd.DataFrame(keypoint_label_arr)
            keypoint_label_df.to_csv("./{}_{}.csv".format(no_extension, vid_num), index=False)
            keypoint_label_arr = []
            vid_num += 1

        frame_count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    completed_videos.append(video)

    # Dump what's left
    keypoint_label_df = pd.DataFrame(keypoint_label_arr)
    keypoint_label_df.to_csv("./{}_{}.csv".format(no_extension, vid_num), index=False)
    keypoint_label_arr = []
    vid_num += 1

    # print("Actual Frames: " + str(frame_on))
    # print("Labeled Duration: " + str(labeled_duration))
    # print("Actual:Labelled: " + str(frame_on / labeled_duration))
    # print("Video length:  " + str(future - now))
    print(completed_videos)
    print("___________________________________")
