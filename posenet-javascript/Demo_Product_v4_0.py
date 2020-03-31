import cv2
import Janus_v4_0
import numpy as np

# API Endpoint
URL = "http://localhost:3000/python/posenet"
headers = {'content-type': "image/jpeg"}
cap = cv2.VideoCapture("./ohp6.mp4")
# cap = cv2.VideoCapture(1)
janus = Janus_v4_0.Janus(cap)

counting_reps = False
rep_count = 0

run_loop = True
# Collect a frame right away
status, frame = janus.get_frame()
if status is False:
    run_loop = False
else:
    janus.set_prev_frame(frame)

while run_loop:
    # Get and draw poses
    status, frame = janus.get_frame()

    raw_frame, frame, num_keypoints_detected, people_location, num_of_people_detected = janus.get_poses()
    # Scale up to display size
    people_location_as_numpy = np.array(people_location) * 1.66666667
    people_location = people_location_as_numpy.tolist()

    if status is False:
        break

    frame = cv2.resize(frame, (500, 500)) # Easier to see

    # Detect workout
    workout_detected = janus.detect_workout(people_location_as_numpy)
    cv2.putText(frame, "{}".format(workout_detected), (30, 75), cv2.FONT_HERSHEY_SIMPLEX,  # (8, 395 + 40)
                3, (255, 0, 0), 3, cv2.LINE_AA)

    if workout_detected != "nothing":
        # Use PoseNet outout to detect repetitons
        rep_count, frame = janus.count_reps(raw_frame, frame)

        # Reporting workout and rep count
        cv2.putText(frame, 'Rep count: {}'.format(rep_count), (8, 420 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 3, cv2.LINE_AA)
    else:
        janus.reset_rep_count()

    cv2.imshow('frame', frame)

    key_pressed = cv2.waitKey(15)
    if key_pressed & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()