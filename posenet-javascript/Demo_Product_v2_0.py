import cv2
import Janus_v2_0

# API Endpoint
URL = "http://localhost:3000/python/posenet"
headers = {'content-type': "image/jpeg"}
cap = cv2.VideoCapture(1)
janus = Janus_v2_0.Janus(cap)

counting_reps = False
rep_count = 0
while(True):
    # Get and draw poses
    status, frame, num_keypoints_detected, people_location, num_of_people_detected = janus.get_poses()
    if status is False:
        break

    if (counting_reps):
        # Use PoseNet outout to detect repetitons
        rep_count = janus.count_reps(people_location)

        # Reporting workout and rep count
        frame = cv2.rectangle(frame, (20 - 15, 380 + 35), (145 - 15, 440 + 35), (255, 255, 255), -1)
        prediction = "Gymnos v2.0"  # posenet.detect_workout_type(i)
        cv2.putText(frame, "{}".format(prediction), (8, 395 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Rep count: {}'.format(rep_count), (8, 420 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key_pressed = cv2.waitKey(15)
    if key_pressed & 0xFF == ord('q'):
        break
    # Hit "s" key
    if key_pressed == 115:
        if counting_reps:
            print("Stop counting reps")
            counting_reps = False
            janus.reset_rep_count()
        else:
            print("Start counting reps")
            counting_reps = True

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()