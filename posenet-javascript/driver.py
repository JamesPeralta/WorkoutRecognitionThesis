import cv2
import requests
import json

# API Endpoint
URL = "http://localhost:3000/python/posenet"
cap = cv2.VideoCapture("./squat0.mp4")


radius = 3
color = (255, 0, 0)
thickness = 3


def get_keypoints(img):
    r = requests.post(url=URL, json={"image": img.tolist()})
    json_file = json.loads(r.text)

    body_point_map = {}
    for i in json_file[0]['keypoints']:
        part = i["part"]
        x_coord = i["position"]["x"]
        y_coord = i["position"]["y"]

        body_point_map[part] = (x_coord, y_coord)

    return body_point_map


def draw_coordinates(keypoints, img):
    # Draw a circle with blue line borders of thickness of 2 px
    for i in keypoints.keys():
        x_coord = int(keypoints[i][0])
        y_coord = int(keypoints[i][1])
        img = cv2.circle(img, (x_coord, y_coord), radius, color, thickness)

    return img


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (257, 300))
    # frame = frame / 255

    keypoints = get_keypoints(frame)
    frame = draw_coordinates(keypoints, frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()