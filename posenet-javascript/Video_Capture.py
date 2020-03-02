import cv2

cap = cv2.VideoCapture(0)


video_size = (500, 500)
i = 0
out = None
capturing_video = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Prepare frame
    frame = cv2.resize(frame, video_size)

    # If capturing the video
    if capturing_video:
        out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    key_pressed = cv2.waitKey(1)
    if key_pressed & 0xFF == ord('q'):
        break
    # Hit "s" key
    if key_pressed == 115:
        if capturing_video:
            print("Stop capturing video")
            capturing_video = False
            i += 1
        else:
            print("Start capturing video")
            capturing_video = True
            video_name = str(input("Enter exercise type: "))
            out = cv2.VideoWriter("{}_{}.mp4".format(video_name, i), cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (500, 500))


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()