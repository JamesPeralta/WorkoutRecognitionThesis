import cv2

cascade = cv2.CascadeClassifier('frontal_face.xml')


def find_and_blur(bw_image, orig_image):
    # detect all faces
    faces = cascade.detectMultiScale(bw_image, 1.1, 4)
    # get the location of all the faces

    for (x, y, w, h) in faces:
        # select the areas where the face was found
        roi_color = orig_image[y:y+h, x:x+w]

        # blur the colored image
        blur = cv2.GaussianBlur(roi_color, (101,101), 0)

        # Insert the ROI back into the image
        orig_image[y:y+h, x:x+w] = blur

    return orig_image

cap = cv2.VideoCapture(0)

while True:
    # get last recorded frame
    _, color = cap.read()

    # transform color -> grayscale
    bw = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # detect the face and blur it
    blur = find_and_blur(bw, color)
    # display output
    cv2.imshow('Video', blur)
    # break if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# turn camera off
cap.release()
# close camera  window
cv2.destroyAllWindows()