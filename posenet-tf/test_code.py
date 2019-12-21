import cv2
import numpy as np

from skimage import data, img_as_float
from skimage import exposure

img = cv2.imread("./images/james.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# # create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
#
# print(img.dtype)
cv2.imshow("James", cl1)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
