import cv2 as cv
import sys
img = cv.imread(cv.samples.findFile("darren_squat.png"))
if img is None:
    sys.exit("Could not read the image.")
hsv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.HOGDescriptor()

cv.imshow("Display window", hsv)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("starry_night.png", img)