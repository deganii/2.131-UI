import cv2
import numpy as np
import cv2 as cv
#import cellPredictor

import numpy as np
import cv2 as cv
import video
import sys

mser = cv.MSER()
img = cv.imread('image1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

regions = mser.detect(gray, None)

#polylines
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv.polylines(vis, hulls, 1, (0, 255, 0))


#boundingboxes
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
mask = cv2.dilate(mask, np.ones((150,150), np.uint8))

for contour in hulls:
	cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

for i, contour in enumerate(hulls):
	x,y,w,h = cv2.boundingRect(contour)
	print(x,y,w,h) #print coordinates and dimensions of bounding boxes


cv2.imshow('img', vis)
cv2.waitKey(0)
