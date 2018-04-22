import cv2
import numpy as np
import cv2 as cv
#import cellPredictor

import numpy as np
import cv2 as cv
import video
import sys

if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = video.create_capture(video_src)
    mser = cv.MSER()   
# mser = cv.MSER_create()

    while True:
        ret, img = cam.read()
	#img = cv2.imdecode(img, cv.CV_LOAD_IMAGE_GRAYSCALE)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        vis = img.copy()
	
	regions = mser.detect(gray, None)
        hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv.polylines(vis, hulls, 1, (0, 255, 0))
	#cv2.namedWindow('img', cv.WINDOW_NORMAL)
	#cv2.resizeWindow('img', 600, 600)
        cv.imshow('img', vis)
        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()


