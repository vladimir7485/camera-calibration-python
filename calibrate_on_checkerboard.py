import numpy as np
import cv2 as cv
import os
import glob
import time
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.interactive(True)
# import matplotlib.pyplot as plt

# pattern size
psize = (9,6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((psize[0] * psize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:psize[0], 0:psize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/iPhone13Pro'
# img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/Pixel5'

images = sorted(glob.glob(f'{img_dir}/*.[jJ][pP][gG]') + glob.glob(f'{img_dir}/*.[jJ][pP][eE][gG]') + glob.glob(f'{img_dir}/*.[pP][nN][gG]'))
# images = sorted(glob.glob(f'{img_dir}/*.dng'))
print(images)

for idx, fname in enumerate(images):
    print(f'Processing {os.path.basename(fname)} ({idx + 1}/{len(images)})', end='\r')
    
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    start_time = time.time()

    # Try detection on original image first
    ret, corners = cv.findChessboardCorners(
        gray, 
        psize, 
        # None,
        cv.CALIB_CB_ADAPTIVE_THRESH \
        + cv.CALIB_CB_NORMALIZE_IMAGE \
        + cv.CALIB_CB_FILTER_QUADS \
        # + cv.CALIB_CB_FAST_CHECK \
        # + cv.CALIB_CB_PLAIN
    )

    print(f"  Final result: {'SUCCESS' if ret else 'FAILED'}")
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, psize, corners2, ret)
        cv.putText(img, f'({idx + 1}/{len(images)})', (10, 200), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
    else:
        cv.putText(img, f'({idx + 1}/{len(images)})', (10, 200), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)

    cv.imshow('img', img)
    cv.waitKey(50)

cv.destroyWindow('img')

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread(images[1])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort with  cv.indistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_undistort.png', dst)
cv.imshow('undistort', dst)

# undistort with cv.remap
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_remap.png', dst)
cv.imshow('remap', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

cv.imshow('orig', img)
cv.waitKey()

cv.destroyAllWindows()
