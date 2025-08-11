import numpy as np
import cv2 as cv
import os
import glob
import time
import rawpy
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.interactive(True)
import matplotlib.pyplot as plt

def create_checkerboard_pattern(width=8, height=6, square_size=100, filename='checkerboard.png'):
    """Create a high-quality checkerboard pattern for display"""
    # Create image with white background
    img = np.ones((height * square_size, width * square_size), dtype=np.uint8) * 255
    
    # Fill with black squares
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 0
    
    # Save the pattern
    cv.imwrite(filename, img)
    print(f"Created checkerboard pattern: {filename}")
    print(f"Pattern size: {width}x{height} squares, {square_size}x{square_size} pixels each")
    return img

# Create a high-quality checkerboard pattern for testing
# Uncomment the next line to generate a pattern
create_checkerboard_pattern(8, 6, 100, 'checkerboard_8x6.png')
# exit()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/OpenCVCheckerboard'
# psize = (7,6)
# img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/Checkerboard'
# img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/iPhone13Pro'
img_dir = '/Users/vladimirkurmanov/Documents/Work/Visages/CalibData/Pixel5'
psize = (8,6)
# images = sorted(glob.glob(f'{img_dir}/*.[jJ][pP][gG]') + glob.glob(f'{img_dir}/*.[jJ][pP][eE][gG]') + glob.glob(f'{img_dir}/*.[pP][nN][gG]'))
images = sorted(glob.glob(f'{img_dir}/*.dng'))
print(images)

for idx, fname in enumerate(images):
    print(f'Processing {os.path.basename(fname)} ({idx + 1}/{len(images)})', end='\r')
    
    # img = cv.imread(fname)
    raw = rawpy.imread(fname)
    img = raw.postprocess()
    img = img[368:2304, 670:3260]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite(f'{img_dir}/gray_{idx}.png', gray)

    cv.imshow('orig', gray)
    cv.waitKey()

    # Try different preprocessing to improve detection
    # 1. Normalize the image
    gray_norm = cv.equalizeHist(gray)
    
    # 2. Try to reduce noise
    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Try adaptive thresholding
    gray_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    
    start_time = time.time()

    # Try detection on original image first
    ret, corners = cv.findChessboardCorners(gray, psize, None)

    print(f"  Time taken: {time.time() - start_time:.2f} seconds")
    
    # If not found, try normalized image
    if False and not ret:
        ret, corners = cv.findChessboardCorners(gray_norm, psize, None)
        print(f"  Trying normalized image: {'SUCCESS' if ret else 'FAILED'}")
    
    # If still not found, try blurred image
    if False and not ret:
        ret, corners = cv.findChessboardCorners(gray_blur, psize, None)
        print(f"  Trying blurred image: {'SUCCESS' if ret else 'FAILED'}")
    
    # If still not found, try thresholded image
    if False and not ret:
        ret, corners = cv.findChessboardCorners(gray_thresh, psize, None)
        print(f"  Trying thresholded image: {'SUCCESS' if ret else 'FAILED'}")
    
    # If still not found, try with different flags
    if False and not ret:
        ret, corners = cv.findChessboardCorners(gray, psize, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        print(f"  Trying with adaptive flags: {'SUCCESS' if ret else 'FAILED'}")
    
    print(f"  Final result: {'SUCCESS' if ret else 'FAILED'}")
    
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
    cv.waitKey()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread(f'{img_dir}/left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort with  cv.indistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_undistort.png', dst)

# undistort with cv.remap
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult_remap.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )

cv.destroyAllWindows()

print("\n" + "="*60)
print("TIPS FOR BETTER CHECKERBOARD DETECTION:")
print("="*60)
print("1. Use a high-contrast pattern (black squares on white background)")
print("2. Ensure even lighting - avoid shadows and glare")
print("3. Keep the pattern flat and undistorted")
print("4. Make sure the entire pattern is visible in the image")
print("5. Try different camera angles and distances")
print("6. Use a printed pattern instead of a screen display if possible")
print("7. Ensure the pattern size matches your code (currently 8x6)")
print("8. Check that your camera is perpendicular to the pattern")
print("9. Try the generated checkerboard pattern by uncommenting the create_checkerboard_pattern line")
print("="*60)