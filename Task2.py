import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
UBIT = 'sahilsuh'
np.random.seed(sum([ord(c) for
c in UBIT]))
img1 = cv2.imread('tsucuba_left.png',0) # queryImage
img2 = cv2.imread('tsucuba_right.png',0) # trainImage

'''Reference: https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html'''

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#Task 2.1
image_A = cv2.drawKeypoints(img1,kp1, None)
cv2.imwrite('task2_sift1.jpg',image_A)

image_B = cv2.drawKeypoints(img2,kp2, None)
cv2.imwrite('task2_sift2.jpg',image_B)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imwrite('task2_matches_knn.jpg',img3)
plt.imshow(img3),plt.show()

pts1 = np.int32([kp1[m.queryIdx].pt for m in good])
pts2 = np.int32([kp2[m.trainIdx].pt for m in good])

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, 5)

#Task 2.2
print("Fundamental Matrix: \n",F)

#Selecting only the inlier pairs
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#For assigning same colors to epilines
ls_colors = []
for i in range(10):
    color = tuple(np.random.randint(0,255,3).tolist())
    ls_colors.append(color)

def drawlines(img1,img2,lines,pts1,pts2, index):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = ls_colors[index]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        index += 1
    return img1,img2

index = 0
pts1_1 = []
pts2_2 = []

#Task 2.3, Randomly selecting 10 inlier matches
for i in range(10):
    index = random.randint(0,min(len(pts1),len(pts2)))
    pts1_1.append(pts1[index])
    pts2_2.append(pts2[index])
    
pts1_1 = np.asarray(pts1_1)
pts2_2 = np.asarray(pts2_2)


''' For each keypoint in the right image, compute the
epiline and draw on the left image'''
lines1 = cv2.computeCorrespondEpilines(pts2_2, 2,F)
lines1 = lines1.reshape(-1,3)
img4,img5 = drawlines(img1,img2,lines1,pts1_1,pts2_2, 0)

''' For each keypoint in the left image, compute the
epiline and draw on the right image'''
lines2 = cv2.computeCorrespondEpilines(pts1_1, 1,F)
lines2 = lines2.reshape(-1,3)
img6,img7 = drawlines(img2,img1,lines2,pts2_2,pts1_1, 0)

#task 2.3
cv2.imwrite("task2_epi_left.jpg",img4)
plt.imshow(img4),plt.show()
cv2.imwrite("task2_epi_right.jpg",img6)
plt.imshow(img6),plt.show()

#Task 2.4
#Calculating the disparity map
'''Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
              https://stackoverflow.com/questions/21702945/attributeerror-module-object-has-no-attribute'''
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=27)
disparity = stereo.compute(img1,img2)

dispa = np.abs(disparity) / np.max(np.abs(disparity))
dispa = dispa*255
dispa = np.asarray(dispa, dtype = 'int32')
cv2.imwrite("task2_disparity.jpg", dispa)
plt.imshow(disparity,'gray')
plt.show()
