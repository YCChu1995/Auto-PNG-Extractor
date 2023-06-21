import cv2 as cv
import numpy as np

image = cv.imread('./04.Dataset/1.png', cv.IMREAD_UNCHANGED)
print(image.shape)

cv.imwrite('2.png', image)
image = cv.imread('2.png', cv.IMREAD_UNCHANGED)
alpha = image[:,:,3]
print(alpha.shape)
# print(np.unique(image[:,:,3]))