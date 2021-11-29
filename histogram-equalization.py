import numpy as np
import cv2
import os


def histEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


dir_path = os.path.dirname(os.path.realpath(__file__))

image1 = cv2.imread(dir_path + '/input/input1.jpg')
image2 = cv2.imread(dir_path + '/input/input2.jpg')

'''gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Two input images use same histogram from one image
eq1 = cv2.equalizeHist(gray1)
eq2 = cv2.equalizeHist(gray2)'''

eq1 = histEqulColor(image1)
eq2 = histEqulColor(image2)

cv2.imwrite(dir_path + '/output/output1.png', np.hstack([eq1, eq2]))

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()