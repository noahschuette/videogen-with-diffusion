# crop image from 512x512 to 512x288

import cv2
import numpy as np
def crop_images():

    for i in range(16):

        # import image
        img = cv2.imread(str(i).zfill(5) + ".png")
        # cv2 to np
        #img = np.array(img)
        # np crop array
        img = img[112:400, 0:512]
        # np to cv2
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(str(i).zfill(5) + "_.png", img)
