# crop image from 512x512 to 512x288

import cv2
import numpy as np
import os

def crop_images():

    #for dir in os.listdir("./"):

        #if not os.path.isdir(dir) and dir != "__pycache__":
        #    continue

    for i in range(16):

        path = f"{str(i).zfill(5)}.png"

        # import image
        img = cv2.imread(path)
        # cv2 to np
        #img = np.array(img)
        # np crop array
        img = img[112:400, 0:512]
        # np to cv2
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(path, img)
