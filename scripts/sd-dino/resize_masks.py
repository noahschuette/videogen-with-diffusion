# resize all images in folder masks
import os
import cv2

path = './masks/'
path_save = './masks_resized/'

#makedir path_save
if not os.path.exists(path_save):
    os.makedirs(path_save)

files = os.listdir(path)
c = 0
for file in files:
    img = cv2.imread(path + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    c += 1
    cv2.imwrite(path_save + str(c).zfill(3) + ".png", img)