import cv2
import sys
import os

# Create directory to store images
if not os.path.exists('processed'):
    os.mkdir('processed')

# Error file
errfile = open("processErr.txt", "w")

# Passed in all images as arg
img_names = sys.argv[1:]

for idx, img_name in enumerate(img_names):
    try:
        image = cv2.imread(img_name)
        image = cv2.resize(image, (128, 128))
        cv2.imwrite("processed/" + str(idx) + ".jpg", image)
    except:
        # Log error
        errfile.write(img_name + '\n')
        print("Error for " + img_name)


errfile.close()