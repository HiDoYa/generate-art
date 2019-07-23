import cv2
import glob
import os


class Load:
    @staticmethod
    def load_data():
        tofilename = os.path.realpath(__file__)
        pathname = "/".join((tofilename.split('/')[:-1])) # Path to current directory

        # Load all images
        X = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/*.jpg")]
        
        return X