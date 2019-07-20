import cv2
import glob
import os
from sklearn.model_selection import train_test_split


class Load:
    @staticmethod
    def load_data():
        tofilename = os.path.realpath(__file__)
        pathname = "/".join((tofilename.split('/')[:-1])) # Path to current directory

        # Load all images
        X = [cv2.imread(file) for file in glob.glob(pathname + "/images/processed/*.jpg")]
        
        # Labels (all real)
        y = [1] * len(X)

        # Randomly sort and split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        return ((X_train, y_train), (X_test, y_test))