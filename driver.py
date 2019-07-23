from dcgan_model import Models
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from load import Load
from dcgan_model import Models
from sklearn.utils import shuffle
import numpy as np
import argparse
import cv2
import os
import math


def all_images(images):
    images = (images * 127.5 + 127.5).astype(np.uint8)

    # Get dimensions
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images.shape[1:4] # for height, width, channel
    image = np.zeros((height * shape[0], width * shape[1], shape[2]))

    for index, img in enumerate(images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(Optional) Whether to save model to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1, help="(Optional) Whether to load model from disk")
ap.add_argument("-t", "--train-model", type=int, default=-1, help="(Optional) Whether to train a previous model.")
ap.add_argument("-w", "--weights", type=str, help="(Optional) Path of weights file")
args = vars(ap.parse_args())

gWeightsPath = "output/" + args["weights"] + ".gener" + ".hdf5"
dWeightsPath = "output/" + args["weights"] + ".discrim" + ".hdf5"

# Load data
_X = Load.load_data()

# Convert to np arrays
X = np.array(_X)

# Scale data to [-1, 1] range
X = (X.astype("float32") - 127.5) / 127.5

# Initialize optimizer and model
print("Loading model...")
loadPath = args["load_model"] > 0 or args["train_model"] > 0
opt = SGD(lr=0.005, momentum=0.9, nesterov=True)
modelG = Models.buildGenerator(
    vectSize=100,
    weightsPath=gWeightsPath if loadPath else None)
modelD = Models.buildDiscriminator(
    numChannels=3, imgRows=128, imgCols=128,
    weightsPath=dWeightsPath if loadPath else None)

# Generator
modelG.compile(loss="binary_crossentropy", optimizer="SGD")

# Discriminator
modelD.compile(loss="binary_crossentropy", optimizer=opt)

# Discriminator on top of generator
modelGD = Sequential()
modelGD.add(modelG)
modelD.trainable = False
modelGD.add(modelD)
modelGD.compile(loss="binary_crossentropy", optimizer=opt)

# Train (only if not loading model)
batchSize = 64
epochs = 1000
noise_for_images = np.random.uniform(-1, 1, size=(batchSize, 100))
train = args["load_model"] < 0
if train:
    for epoch in range(epochs):
        numBatches = int(X.shape[0] / batchSize)
        for index in range(numBatches): 
            # Output generated images every epoch
            if index == 0:
                image = all_images(modelG.predict(noise_for_images))
                cv2.imwrite("images/art/" + str(epoch) + ".jpg", image)

                if not os.path.exists("output"):
                    os.mkdir("output")
                modelD.save_weights(dWeightsPath)
                modelG.save_weights(gWeightsPath)

            # Random noise of vector 100 (batchSize number of them)
            noiseD = np.random.uniform(-1, 1, size=(batchSize, 100))

            # Select current batch of real images and generate fkae images
            # Note: y_current is just a vectors of 1s
            X_real = X[index * batchSize:(index + 1) * batchSize]
            X_fake = modelG.predict_on_batch(noiseD)
            y_real = np.random.uniform(0.9, 1, size=batchSize)
            y_fake = np.random.uniform(0, 0.1, size=batchSize)

            # Add X_train and y_train
            _X_current = np.concatenate([X_real, X_fake])
            _y_current = np.concatenate([y_real, y_fake])

            # Shuffle randomly
            X_current, y_current = shuffle(_X_current, _y_current)

            # Feed into discriminator and train discriminator
            # Discriminator wants to guess the correct label
            modelD.trainable = True
            d_loss = modelD.train_on_batch(X_current, y_current)
            modelD.trainable = False

            # Create new noise and train generator
            # Generator wants to generate image but have discrim think its real
            noiseG = np.random.uniform(-1, 1, size=(batchSize, 100))
            y_train_labels = np.random.uniform(0.9, 1, size=batchSize)
            g_loss = modelGD.train_on_batch(noiseG, y_train_labels)

            print("Epoch/Batch: {:>3}/{:<3}, d_loss:{:2f}, g_loss:{:2f}".format(epoch, index, d_loss, g_loss))

# Save model
save = args["save_model"] > 0
if save:
    print("Saving model to file...")
    if not os.path.exists("output"):
        os.mkdir("output")
    modelD.save_weights(dWeightsPath)
    modelG.save_weights(gWeightsPath)

# Generate a few images
noise = np.random.uniform(-1, 1, size=(batchSize, 100))
image = all_images(modelG.predict(noise, verbose=1))
cv2.imwrite("images/art/" + str(epoch) + '_' + str(index) + ".jpg", image)
