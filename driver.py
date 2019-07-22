from dcgan_model import Models
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from load import Load
from dcgan_model import Models
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
(_X, _y) = Load.load_data()

# Convert to np arrays
X = np.array(_X)
y = np.array(_y)

# Scale data to [-1, 1] range
X = (X.astype("float32") - 127.5) / 127.5

# Initialize optimizer and model
print("Loading model...")
loadPath = args["load_model"] > 0 or args["train_model"] > 0
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
modelG = Models.buildGenerator(
    vectSize=100,
    weightsPath=gWeightsPath if loadPath else None)
modelD = Models.buildDiscriminator(
    numChannels=3, imgRows=128, imgCols=128,
    weightsPath=dWeightsPath if loadPath else None)

# Generator
modelG.compile(loss="binary_crossentropy", optimizer=opt)

# Discriminator
modelD.compile(loss="binary_crossentropy", optimizer=opt)

# Discriminator on top of generator
modelDG = Sequential()
modelDG.add(modelG)
modelD.trainable = False
modelDG.add(modelD)
modelDG.compile(loss="binary_crossentropy", optimizer=opt)

# Train (only if not loading model)
batchSize = 32
outputTime = 10
epochs = 1000
train = args["load_model"] < 0
if train:
    # 100 epochs
    for epoch in range(epochs):
        numBatches = int(X.shape[0] / batchSize)
        for index in range(numBatches):
            # Random noise of vector 100 (batchSize number of them)
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))

            # Generate image, and give it label 0.
            X_fake = modelG.predict(noise)
            y_fake = [0] * X_fake.shape[0]

            # Output generated images every number of epochs
            if index % outputTime == 0:
                image = all_images(X_fake)
                cv2.imwrite("images/art/" + str(epoch) + '_' + str(index) + ".jpg", image)

                if not os.path.exists("output"):
                    os.mkdir("output")
                modelD.save_weights(dWeightsPath)
                modelG.save_weights(gWeightsPath)

            # Select current batch of real images
            # Note: y_current is just a vectors of 1s
            X_real = X[index * numBatches:(index + 1) * numBatches]
            y_real = y[index * numBatches:(index + 1) * numBatches]

            # Add X_train and y_train
            X_current = np.concatenate([X_real, X_fake])
            y_current = np.concatenate([y_real, y_fake])

            # Feed into discriminator and train discriminator
            # Discriminator wants to guess the correct label
            d_loss = modelD.train_on_batch(X_current, y_current)
            # Create new noise and train generator
            # Generator wants to generate image but have discrim think its real
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))
            y_train_labels = [1] * batchSize
            g_loss = modelDG.train_on_batch(noise, y_train_labels)

            print("Epoch/Batch: {}/{}, d_loss:{}, g_loss:{}".format(epoch, index, d_loss, g_loss))

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
