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
((trainData, trainLabels), (testData, testLabels)) = Load.load_data()

# Convert to np arrays
X_train = np.array(trainData)
X_test = np.array(testData)
y_train = np.array(trainLabels)
y_test = np.array(testLabels)

# Scale data to [-1, 1] range
X_train = (X_train.astype("float32") - 127.5) / 127.5
X_test = (X_test.astype("float32") - 127.5) / 127.5

# Initialize optimizer and model
print("Loading model...")
loadPath = args["load_model"] > 0 or args["train_model"] > 0
opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
modelG = Models.buildGenerator(
    vectSize=100,
    weightsPath=gWeightsPath if loadPath else None)
modelD = Models.buildDiscriminator(
    numChannels=3, imgRows=128, imgCols=128,
    weightsPath=dWeightsPath if loadPath else None)

# Discriminator on top of generator
modelDG = Sequential()
modelDG.add(modelG)
modelD.trainable = False
modelDG.add(modelD)
modelDG.compile(loss="binary_crossentropy", optimizer=opt)

# Generator
modelG.compile(loss="binary_crossentropy", optimizer=opt)

# Discriminator
modelD.trainable = True
modelD.compile(loss="binary_crossentropy", optimizer=opt)

# Train (only if not loading model)
batchSize = 32
outputTime = 5
epochs = 1000
train = args["load_model"] < 0
if train:
    # 100 epochs
    for epoch in range(epochs):
        numBatches = int(X_train.shape[0] / batchSize)
        for index in range(numBatches):
            # Random noise of vector 100 (batchSize number of them)
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))

            # Generate image, and give it label 0.
            X_train_fake = modelG.predict(noise)
            y_train_fake = [0] * X_train_fake.shape[0]

            # Output generated images every number of epochs
            if index % outputTime == 0:
                image = all_images(X_train_fake)
                image = (image * 127.5 + 127.5).astype(np.uint8)
                cv2.imwrite("images/art/" + str(epoch) + '_' + str(index) + ".jpg", image)

                if not os.path.exists("output"):
                    os.mkdir("output")
                modelD.save_weights(dWeightsPath)
                modelG.save_weights(gWeightsPath)

            # Select current batch of real images
            # Note: y_train_current is just a vectors of 1s
            X_train_real = X_train[index * numBatches:(index + 1) * numBatches]
            y_train_real = y_train[index * numBatches:(index + 1) * numBatches]

            # Add X_train and y_train
            X_train_current = np.concatenate([X_train_real, X_train_fake])
            y_train_current = np.concatenate([y_train_real, y_train_fake])

            # Feed into discriminator and train discriminator
            # Discriminator wants to guess the correct label
            d_loss = modelD.train_on_batch(X_train_current, y_train_current)

            # Create new nosie and train generator
            # Generator wants to generate image but have discrim think its real
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))
            y_train_labels = [1] * batchSize
            modelD.trainable = False
            g_loss = modelDG.train_on_batch(noise, y_train_labels)
            modelD.trainable = True

            print("Epoch/Batch: {}/{}, d_loss:{}, g_loss:{}\n".format(epoch, index, d_loss, g_loss))

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
image = (image * 127.5 + 127.5).astype(np.uint8)
cv2.imwrite("images/art/" + str(epoch) + '_' + str(index) + ".jpg", image)
