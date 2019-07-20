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


def all_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = images.shape[1:3] # for height, width, channel
    image = np.zeros((height * shape[0], width * shape[1]))
    for index, image in enumerate(images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = image
    return image


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(Optional) Whether to save model to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1, help="(Optional) Whether to load model from disk")
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

# Scale data to [0, 1] range
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#X_train = X_train[:, :, :, None]
#X_test = X_test[:, :, :, None]

# Initialize optimizer and model
print("Loading model...")
opt = SGD(lr=0.001, momentum=0.9)
modelG = Models.buildGenerator(
    vectSize=100, weightsPath=gWeightsPath if args["load_model"] > 0 else None)
modelD = Models.buildDiscriminator(
    numChannels=3, imgRows=128, imgCols=128,
    weightsPath=dWeightsPath if args["load_model"] > 0 else None)

# Discriminator on top of generator
modelDG = Sequential()
modelDG.add(modelG)
modelD.trainable = False
modelDG.add(modelD)
modelDG.compile(loss="categorical_crossentropy", optimizer=opt)

# Generator
modelG.compile(loss="categorical_crossentropy", optimizer=opt)

# Discriminator
modelD.trainable = True
modelD.compile(loss="categorical_crossentropy", optimizer=opt)

# Train (only if not loading model)
batchSize = 32
outputTime = 10
if args["load_model"] < 0:
    # 100 epochs
    for epoch in range(10):
        numBatches = int(X_train.shape[0] / batchSize)
        for index in range(numBatches):
            # Random noise of vector 100 (batchSize number of them)
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))

            # Generate image, and give it label 0.
            X_train_fake = modelG.predict(noise, verbose=1)
            y_train_fake = [0] * X_train_fake.shape[0]

            # Output generated images every number of epochs
            if index % outputTime == 0:
                print("Output after " + str(index) + " epochs")

            # Select current batch of real images
            # Note: y_train_current is just a vectors of 1s
            X_train_real = X_train[index * numBatches:(index + 1) * numBatches]
            y_train_real = y_train[index * numBatches:(index + 1) * numBatches]

            print(X_train_real.shape)
            print(X_train_fake.shape)
            # Add X_train and y_train
            X_train_current = np.concatenate(X_train_real, X_train_fake)
            y_train_current = np.concatenate(y_train_real, y_train_fake)

            # Feed into discriminator and train discriminator
            # Discriminator wants to guess the correct label
            d_loss = modelD.train_on_batch(X_train_current, y_train_current)

            # Create new nosie and train generator
            # Generator wants to generate image but have discrim think its real
            noise = np.random.uniform(-1, 1, size=(batchSize, 100))
            modelD.trainable = False
            g_loss = modelDG.train_on_batch(noise, y_train_real)
            modelD.trainable = True

# Save model
if args["save_model"] > 0:
    print("Saving model to file...")
    if not os.path.exists("output"):
        os.mkdir("output")
    modelD.save_weights(dWeightsPath)
    modelG.save_weights(gWeightsPath)

# Generate a few images
noise = np.random.uniform(-1, 1, size=(batchSize, 100))
image = all_images(modelG.predict(noise, verbose=1))
image = image * 255
cv2.imshow("Art", image)