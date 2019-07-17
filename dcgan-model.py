from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras import backend as K


class Models:
    @staticmethod
    def buildGenerator(imgRows, imgCols, numChannels, weightsPath=None):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        model.add(Dense(65536, input_shape=inputShape))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU())

        model.add(Conv2D(512, 16, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv2D(256, 32, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(128, 64, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(64, 128, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(3, 128, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

    @staticmethod
    def buildDiscriminator(imgRows, imgCols, numChannels, weightsPath=None):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        model.add(Conv2D(64, 64, strides=(2, 2), padding="same", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(128, 32, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(256, 16, strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(512, 16, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(1024, 8, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Flatten())
        model.add(Dense())
        model.add(Activation("sigmoid"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
