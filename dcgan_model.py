from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras import backend as K


class Models:
    @staticmethod
    def buildGenerator(vectSize, weightsPath=None):
        model = Sequential()

        model.add(Dense(65536, input_dim=vectSize))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(512, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv2DTranspose(256, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(128, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(3, 5, strides=(1, 1), padding="same"))
        model.add(Activation("tanh"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

    @staticmethod
    def buildDiscriminator(imgRows, imgCols, numChannels, weightsPath=None):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        model.add(Conv2D(64, 5, strides=(2, 2), padding="same", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(128, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(256, 5, strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(512, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2D(1024, 5, strides=(2, 2), padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
