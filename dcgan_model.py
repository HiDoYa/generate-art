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
        shape = (8, 8, 1024)

        model.add(Dense(shape[0] * shape[1] * shape[2], input_dim=vectSize))
        model.add(Reshape(shape))
        model.add(Activation("relu"))

        # Note: All filter numbers have been cut in half (due to AWS constraints)
        model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(3, kernel_size=5, strides=1, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

    @staticmethod
    def buildDiscriminator(imgRows, imgCols, numChannels, weightsPath=None):
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)
        reluAlpha = 0.2

        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same", input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=reluAlpha))

        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=reluAlpha))

        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=reluAlpha))

        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=reluAlpha))

        model.add(Conv2D(1024, kernel_size=5, strides=1, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=reluAlpha))

        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.summary()

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
