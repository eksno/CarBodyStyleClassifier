import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


class Model:
    def __init__(self, train, test, num_classes):
        self.num_classes = num_classes

        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]

        self.model = None

    def create_model(self):
        self.model = Sequential()

        # Input Layer
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        # Hidden Layers
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())

        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # Output Layer
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])

    def train_model(self, batch_size=2, epochs=12):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

        score = self.model.evaluate(self.x_test, self.y_test, verbose=2)

        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')
