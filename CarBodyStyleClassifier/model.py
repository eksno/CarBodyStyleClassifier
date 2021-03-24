import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Model:
    def __init__(self, train, test):
        self.x_train = train[0]
        self.y_train = train[1]
        self.x_test = test[0]
        self.y_test = test[1]

        self.num_classes = self.y_train.shape[1]

        self.model = None

    def create_model(self):
        self.model = Sequential()

        # Input Layer
        self.model.add(Input(self.x_train.shape[1:]))

        self.model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 2nd Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 3rd Convolutional Layer
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 4th Convolutional Layer
        self.model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 5th Convolutional Layer
        self.model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # Passing it to a Fully Connected layer
        self.model.add(Flatten())

        # 1st Fully Connected Layer
        self.model.add(Dense(4096))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        self.model.add(Dense(4096))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        self.model.add(Dense(1000))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.4))

        # Output Layer
        self.model.add(Dense(self.num_classes))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))

        # Compile the model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        


    def train_model(self, batch_size=5, epochs=100):
        lrr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01, patience=3, min_lr=1e-5)

        train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=0.1)
        test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=0.1)

        train_generator.fit(self.x_train)
        test_generator.fit(self.x_test)

        self.model.fit(train_generator.flow(self.x_train, self.y_train, batch_size=batch_size),
                                epochs=epochs,
                                steps_per_epoch = self.x_train.shape[0] // batch_size,
                                validation_data = test_generator.flow(self.x_test, self.y_test, batch_size=batch_size),
                                validation_steps = 250,
                                callbacks = [lrr],
                                shuffle=True)

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')
