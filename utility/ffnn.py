'''
Author: Ramin Anushiravani
Date: April 12th/23
Feed Forward Neural Network Helper
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from .model_utils import ModelUtil
import keras


class FFNN(ModelUtil):
    '''
    Implements 5 different feedforward neural network architectures and
    trains them with time-ordered cross-validation. Inherits the data utility
    and model evaluation helpers from ModelUtil.
    '''

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _build_model(self, model_num, lr, inp_shape):
        '''
        Builds one of the 5 architectures by model number.

        Raises:
            ValueError: if model_num is not between 1 and 5.
        '''
        builders = {
            1: self.create_model_1,
            2: self.create_model_2,
            3: self.create_model_3,
            4: self.create_model_4,
            5: self.create_model_5,
        }
        if model_num not in builders:
            raise ValueError(
                "model_num must be one of 1-5, got {!r}".format(model_num))
        return builders[model_num](lr, inp_shape)

    def create_model_1(self, lr=0.001, inp_shape=None):
        '''
        Builds feedforward architecture 1: the deepest and largest model,
        intended mainly for raw (full-dimensionality) data.

        Args:
            lr: learning rate for the Adam optimizer.
            inp_shape: number of input features.

        Returns:
            A compiled keras Sequential model.
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(1024, input_shape=(inp_shape,), activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_2(self, lr=0.001, inp_shape=None):
        '''
        Builds feedforward architecture 2: a mid-sized model with fewer
        parameters than architecture 1.

        Args:
            lr: learning rate for the Adam optimizer.
            inp_shape: number of input features.

        Returns:
            A compiled keras Sequential model.
        '''

        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(256, input_shape=(inp_shape,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_3(self, lr=0.001, inp_shape=None):
        '''
        Builds feedforward architecture 3: fewer neurons to avoid overfitting
        on low-dimensional PCA features.

        Args:
            lr: learning rate for the Adam optimizer.
            inp_shape: number of input features.

        Returns:
            A compiled keras Sequential model.
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_4(self, lr=0.0001, inp_shape=None):
        '''
            architecture 4 - smaller still, tuned for low-dimensional PCA features
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_5(self, lr=0.0001, inp_shape=None):
        '''
            architecture 5 - linear activations, the smallest model
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(64, activation='linear'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='linear'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='linear'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def train_FFNN(
            self,
            X_train,
            y_train,
            model_num=1,
            lr=0.001,
            num_folds=5,
            epochs=5,
            batch_size=32):
        '''
        Trains a feedforward neural network with time-ordered cross-validation
        (see gen_data). A fresh model is built for every fold.

        Args:
            X_train: training matrix.
            y_train: training labels.
            model_num: which architecture to use (1-5).
            lr: learning rate.
            num_folds: number of cross-validation folds. With num_folds=1,
                trains once on the full data.
            epochs: number of training epochs per fold.
            batch_size: batch size.

        Returns:
            The trained keras model (from the last fold).
        '''

        inp_shape = X_train.shape[1]
        model = self._build_model(model_num, lr, inp_shape)

        if num_folds == 1:
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=self.verbose)
            return model

        GEN = self.gen_data(X_train, y_train, num_folds=num_folds)
        while True:
            try:
                training_data, training_label, validation_data, validation_label = next(
                    GEN)
                # Rebuild a fresh model for each fold so no weights leak
                # from the previous fold.
                model = self._build_model(model_num, lr, inp_shape)
                model.fit(
                    training_data,
                    training_label,
                    validation_data=(
                        validation_data,
                        validation_label),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=self.verbose)
            except StopIteration:
                break
        return model
