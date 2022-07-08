import pandas as pd
from keras import Sequential
from keras.layers import InputLayer, Dense
from keras import metrics
from keras.optimizers import SGD
from keras.losses import MeanSquaredError
from keras.callbacks import History


def create_neural_network(n_inputs: int, n_outputs: int) -> Sequential:
    number_neurons_hidden_layers: list = [32, 32]

    model: Sequential = Sequential(name='neural_network')
    input_layer: InputLayer = InputLayer(input_shape=(n_inputs,))

    for neurons in number_neurons_hidden_layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    return model


def compile_neural_network(model: Sequential, learning_rate: float):
    model.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=learning_rate),
                  metrics=metrics.MeanSquaredError())


def train_neural_network(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
                         n_epochs: int, batch_size: int) -> History:
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0,
                        validation_data=(x_test, y_test))
    return history
