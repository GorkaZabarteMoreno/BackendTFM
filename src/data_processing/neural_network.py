import matplotlib.pyplot as plt
import pandas as pd

from keras import Sequential
from keras.layers import InputLayer, Dense
from keras import metrics
from keras.optimizers import SGD
from keras.losses import MeanSquaredError
from keras.callbacks import History

from time import perf_counter

from src.data_processing.training_split import split


def create_NN(n_inputs: int, n_outputs: int) -> Sequential:
    number_neurons_hidden_layers: list = [300, 300, 200, 150]

    model: Sequential = Sequential(name='neural_network')
    input_layer: InputLayer = InputLayer(input_shape=(n_inputs,))
    model.add(layer=input_layer)

    for neurons in number_neurons_hidden_layers:
        model.add(layer=Dense(units=neurons, activation='relu'))
    model.add(layer=Dense(units=n_outputs, activation='sigmoid'))

    return model


def compile_NN(model: Sequential, learning_rate: float):
    mean_metrics: list = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError, metrics.MeanRelativeError]
    model.compile(loss=MeanSquaredError(), optimizer=SGD(learning_rate=learning_rate),
                  metrics=mean_metrics)


def train_NN(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
             batch_size: int, n_epochs: int) -> History:
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs, verbose=2,
                        validation_data=(x_test, y_test))
    return history


def predict_NN(model, x_test: pd.DataFrame, batch_size: int):
    return model.predict(x=x_test, batch_size=batch_size)


def evaluate_NN(model, x_test: pd.DataFrame, y_test: pd.Series, batch_size: int):
    return model.evaluate(x=x_test, y=y_test, batch_size=batch_size)


def neural_network(dataframe: pd.DataFrame, target: str):
    x_train, y_train, x_test, y_test = split(dataframe=dataframe, label_name=target, training_ratio=0.75)
    start_time = perf_counter()

    model: Sequential = create_NN(n_inputs=x_train.shape[1], n_outputs=1)
    compile_NN(model=model, learning_rate=0.001)
    errors = train_NN(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, batch_size=64, n_epochs=1000)
    results = predict_NN(model=model, x_test=x_test, batch_size=64)
    evaluation = evaluate_NN(model=model, x_test=x_test, y_test=y_test, batch_size=64)

    print(model.metrics_names)
    print(errors)
    print(type(errors))

    x = pd.DataFrame(errors.history)
    x.plot(figsize=(8, 5))
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy - Mean Log Loss")
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0,1]
    plt.show()

    print("Error (training): ", round((1 - results.mean_squared_error.values[-1:][0]) * 100, 1), "%")
    print("Error (development test): ", round((1 - results.val_mean_squared_error.values[-1:][0]) * 100, 1), "%")
    print("Loss (training): ", round((1 - results.loss.values[-1:][0]) * 100, 1), "%")
    print("Loss (development test): ", round((1 - results.val_loss.values[-1:][0]) * 100, 1), "%")
    print("Time: ", round((perf_counter() - start_time)), "seconds")
    print("Loss final: ", evaluation[0])
    print("Mean Squared Error final: ", evaluation[1])
