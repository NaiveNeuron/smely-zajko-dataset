from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop


def mlp(n_input=75, architecture=[(8, 'sigmoid'), (1, 'sigmoid')],
        lr=0.01, optimizer=None, loss='mse', metrics=None):

    if metrics is None:
        metrics = []

    model = Sequential()

    for i, l in enumerate(architecture):
        n_neurons, f_act = l
        if i == 0:
            model.add(Dense(n_neurons, input_shape=(n_input,)))
        else:
            model.add(Dense(n_neurons))

        model.add(Activation(f_act))

    if optimizer is None:
        optimizer = RMSprop(lr=lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.arch = architecture
    return model
