from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop


def mlp(n_input=75, architecture=[(8, 'sigmoid'), (1, 'sigmoid')],
        lr=0.01):
    model = Sequential()

    for i, l in enumerate(architecture):
        n_neurons, f_act = l
        if i == 0:
            model.add(Dense(n_neurons, input_shape=(n_input,)))
        else:
            model.add(Dense(n_neurons))

        model.add(Activation(f_act))

    rms = RMSprop(lr=lr)
    model.compile(loss='mse', optimizer=rms)
    model.arch = architecture
    return model
