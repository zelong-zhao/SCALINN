
from functools import partial
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

def model_builder(hp,input_dict:dict):
    n_x=input_dict['n_x']
    n_y=input_dict['n_y']
    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(n_x,)))
    layer=hp.Int("num_layers", min_value= 4,max_value= 24,step=4)
    # layer = hp.Choice('num_layers', values=[4, 8, 12, 16])
    for i in range(layer):
        model.add(
            keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=int(n_y/2), max_value=int(n_y*2), step=4),
                activation=hp.Choice(f"activation_{i}", ["relu", "tanh","softmax",'sigmoid']),
            )
        )
        if hp.Boolean("dropout"):
            model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Dense(n_y))
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    loss_fn = 'mse'
    matrics = ['mae', 'mse']

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=loss_fn,
        metrics=matrics,
    )
    return model

def hyperband_train(x_train,y_train,input_dict,directory,project_name):
    tuner = kt.Hyperband(hypermodel=partial(model_builder,input_dict=input_dict),
                        objective='val_loss',
                        max_epochs=200,
                        factor=10,
                        directory=directory,
                        project_name=project_name)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print(tuner.search_space_summary())
    tuner.search(x_train, y_train,validation_split=0.2,epochs=20, callbacks=[stop_early])

    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

    best_model=tuner.get_best_models(num_models=1)
    return best_model[0]