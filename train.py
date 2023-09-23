import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer, BatchNormalizationV2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

training_input = pd.read_csv('data/Training_In_Norm.csv').to_numpy()
training_output = pd.read_csv('data/Training_Out_Norm.csv').to_numpy()

training_input.reshape(-1,1,8)
training_output.reshape(-1,1,4)

x_train = training_input[:650000]
y_train = training_output[:650000]

x_val = training_input[650000:700000]
y_val = training_output[650000:700000]

model = Sequential([
    InputLayer(input_shape=(8,)),
    Dense(units=15,activation='relu'),
    Dense(units=15,activation='relu'),
    Dense(units=4,activation= 'linear')
],name='CFD')

checkpoint = ModelCheckpoint('checkpoint_models/',save_best_only=True,monitor='val_loss', save_weights_only=True)
early_stop = EarlyStopping(patience=10)

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    metrics=[keras.metrics.RootMeanSquaredError()])

model.fit(x_train,
          y_train,
          validation_data=(x_val, y_val),
          batch_size=512,
          epochs=96,
          callbacks=[checkpoint,early_stop])

model.summary()

keras.models.save_model(model, 'models/CFD_FDA_Nozzle_v2.keras')

