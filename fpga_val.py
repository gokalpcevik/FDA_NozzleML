import numpy as np
from keras.models import Sequential, Model, load_model 
import pandas as pd

#x = np.array([[0.0466886037852477,0.496249569646695,0.491015544785067,0.735867427634285,0.124114404637985,0.42001238206692,0.121747214260813,0.5]])

X_test = pd.read_csv('data/Test_In_Norm.csv').to_numpy()[:5]
X_test.reshape(-1,1,8)

#x = X_test
x = np.ones(shape=(1,8))*0.5


model:Sequential = load_model('models/CFD_FDA_Nozzle_v2.keras')
layer_name = 'dense'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
output = intermediate_layer_model.predict(x)
print(f'----------------------------Layer 1 output----------------------------\n{output}')

layer_name = 'dense_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
output = intermediate_layer_model.predict(x)
print(f'----------------------------Layer 2 output----------------------------\n{output}')

layer_name = 'dense_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
output = intermediate_layer_model.predict(x)
print(f'----------------------------Layer 3 output----------------------------\n{output}')
