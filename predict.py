import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd

Y_train = pd.read_csv('data/Training_Out.csv')
Y_train_Ux = Y_train['Ux'].to_numpy()
Y_train_Uy = Y_train['Uy'].to_numpy()
Y_train_Uz = Y_train['Uz'].to_numpy()
Y_train_P  = Y_train['P'].to_numpy()

Ux_max = Y_train_Ux.max()
Uy_max = Y_train_Uy.max()
Uz_max = Y_train_Uz.max()
P_max  = Y_train_P.max()

Ux_min = Y_train_Ux.min()
Uy_min = Y_train_Uy.min()
Uz_min = Y_train_Uz.min()
P_min  = Y_train_P.min()

X_test = pd.read_csv('data/Test_In_Norm.csv').to_numpy()
Y_test = pd.read_csv('data/Test_Out_Norm.csv').to_numpy()
X_test.reshape(-1,1,8)
Y_test.reshape(-1,1,4)

model:Sequential = keras.models.load_model('models/CFD_FDA_Nozzle_v2.keras')

model.evaluate(X_test, y=Y_test,  batch_size=512)

Y_hatn = model.predict(X_test, batch_size=512)

# Unnormalize the prediction
Y_pred = pd.DataFrame()
Y_pred['Ux'] = Y_hatn[:,0] * (Ux_max - Ux_min) + (Ux_min)  
Y_pred['Uy'] = Y_hatn[:,1] * (Uy_max - Uy_min) + (Uy_min)  
Y_pred['Uz'] = Y_hatn[:,2] * (Uz_max - Uz_min) + (Uz_min)  
Y_pred['P']  = Y_hatn[:,3] * (P_max - P_min)   + (P_min)

# Write Velocity
with open('out/Test_Out_U.txt','w') as f:
    for index, row in Y_pred.iterrows():
        f.writelines(f'({row["Ux"]} {row["Uy"]} {row["Uz"]})\n')

# Write Pressure
with open('out/Test_Out_P.txt','w') as f:
    for index, row in Y_pred.iterrows():
        f.writelines(f'{row["P"]}\n')
