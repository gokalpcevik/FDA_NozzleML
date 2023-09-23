import tensorflow as tf
import random
import json
import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

training_input = data.get_training_normalized_input()
training_output = data.get_training_normalized_output()

Ux = training_output[:,0]
Uy = training_output[:,1]
Uz = training_output[:,2]
P  = training_output[:,3]

Ux_max = Ux.max()
Uy_max = Uy.max()
Uz_max = Uz.max()
P_max  = P.max()

Ux_min = Ux.min()
Uy_min = Uy.min()
Uz_min = Uz.min()
P_min  = P.min()

test_input = data.get_test_normalized_input()
Re5000_P = data.get_Re5000_P()
Re5000_U = data.get_Re5000_U()

def representative_dataset():
    for r in training_input[:250,:]:
        yield [tf.dtypes.cast(r, tf.float32)]

model = tf.keras.models.load_model('models/CFD_FDA_Nozzle.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = representative_dataset

model_lite = converter.convert()

interpreter = tf.lite.Interpreter(model_content=model_lite)
interpreter.allocate_tensors()

print(interpreter.get_tensor_details()[0])
print('>>>>>>>------------------<<<<<<<<<<<')
print(interpreter.get_tensor_details()[1])
print('>>>>>>>------------------<<<<<<<<<<<')
print(interpreter.get_tensor_details()[2])

input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

print(interpreter.get_tensor_details())
#for i in range(0,5):
#    interpreter.set_tensor(input_idx, tf.dtypes.cast(test_input[i,:].reshape(1,8),tf.float32))
#    interpreter.invoke()
#    P = interpreter.get_tensor(output_idx)
#    print(P)

    
    