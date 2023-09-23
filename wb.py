import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers, Sequential
from keras.models import load_model

# * NOT INCLUDING THE SIGN BIT
INTEGER_BITS = 2
FRACTIONAL_BITS = 15
TOTAL_BITS = INTEGER_BITS + FRACTIONAL_BITS + 1

model:Sequential = keras.models.load_model('models/CFD_FDA_Nozzle_v2.keras')

def Q_to_float(Q: int, fractional_bits: int) -> float:
    return float(Q) * pow(2.0, -fractional_bits)

def float_to_Q(f: float, fractional_bits: int) -> int:
    return int(f * pow(2.0, fractional_bits))

def to_twos_comp_hex(Q: float) -> int:
    twos_complement_hex = hex(((float_to_Q(Q, FRACTIONAL_BITS) + (1 << TOTAL_BITS)) % (1 << TOTAL_BITS)))
    twos_complement_hex = twos_complement_hex[2:]
    twos_complement_hex = twos_complement_hex.zfill(5)
    return twos_complement_hex

def layer_to_mem(layer: keras.layers.Layer) -> None:
    Ws = layer.weights[0].numpy()
    Bias = layer.weights[1].numpy()
    with open(f'wb/{layer.name}_WB_Q{INTEGER_BITS}_{FRACTIONAL_BITS}.mem','w') as f:
        f.writelines('@00000000\n')
        # * Transpose so every row holds the weight of a single neuron in that layer
        for w in Ws.transpose().flatten():
            twos_complement_hex = hex(((float_to_Q(w, FRACTIONAL_BITS) + (1 << TOTAL_BITS)) % (1 << TOTAL_BITS)))
            twos_complement_hex = twos_complement_hex[2:]
            twos_complement_hex = twos_complement_hex.zfill(5)
            f.writelines(str(twos_complement_hex) + '\n')
        for b in Bias:
            twos_complement_hex = hex(((float_to_Q(b, FRACTIONAL_BITS) + (1 << TOTAL_BITS)) % (1 << TOTAL_BITS)))
            twos_complement_hex = twos_complement_hex[2:]
            twos_complement_hex = twos_complement_hex.zfill(5)
            f.writelines(str(twos_complement_hex) + '\n')
    pass


for l in model.layers:
    layer_to_mem(l)
