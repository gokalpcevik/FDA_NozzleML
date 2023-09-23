import sys
from ctypes import sizeof
import numpy as np
import pandas as pd
from serial import Serial

BAUD_RATE = 921600
CFD_DATA_TYPE = np.dtype(np.float32)
CFD_DATA_BYTE_SIZE = np.dtype(np.float32).itemsize

Y_test = pd.read_csv("data/Test_Out_Norm.csv")

ser = Serial("COM6", BAUD_RATE, timeout=None, parity="N")
print(f"Polling for input from {ser.name}(CBR={BAUD_RATE}):")
ser.set_buffer_size(rx_size= 4096 * 8)
ser.reset_input_buffer()

# In bytes
amnt_received = 0
# In KiB
total_data_sz_kib = Y_test.shape[0] * CFD_DATA_BYTE_SIZE * 4 / 1024

byte_stream: bytearray = bytearray()

while True:
    b0 = ser.read(2)
    if b0 == bytes(">>", encoding="ascii"):
        sys.stdout.write(str(ser.readline().decode("ascii")))
        continue
    elif b0 == bytes("++", encoding="ascii"):
        byte_stream += ser.read(CFD_DATA_BYTE_SIZE * 4 * 4)
        amnt_received += CFD_DATA_BYTE_SIZE * 4 * 4
        if amnt_received % 65536 == 0:
            progress = amnt_received / (total_data_sz_kib * 1024.0) * 100.0
            sys.stdout.write(
                f"> Total TX Amount(KiB): {amnt_received / 1024.0} of {total_data_sz_kib}| Progress: {progress:.2f}% \r")
    elif b0 == bytes("!!", encoding="ascii"):
        print("> Termination bytestream received, closing port.")
        ser.close()
        break

print("> Parsing predictions from bytestream.")

print(f'> Bytestream total size: {len(byte_stream)}')
byte_stream += np.random.bytes(CFD_DATA_BYTE_SIZE * 4)
predictions = np.ndarray(shape=(Y_test.shape), dtype=CFD_DATA_TYPE, buffer=byte_stream, offset=0)

df = pd.DataFrame(predictions)
df["Ux"] = predictions[:, 0]
df["Uy"] = predictions[:, 1]
df["Uz"] = predictions[:, 2]
df["P"] = predictions[:, 3]

with open(f"out/FPGA_Out_U_Norm_{CFD_DATA_TYPE.name}.txt", "w") as f:
    for index, row in df.iterrows():
        f.writelines(f'({row["Ux"]} {row["Uy"]} {row["Uz"]})\n')

with open(f"out/FPGA_Out_P_Norm_{CFD_DATA_TYPE.name}.txt", "w") as f:
    for index, row in df.iterrows():
        f.writelines(f'{row["P"]}\n')
