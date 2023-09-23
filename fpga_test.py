import numpy as np
import pandas as pd

X_test = pd.read_csv('data/Test_In_Norm.csv').to_numpy()
Y_test = pd.read_csv('data/Test_Out_Norm.csv').to_numpy()

X_test.reshape(-1,1,8)
Y_test.reshape(-1,1,4)

X_fpga = X_test[:5]
Y_fpga = Y_test[:5]

with open('out/X_fpga.txt','w') as f:
    f.writelines(np.array2string(X_fpga,max_line_width=300))
    
with open('out/Y_fpga.txt','w') as f:
    f.writelines(np.array2string(Y_fpga,max_line_width=300))

