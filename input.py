import numpy as np
import pandas as pd

X_test = pd.read_csv('data/Test_In_Norm.csv').to_numpy()
X_test.reshape(-1,1,8)

f = open('data/test_in.c', 'w')
f.write('const CFD_Input_t test_inputs[] = {\n\t')

for x in X_test:
    f.write('{')
    for i in range(len(x)):
        f.write(f'{x[i]}')
        if i != 8:
            f.write(',')
    f.write('},')
    f.write('\n\t')
f.write('}\n')
f.close()
