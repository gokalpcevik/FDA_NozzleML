import numpy as np
from numpy import genfromtxt

dir = 'data/'

def get_training_output():
    return genfromtxt(dir + 'Training_Out.csv',delimiter=',')

def get_training_normalized_input():
    return genfromtxt(dir + 'Training_In_Norm.csv',delimiter=',')
        
def get_training_normalized_output():
    return genfromtxt(dir + 'Training_Out_Norm.csv',delimiter=',')

def get_test_normalized_input():
    return genfromtxt(dir + 'Test_In_Norm.csv',delimiter=',')

def get_test_normalized_output():
    return genfromtxt(dir + 'Test_Out_Norm.csv',delimiter=',')

def get_matlab_est_normalized():
    return genfromtxt(dir + 'MATLAB_Estimation_Norm.csv',delimiter=',')

def get_Re5000_U():
    return genfromtxt(dir + 'Re5000_U.csv',delimiter=',')

def get_Re5000_P():
    return genfromtxt(dir + 'Re5000_P.csv',delimiter=',')


