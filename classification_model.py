import pandas as pd
import numpy as np

model_params = pd.read_pickle('./data/Iris-LR-scratch.pkl')

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def predict(X):
    X = np.array(X, dtype="float64") 
    n = len(X)
    w_s = model_params.loc["iris-setosa"][:n].to_numpy()
    w_vc = model_params.loc["iris-versicolor"][:n].to_numpy()
    w_vg = model_params.loc["iris-virginica"][:n].to_numpy()
    b_s = model_params.loc["iris-setosa"][-1]
    b_vc = model_params.loc["iris-versicolor"][-1]
    b_vg = model_params.loc["iris-virginica"][-1]
   
    z_wb_s = np.dot(w_s, X) + b_s
    z_wb_vc = np.dot(w_vc, X) + b_vc
    z_wb_vg = np.dot(w_vg, X) + b_vg
        
    # Calculate the prediction for setosa
    f_wb_s = sigmoid(z_wb_s)
        
    # Calculate the prediction for versicolor
    f_wb_vc = sigmoid(z_wb_vc)
        
    # Calculate the prediction for virginica
    f_wb_vg = sigmoid(z_wb_vg)

    # Apply the threshold
    f_wb = [f_wb_s, f_wb_vc, f_wb_vg]

    return model_params.index.to_list()[f_wb.index(max(f_wb))]