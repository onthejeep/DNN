import numpy as np;

def NormalScaler(x):
    return [np.mean(x, axis= 0), np.std(x, axis= 0)];

def NormalScaler_Transform(x, scaler):
    Scaled = (x - scaler[0]) / scaler[1];
    return Scaled;

def NormalScaler_Inverse(x, scaler):
    Scaled = x * scaler[1] + scaler[0];
    return Scaled;

def MinMaxScaler(x):
    ValueRange = np.max(x, axis= 0) - np.min(x, axis= 0);
    ScaleRange = 1 - 2 * 1e-5;
    return [np.min(x, axis= 0), ValueRange, ScaleRange];

def MinMaxScaler_Transform(x, scaler):
    Scaled = scaler[2] * (x - scaler[0]) / scaler[1] + 1e-5;
    return Scaled;

def MinMaxScaler_Inverse(x, scaler):
    Scaled = (x - 1e-5) * scaler[1] / scaler[2] + scaler[0];
    return Scaled;

def ComboScaler(x):
    Scaler = [None] * 2;
    Scaler[0] = NormalScaler(x);
    Normal_Trans = NormalScaler_Transform(x, Scaler[0]);
    Scaler[1] = MinMaxScaler(Normal_Trans);
    return Scaler;

def ComboScaler_Transform(x, scaler):
    Normal_Transform = NormalScaler_Transform(x, scaler[0]);
    MinMax_Transform = MinMaxScaler_Transform(Normal_Transform, scaler[1]);
    return MinMax_Transform;

def ComboScaler_Inverse(x, scaler):
    MinMax_Inverse = MinMaxScaler_Inverse(x, scaler[1]);
    Normal_Inverse = NormalScaler_Inverse(MinMax_Inverse, scaler[0]);
    return Normal_Inverse;

