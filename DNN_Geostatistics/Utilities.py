import numpy as np;

Destination_Col = 45;
Destination_Row = 47;

def Find_FID(colIndex, rowIndex):
    FID = 100 * (colIndex - 1) + (rowIndex - 1);
    return FID;

def Find_ColRow(fid):
    Col = np.ceil(fid / 100);
    Row = (fid - 100 * Col)  % 100 + 1;
    return Col, Row;