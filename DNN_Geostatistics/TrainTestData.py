import sqlite3
import sys
import numpy as np
import Utilities


def TrainData(if_train = True):
    '''
    output:
        input: TimeIndex, Col, Row, Manhattan, Euclid
        output: avg, std, cov, p90, p95, bt, pt
    '''
    if if_train:
        TimeIndex = TrainData_TimeIndex();
    else:
        TimeIndex = TestData_TimeIndex();

    Input = []; Output = [];
    for ti in TimeIndex:
        SingleInput, SingleOutput = Data_TimeIndex(ti);
        Input += SingleInput;
        Output += SingleOutput;
    
    return np.asarray(Input, dtype = np.float), \
            np.reshape(np.asarray(Output, dtype = np.float)[:,1], newshape = [-1, 1]);

def Normalization(input, output):
    Input = (input - np.mean(input, axis= 0)) / np.std(input, axis= 0);
    Input = (1 - 2 * 1e-5) * (Input - np.min(Input, axis= 0)) / (np.max(Input, axis= 0) - np.min(Input, axis= 0)) + 1e-5;

    Output = (output - np.mean(output, axis= 0)) / np.std(output, axis= 0);
    Output = (1 - 2 * 1e-5) * (Output - np.min(Output, axis= 0)) / (np.max(Output, axis= 0) - np.min(Output, axis= 0)) + 1e-5;
    return Input, Output;



def Data_TimeIndex(ti, dow = 'tuesday'):
    '''
    function: retrieve data from sqlite given a time index and day of week
    params: 
        ti: time index
        dow: day of week
    output:
        type(output): list
    '''
    GridDatabase = 'Result/ttr_to_(col_45_row_47).sqlite';
    Connection = sqlite3.connect(GridDatabase);

    ColumnNames = ColumnName_TimeIndex(ti);
    SelectCommand = 'select {} from grid_kunshan_{} where ti_{}_sz > 6 and ti_{}_bt > 0'.format(ColumnNames, dow, ti, ti);
    # print(SelectCommand);
    Cursor = Connection.cursor();
    Cursor.execute(SelectCommand);

    Result = Cursor.fetchall();
    # Result = [ list(row) for row in Result];
    Cursor.close();
    Connection.close();

    Input = []; Output = [];

    for row in Result:
        Col, Row = Utilities.Find_ColRow(row[0]);
        Manhattan = abs(Col - Utilities.Destination_Col) + abs(Row - Utilities.Destination_Row);
        Euclid = np.sqrt(np.square(Col - Utilities.Destination_Col) + np.square(Row - Utilities.Destination_Row));
        SingleInput = [ti, Col, Row, Manhattan, Euclid]; # 

        Input.append(SingleInput);
        Output.append(list(row[2:]));

    return Input, Output;


def ColumnName_TimeIndex(ti):
    '''
    column names include:
        grid id, sample size
        avg, std, cov, p90, p95, bt, pt
    '''
    ColumnName = 'id, ti_{}_sz as sz, ti_{}_avg as avg, ti_{}_std as std, ti_{}_cov as cov, \
    ti_{}_p90 as p90, ti_{}_p95 as p95, ti_{}_bt as bt, ti_{}_pt as pt'.format(ti, ti, ti, ti, ti, ti, ti, ti);
    return ColumnName;

def TrainData_TimeIndex():
    return [29, 31, 33, 35, 36, 37, 39, 40];

def TestData_TimeIndex():
    return [30, 32, 34, 38];



if __name__ == '__main__':
    Input, Output = TrainData(if_train = True);
    print(Input[1:5, 0:2]);
    
    print('\n');