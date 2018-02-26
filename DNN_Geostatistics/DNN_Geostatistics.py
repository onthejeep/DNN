import tensorflow as tf;
import numpy as np;
import TrainTestData;
import Scaler;


tf.app.flags.DEFINE_integer('NumNeural', 64, 'number of neurals in a hidden layer');
tf.app.flags.DEFINE_integer('NumHiddenLayer', 5, 'number of hidden layers in deep neural network (total number of hidden layers = NumHiddenLayer + 2)');
tf.app.flags.DEFINE_float('Dropout', 0.8, 'keep_prob in tf.nn.dropout()');
tf.app.flags.DEFINE_integer('NumInput', 5, 'number of input (TimeIndex, Col, Row, Manhattan, Euclid)');
tf.app.flags.DEFINE_integer('NumOutput', 1, 'number of output (avg, std, cov, p90, p95, bt, pt)');
tf.app.flags.DEFINE_integer('BatchSize', 64, 'batch size used for training');
tf.app.flags.DEFINE_integer('NumBootstrap', 100, 'number of bootstrap');

Flags = tf.app.flags.FLAGS;


def DNN_Geostatistics(input_placeholder, numNeural, numHiddenLayer, numStatistics):

    Weights = [None] * (Flags.NumHiddenLayer + 1);
    Bias = [None] * (Flags.NumHiddenLayer + 1);
    Layers = [None] * (Flags.NumHiddenLayer + 1);
    Weights[0] = tf.Variable(tf.random_normal([Flags.NumInput, numNeural], stddev= 0.05));
    Bias[0] = tf.Variable(tf.random_normal([numNeural], stddev= 0.05));
    Layers[0] = tf.nn.relu(tf.matmul(input_placeholder, Weights[0]) + Bias[0]);

    for i in range(1, Flags.NumHiddenLayer):
        Weights[i] = tf.Variable(tf.random_normal([numNeural, numNeural], stddev= 0.05));
        Bias[i] = tf.Variable(tf.random_normal([numNeural], stddev= 0.05));
        Layers[i] = tf.nn.relu(tf.matmul(Layers[i - 1], Weights[i]) + Bias[i]);

    Weights[Flags.NumHiddenLayer] = tf.Variable(tf.random_normal([numNeural, numStatistics], stddev= 0.05));
    Bias[Flags.NumHiddenLayer] = tf.Variable(tf.random_normal([numStatistics], stddev= 0.05));
    Hypothesis = tf.nn.sigmoid(tf.matmul(Layers[Flags.NumHiddenLayer - 1], \
        Weights[Flags.NumHiddenLayer]) + Bias[Flags.NumHiddenLayer]);
    # Hypothesis = tf.nn.dropout(Hypothesis, keep_prob = Flags.Dropout);

    return Hypothesis;



def TrainDNN(input, output, bootstrapIndex):
    print('define placehoder of input and output ... ');
    Input_Placeholder = tf.placeholder(tf.float32, shape = [None, Flags.NumInput]);
    Output_Placeholder = tf.placeholder(tf.float32, shape = [None, Flags.NumOutput]);
    LearningRate = tf.placeholder(tf.float32, shape = ());
    DataSize = input.shape[0];
    TotalBatch = int(DataSize / Flags.BatchSize) + 1;

    print('define flow of tensor ... ');
    Hypothesis = DNN_Geostatistics(Input_Placeholder, Flags.NumNeural, Flags.NumHiddenLayer, Flags.NumOutput);    
    Cost = tf.reduce_mean(tf.square(Hypothesis - Output_Placeholder));
    Optimization = tf.train.AdamOptimizer(LearningRate).minimize(Cost);

    ModelSaver = tf.train.Saver();
    print('start initializing variables ... ');
    with tf.Session() as Sess:
        Sess.run(tf.global_variables_initializer());

        print('create batches of data ... ');

        LearningRate_Decay = np.repeat(0.5, 200);
        LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.02, 200));
        LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.001, 200));

        TrainingEpochs = len(LearningRate_Decay);

        print('start learning ... ');
        for epoch in range(TrainingEpochs):
            AverageCost = 0;
            Shuffled = np.random.choice(DataSize, size = DataSize, replace = False);

            for batch in range(TotalBatch):
                StartIndex = batch * Flags.BatchSize;
                EndIndex = (batch + 1) * Flags.BatchSize - 1;
                if EndIndex >= DataSize:
                    EndIndex = DataSize - 1;

                RowIndex = Shuffled[StartIndex: EndIndex];
                InputBatch, OutputBatch = input[RowIndex, :], output[RowIndex, :];
                C, _ = Sess.run([Cost, Optimization], feed_dict= {Input_Placeholder: InputBatch,
                            Output_Placeholder: OutputBatch,
                            LearningRate: LearningRate_Decay[epoch]});
                
                AverageCost += C / TotalBatch;
            print('Epoch: {0:04d}   Cost = {1:0.9f}'.format(epoch + 1, AverageCost));
            ModelSaver.save(Sess, 'Result/DNN_{}.ckpt'.format(bootstrapIndex));




def PredictionDNN(input):
    print('define placehoder of input and output ... ');
    Input_Placeholder = tf.placeholder(tf.float32, shape = [None, Flags.NumInput]);
    print('define flow of tensor ... ');
    Hypothesis = DNN_Geostatistics(Input_Placeholder, Flags.NumNeural, Flags.NumHiddenLayer, Flags.NumOutput);

    ModelSaver = tf.train.Saver();
    with tf.Session() as Sess:
        Sess.run(tf.global_variables_initializer());
        ModelSaver.restore(Sess, 'Result/DNN_0.ckpt');

        Prediction = Sess.run(Hypothesis, feed_dict= {Input_Placeholder: input});

    return Prediction;

# 0.003667966
# 0.002768564
# 0.002209568
# 0.002357085
# 0.002186672

if __name__ == '__main__':
    Input, Output = TrainTestData.TrainData(if_train = True);
    # Input, Output = TrainTestData.Normalization(Input, Output);
    InputScaler, OutputScaler = Scaler.ComboScaler(Input), Scaler.ComboScaler(Output);
    Input_Trans, Output_Trans = Scaler.ComboScaler_Transform(Input, InputScaler), \
                Scaler.ComboScaler_Transform(Output, OutputScaler);
    

    TrainDNN(Input_Trans, Output_Trans, 0);

    # Prediction = PredictionDNN(Input_Trans);
    # Prediction_Inverse = Scaler.ComboScaler_Inverse(Prediction, OutputScaler);
    # MAE = np.mean(np.abs(Prediction_Inverse - Output), axis= 0);
    # MAPE = np.mean(np.abs(Prediction_Inverse / Output - 1), axis= 0) * 100;
    # RMSE = np.sqrt(np.mean(np.square(Prediction_Inverse - Output), axis= 0));
    # print(MAE, MAPE, RMSE);
    
