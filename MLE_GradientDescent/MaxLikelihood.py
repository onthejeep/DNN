import tensorflow as tf;
import numpy as np;
import os;
import matplotlib.pyplot as plt;
import PenaltyImpe;



tf.app.flags.DEFINE_float('mu1', 4, 'mode1-mu');
tf.app.flags.DEFINE_float('mu2', 7, 'mode2-mu');
tf.app.flags.DEFINE_float('mu3', 10, 'mode3-mu');

tf.app.flags.DEFINE_float('std1', 1, 'mode1-std');
tf.app.flags.DEFINE_float('std2', 2, 'mode2-std');
tf.app.flags.DEFINE_float('std3', 4, 'mode3-std');

tf.app.flags.DEFINE_float('w1', 0.6, 'mode1-std');
tf.app.flags.DEFINE_float('w2', 0.3, 'mode1-std');
tf.app.flags.DEFINE_float('w3', 0.1, 'mode1-std');

tf.app.flags.DEFINE_integer('datasize', 100000, 'data-size');
tf.app.flags.DEFINE_integer('number_mixture', 3, 'number of mixture component');

Flags = tf.app.flags.FLAGS;
np.random.seed(12345);
Data = np.random.normal(Flags.mu1, Flags.std1, size = int(Flags.datasize * Flags.w1));
Data = np.append(Data, np.random.normal(Flags.mu2, Flags.std2, size = int(Flags.datasize * Flags.w2)));
Data = np.append(Data, np.random.normal(Flags.mu3, Flags.std3, size = int(Flags.datasize * Flags.w3)));

#CENTER = np.mean(Data);
#SCALE = np.std(Data);
#Data = (Data - CENTER) / SCALE;



Mu1 = tf.Variable(3, trainable = True, dtype = tf.float32);
Mu2 = tf.Variable(8, trainable = True, dtype = tf.float32);
Mu3 = tf.Variable(12, trainable = True, dtype = tf.float32);
Phi1 = tf.Variable(1, trainable = True, dtype = tf.float32);
Phi2 = tf.Variable(1.2, trainable = True, dtype = tf.float32);
Phi3 = tf.Variable(2, trainable = True, dtype = tf.float32);
Std1 = tf.square(Phi1);
Std2 = tf.square(Phi2);
Std3 = tf.square(Phi3);

W1_Phi = tf.Variable(0.77, trainable = True, dtype = tf.float32);
W2_Phi = tf.Variable(0.5, trainable = True, dtype = tf.float32);
W3_Phi = tf.Variable(0.5, trainable = True, dtype = tf.float32);
W1 = tf.square(W1_Phi);
W2 = tf.square(W2_Phi);
W3 = tf.square(W3_Phi);

def NormalPdf(data, mu, std):
    Exponential = - tf.square(data - mu) / (2 * tf.square(std));
    Probability = 1 / (tf.sqrt(2 * np.pi * tf.square(std))) * tf.exp(Exponential);
    return Probability;

def Loglikelihood_SingleValue(data, k, mu, std, weight):
    Result = 0;
    for i in range(k):
        Result += weight[i] * NormalPdf(data, mu[i], std[i]);
    return tf.log(Result);

print('Define placeholder');
Data_Placeholder = tf.placeholder(dtype = tf.float32);
LearningRate_Placeholder = tf.placeholder(tf.float32, shape = ());


Hypothesis = Loglikelihood_SingleValue(Data_Placeholder, k = Flags.number_mixture, mu = [Mu1, Mu2, Mu3], std = [Std1, Std2, Std3], weight = [W1, W2, W3]);
Cost = -1.0 * tf.reduce_sum(Hypothesis) / Flags.datasize;
Cost += PenaltyImpe.Penalty.PenaltyFunction(weights = [W1, W2, W3], iterationIndex = 1);

Optimizer = tf.train.AdamOptimizer(LearningRate_Placeholder).minimize(Cost);

print('Start initializing variables ... ');
Cost_Eval = [];

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer());

    LearningRate_Decay = np.repeat(0.01, 200);
    LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.005, 100));
    LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.002, 100));
    LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.001, 100));
    TrainStep = len(LearningRate_Decay);

    for i in range(TrainStep):
        C, _, Mu1_Eval, Mu2_Eval, Mu3_Eval, \
        Std1_Eval, Std2_Eval, Std3_Eval, \
        W1_Eval, W2_Eval, W3_Eval = Sess.run([Cost, Optimizer, Mu1, Mu2, Mu3, Std1, Std2, Std3, W1, W2, W3],
                                             feed_dict = {Data_Placeholder: Data,
                                                          LearningRate_Placeholder: LearningRate_Decay[i]
                                                          });
        Cost_Eval.append(C);

        print(i);
        print('Cost = ', C);
        print('Mu: ', Mu1_Eval, Mu2_Eval, Mu3_Eval);
        print('Std: ', Std1_Eval, Std2_Eval, Std3_Eval);
        print('Weight: ', W1_Eval, W2_Eval, W3_Eval);

plt.plot(Cost_Eval);
plt.show();



