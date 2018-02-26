import numpy as np;
import tensorflow as tf;


class Penalty:
    def __init__(self):
        pass;

    #Theta_Thres1 = tf.constant(0.001);
    #Theta_Thres2 = tf.constant(0.1);
    #Theta_Thres3 = tf.constant(1);
    def Theta_Result1(): return tf.constant(0.05, dtype = tf.float32);
    def Theta_Result2(): return tf.constant(0.1, dtype = tf.float32);
    def Theta_Result3(): return tf.constant(1, dtype = tf.float32);
    def Theta_Result4(): return tf.constant(2, dtype = tf.float32);
    def Theta_Result5(): return tf.constant(3, dtype = tf.float32);

    def PenaltyFunction(weights, iterationIndex):
        Result = Penalty.h_function(iterationIndex) * Penalty.H_function(weights);
        return(Result);


    def theta_function(q):

        Comparison = tf.case({tf.less(q, 0.000001): Penalty.Theta_Result1,
                              tf.logical_and(tf.greater(q, 0.000001), tf.less(q, 0.00001)): Penalty.Theta_Result2,
                              tf.logical_and(tf.greater(q, 0.00001), tf.less(q, 0.001)): Penalty.Theta_Result3,
                              tf.logical_and(tf.greater(q, 0.001), tf.less(q, 1)): Penalty.Theta_Result4}, 
                             default = Penalty.Theta_Result4, exclusive = True);
        return Comparison;

    def gamma_function(q):
        Comparison = tf.case({tf.less(q, 1.0): lambda: tf.constant(1.0)}, default = lambda: tf.constant(2.0), exclusive = True);
        return Comparison;

    def Constraints(weights):
        return tf.reduce_sum(weights) - 1.00, 1.00 - tf.reduce_sum(weights);


    def q_function(weights):
        Gs = Penalty.Constraints(weights);
        return tf.reduce_max([tf.constant(1e-8), Gs[0]]), tf.reduce_max([tf.constant(1e-8), Gs[1]]);


    def H_function(weight):
        Q1, Q2 = Penalty.q_function(weight);
        
        Result = Penalty.theta_function(Q1) * tf.pow(Q1, Penalty.gamma_function(Q1)) + \
            Penalty.theta_function(Q2) * tf.pow(Q2, Penalty.gamma_function(Q2));
        return Result;

    def h_function(iterationIndex):
        """
        the function implements h(k)
        iterationIndex is k
        Important note: h(k) is set to be 1 in this code
        """
        Result = tf.constant(1.00, dtype = tf.float32); #iterationIndex**1.0;
        return(Result);