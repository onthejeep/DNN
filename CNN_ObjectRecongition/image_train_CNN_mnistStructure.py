import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join

# https://github.com/yeephycho/tensorflow_input_image_by_tfrecord/blob/master/src/flower_train_cnn.py

tf.app.flags.DEFINE_string('data_dir', 'data/image', 'help1');
tf.app.flags.DEFINE_integer('training_size', 300, 'help2');
tf.app.flags.DEFINE_integer('batch_size', 50, 'help3');
tf.app.flags.DEFINE_integer('image_size', 128, 'help4');
tf.app.flags.DEFINE_integer('class_number', 4, 'help5');

FLAGS = tf.app.flags.FLAGS;

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]));

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]));

class ImageObject:
    def __init__(self):
        self.Image = tf.Variable([], dtype = tf.string);
        self.Height = tf.Variable([], dtype = tf.int64);
        self.Width = tf.Variable([], dtype = tf.int64);
        self.Filename = tf.Variable([], dtype = tf.string);
        self.Label = tf.Variable([], dtype = tf.int32);
        self.Text = tf.Variable([], dtype = tf.string);

def Read_and_Decode(filename_queue):
    RecordReader = tf.TFRecordReader()
    _, SerializedExample = RecordReader.read(filename_queue);

    ImageFeatures = tf.parse_single_example(SerializedExample, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),
        'image/class/text': tf.FixedLenFeature([], tf.string)}
                                       );

    ImageEncoded = ImageFeatures["image/encoded"];
    ImageRaw = tf.image.decode_jpeg(ImageEncoded, channels=3);
    ImgObj = ImageObject();
    ImgObj.Image = tf.image.resize_images(ImageRaw, [FLAGS.image_size, FLAGS.image_size], tf.image.ResizeMethod.BICUBIC);
    ImgObj.Height = ImageFeatures["image/height"];
    ImgObj.Width = ImageFeatures["image/width"];
    ImgObj.FileName = ImageFeatures["image/filename"];
    ImgObj.Label = tf.cast(ImageFeatures["image/class/label"], tf.int64);
    ImgObj.Text = ImageFeatures['image/class/text'];

    return ImgObj;

def ImageInputBatch(if_random = True, if_training = True, batchSize = FLAGS.batch_size):
    if(if_training):
        FileNames = [os.path.join(FLAGS.data_dir, "train-0000%d-of-00002.tfdata" % i) for i in range(0, 2)];
    else:
        FileNames = [os.path.join(FLAGS.data_dir, "eval-0000%d-of-00002.tfdata" % i) for i in range(0, 2)];

    for file in FileNames:
        if not tf.gfile.Exists(file):
            raise ValueError("Failed to find file: " + file);

    # Part 1: Read data from *.tfdata
    FileName_Queue = tf.train.string_input_producer(FileNames);
    ImgObj = Read_and_Decode(FileName_Queue);

    Image = tf.image.per_image_standardization(ImgObj.Image);
    Label = ImgObj.Label;
    FileName = ImgObj.FileName;

    # Part 2: Create batch (pipeline)
    if(if_random):
        min_fraction_of_examples_in_queue = 0.4;
        min_queue_examples = int(FLAGS.training_size * min_fraction_of_examples_in_queue);
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples);

        ImageBatch, LabelBatch, FileNameBatch = tf.train.shuffle_batch([Image, Label, FileName], batch_size = FLAGS.batch_size, num_threads = 1,
                                                                       capacity = min_queue_examples + 3 * FLAGS.batch_size, min_after_dequeue = min_queue_examples);
        return ImageBatch, LabelBatch, FileNameBatch;
    else:
        ImageBatch, LabelBatch, FileNameBatch = tf.train.batch([Image, Label, FileName], batch_size = batchSize, num_threads = 1, allow_smaller_final_batch = True); # 
        return ImageBatch, LabelBatch, FileNameBatch;



def InitializeWeight(shape):
    ''' Template initialization
    '''
    Weight = tf.truncated_normal(shape, stddev = 0.03);
    return tf.Variable(Weight);

def InitializeBias(shape):
    Bias = tf.random_normal(shape, stddev = 0.03);
    return tf.Variable(Bias);

def Convolution2D(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME');

def MaxPool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME');

def CnnDefinition(imageBatch):
    
    W2 = InitializeWeight([4, 4, 3, 16]);
    b2 = InitializeBias([16]);
    L2 = tf.nn.relu(Convolution2D(imageBatch, W2) + b2);
    L2 = MaxPool_2x2(L2); # 64
    
    W3 = InitializeWeight([4, 4, 16, 32]);
    b3 = InitializeBias([32]);
    L3 = tf.nn.relu(Convolution2D(L2, W3) + b3);
    L3 = MaxPool_2x2(L3); # 32
    # L3 = tf.nn.lrn(L3, 4, bias = 1.0, alpha = 0.001 / 4.0, beta = 0.75, name = 'norm3');

    W4 = InitializeWeight([2, 2, 32, 64]);
    b4 = InitializeBias([64]);
    L4 = tf.nn.relu(Convolution2D(L3, W4) + b4);
    L4 = MaxPool_2x2(L4); # 16
    # L4 = tf.nn.lrn(L4, 4, bias = 1.0, alpha = 0.001 / 4.0, beta = 0.75, name = 'norm4');

    W5 = InitializeWeight([2, 2, 64, 128]);
    b5 = InitializeBias([128]);
    L5 = tf.nn.relu(Convolution2D(L4, W5) + b5);
    L5 = MaxPool_2x2(L5); # 8

    # L5 = tf.nn.lrn(L5, 4, bias = 1.0, alpha = 0.001 / 4.0, beta = 0.75, name = 'norm5');
    L5_Flat = tf.reshape(L5, [-1, 8*8*128]);
    
    W1_FullConnection = tf.get_variable('W1_FullConnection', shape = [8 * 8 * 128, 512], initializer = tf.contrib.layers.xavier_initializer());
    b1_FullConnection = InitializeBias([512]);
    L1_FullConnection = tf.nn.sigmoid(tf.matmul(L5_Flat, W1_FullConnection) + b1_FullConnection);

    W2_FullConnection = tf.get_variable('W2_FullConnection', shape = [512, FLAGS.class_number], initializer = tf.contrib.layers.xavier_initializer());
    b2_FullConnection = InitializeBias([FLAGS.class_number]);
    
    Hypothesis = tf.matmul(L1_FullConnection, W2_FullConnection) + b2_FullConnection;
    #Hypothesis = tf.nn.sigmoid(Hypothesis);

    return Hypothesis;


def TrainCNN():
    # https://stackoverflow.com/questions/33610685/in-tensorflow-what-is-the-difference-between-session-run-and-tensor-eval
    print('Start training a Convolutional Neural Network ... ');
    TotalBatch = int(FLAGS.training_size / FLAGS.batch_size);

    Images_Placeholder = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3]); # FLAGS.batch_size
    Labels_Placeholder = tf.placeholder(tf.int64, shape = [None]);
    # Labels_Placeholder_Onehot = tf.one_hot(Labels_Placeholder, depth = FLAGS.class_number, on_value = 1.0, off_value = 0.0);

    print('Define flow of tensors ... ');
    Hypothesis = CnnDefinition(Images_Placeholder);
    # Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Hypothesis, labels = Labels_Placeholder_Onehot));
    Cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = Hypothesis, labels = Labels_Placeholder));

    LearningRate = tf.placeholder(tf.float32, shape = ());
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Cost);

    ModelSaver = tf.train.Saver();

    print('Start initializing variables ... ');
    with tf.Session() as Sess:
        Sess.run(tf.global_variables_initializer());

        print('Create batches of images and labels ... ');
        ImageBatch, LabelBatch, FileNameBatch = ImageInputBatch(if_random = False, if_training = True, batchSize = FLAGS.batch_size);

        Coord = tf.train.Coordinator();
        Threads = tf.train.start_queue_runners(coord = Coord, sess = Sess);

        LearningRate_Decay = np.repeat(0.001, 200);
        LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.0001, 1000));
        LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.00001, 1000));
        # LearningRate_Decay = np.append(LearningRate_Decay, np.repeat(0.000001, 1000));

        print(LearningRate_Decay);
        TrainingEpochs = len(LearningRate_Decay);

        print('Start training ... ');
        for epoch in range(TrainingEpochs):
            AverageCost = 0;
            for batch in range(TotalBatch):

                ImageBatch_Eval, LabelBatch_Eval, FileName_Eval = Sess.run([ImageBatch, LabelBatch, FileNameBatch]);

                _, C, H = Sess.run([Optimizer, Cost, Hypothesis], feed_dict = {Images_Placeholder: ImageBatch_Eval, 
                                                                               Labels_Placeholder: LabelBatch_Eval,
                                                                               LearningRate: LearningRate_Decay[epoch]
                                                                               });
                AverageCost += C / TotalBatch;

            print('Epoch:', '%04d' % (epoch + 1), 'Cost = ', '{:0.9f}'.format(AverageCost));
            # print('Hypothesis result {}'.format(H[0]));
            ModelSaver.save(Sess, 'data/image/CNN.ckpt');

        Coord.request_stop();
        Coord.join(Threads);



def EvaluationCNN():

    BatchSize = 300;

    Images_Placeholder = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3]); # FLAGS.batch_size
    Labels_Placeholder = tf.placeholder(tf.int64, shape = [None]);

    print('Define flow of tensors ... ');
    Hypothesis = CnnDefinition(Images_Placeholder);
    PredictionLable = tf.arg_max(Hypothesis, 1);
    CorrectPrediction = tf.equal(PredictionLable, Labels_Placeholder);
    Accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32));

    ModelSaver = tf.train.Saver();

    ImageBatch, LabelBatch, FileNameBatch = ImageInputBatch(if_random = False, if_training = True, batchSize = BatchSize);

    with tf.Session() as Sess:
        Sess.run(tf.global_variables_initializer());
        ModelSaver.restore(Sess, 'data/image/CNN.ckpt');

        Coord = tf.train.Coordinator();
        Threads = tf.train.start_queue_runners(coord = Coord, sess = Sess);

        ImageBatch_Eval, LabelBatch_Eval, FileName_Eval = Sess.run([ImageBatch, LabelBatch, FileNameBatch]);

        AccuracyEval, Probability, PreLabel = Sess.run([Accuracy, Hypothesis, PredictionLable], feed_dict = {
            Images_Placeholder: ImageBatch_Eval, 
            Labels_Placeholder: LabelBatch_Eval});

        print('Accuracy = ', AccuracyEval);
        print('FileName', FileName_Eval);
        print('Groudtruth: ', LabelBatch_Eval);
        print('Probability: ', Probability);
        print('Precition: ', PreLabel);

        Coord.request_stop();
        Coord.join(Threads);

if __name__ == '__main__':
    # TrainCNN();
    EvaluationCNN();

    #ImageBatch, LabelBatch, FileNameBatch = ImageInputBatch(if_random = False, if_training = True, batchSize = 20);

    #with tf.Session() as Sess:
    #    Sess.run(tf.global_variables_initializer());

    #    Coord = tf.train.Coordinator();
    #    Threads = tf.train.start_queue_runners(coord = Coord, sess = Sess);

    #    Label_Eval, FileName_Eval = Sess.run([LabelBatch, FileNameBatch]);
    #    print(Label_Eval, FileName_Eval);


    #    Coord.request_stop();
    #    Coord.join(Threads);
