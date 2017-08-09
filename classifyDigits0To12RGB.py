"""ConvMT: a convolutional neural network for classifying micrographs into groups with and without MTs"""
#imports associated with tensorflow tutorial: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import tensorflow as tf
import time # 

tf.logging.set_verbosity(tf.logging.INFO)
# Essential Imports
# math etc.
from scipy import misc
import numpy as np
#import scipy as spf

#plotting
import matplotlib as mpl
from matplotlib import pyplot as plt

#image functions, esp. resizing
import cv2
#directory functions
import os

#for import .mat files
from scipy import io



# Describe the network architecture
# The layers will be input -> reLu(conv1) -> pooling1 -> reLU(conv2) -> pooling2 -> hiddenLayer > logitsLayer
# Input images are 100x100 px FOVs from VE-BF or TIE microscopy of microtubules
# reference: https://www.tensorflow.org/tutorials/layers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('maxSteps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('dropout', 0.1,
                            """proportion of weights to drop out.""")
tf.app.flags.DEFINE_float('learningRate', 1e-3,
                            """Learning Rate.""")
tf.app.flags.DEFINE_integer('batchSize', 66,
                            """Minibatch size""")
tf.app.flags.DEFINE_integer('mySeed',1337,"""pseudorandom number gen. seed""")

tf.app.flags.DEFINE_integer('iRGB',3,"""color channels (3 for RGB images, change to 1 for grayscale)""")

lR = FLAGS.learningRate
dORate = FLAGS.dropout
maxSteps = FLAGS.maxSteps
batchSize = FLAGS.batchSize
mySeed = FLAGS.mySeed

# Graph parameters
numLabels = 13
convDepth = 4
imgHeight = 48
imgWidth = 64
#imgWidth = 64		
pool1Size = 4
pool2Size = 4
kern1Size = 10
kern2Size = 5
iRGB = 3
# logit size to 4 choices
# hidden layer to 784 (from 1024)
# second hidden layer (copy of first)

nVisible = imgHeight * imgWidth
nHiddenDense = 1024
nFlatPool = round(imgHeight/pool1Size/pool2Size)
nPOOL = 96

# learning parameters
#lR = 1e-2 # learning rate
#batchSize = 79


        
def cNNMTModel(data, labels, mode):
    # mode is a boolean that determines whether to apply dropout (for training)
    # or keep all layers (evaluation/test data)
    #inputLayer = tf.reshape(data, [-1,100,100,1])
    #print(np.shape(data))
    inputLayer = tf.reshape(data, [-1, imgHeight, imgWidth, iRGB])
    #print(np.shape(data))
    # First convolutional layer and pooling 
    conv1 = tf.layers.conv2d(
        inputs = inputLayer,
        filters = convDepth,
        kernel_size = [kern1Size,kern1Size],
        padding = "same",
        activation = tf.nn.relu)
    # pooling (reduce size for faster learning)
    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = pool1Size,
        strides = pool1Size)
    # Second convo layer and pooling
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters = convDepth*2,
        kernel_size = [kern2Size,kern2Size],
        padding = "same",
        activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = pool2Size, # for square pool sizes can specify single number as size
        strides = pool2Size)
    
    # dense layers
    # 5x5 depends on the max pooling. Here I've using max_pool2d with 
    # pool sizes of 5 and 4, so the dimension should be 100/5/4 = 5
    #
    print(np.shape(pool2))

    pool2Flat = tf.reshape(pool2,
                           [-1,
                            nPOOL])
    
#    pool2Flat = tf.reshape(pool2,
 #                          [-1,
  #                          nFlatPool*nFlatPool*convDepth*2])
    denseHL1 = tf.layers.dense(inputs=pool2Flat,
                             units=nHiddenDense,
                             activation=tf.nn.relu)
    
    # If mode is True, than apply dropout (training)
    dropout1 = tf.layers.dropout(inputs=denseHL1,
                               rate=dORate,
                               training = mode == learn.ModeKeys.TRAIN)

    denseHL2 = tf.layers.dense(inputs=dropout1,
                             units=nHiddenDense,
                             activation=tf.nn.relu)
    
    # If mode is True, than apply dropout (training)
    dropout2 = tf.layers.dropout(inputs=denseHL1,
                               rate=dORate,
                               training = mode == learn.ModeKeys.TRAIN)


    denseHL3 = tf.layers.dense(inputs=dropout2,
                             units=nHiddenDense,
                             activation=tf.nn.relu)
    
    # If mode is True, than apply dropout (training)
    dropout3 = tf.layers.dropout(inputs=denseHL2,
                               rate=dORate,
                               training = mode == learn.ModeKeys.TRAIN)



    denseHL4 = tf.layers.dense(inputs=dropout3,
                             units=nHiddenDense,
                             activation=tf.nn.relu)
    
    # If mode is True, than apply dropout (training)
    dropout4 = tf.layers.dropout(inputs=denseHL3,
                               rate=dORate,
                               training = mode == learn.ModeKeys.TRAIN)


    logits = tf.layers.dense(inputs=dropout4,
                             units=numLabels) # 2 units for 2 classes: w & w/o MTs
    

    # loss and training op are None
    loss = None
    #trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss)
    trainOp = None
    
    # Loss for TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
        oneHotLabels = tf.one_hot(indices = tf.cast(labels,tf.int32),
                                 depth=numLabels) 
        
        # Because my labels are already one hot (not indexes), don't have to call tf.one_hot
        #oneHotLabels = labels#nm
        loss = tf.losses.softmax_cross_entropy(onehot_labels = oneHotLabels,
                                              logits = logits)
        
        #tf.summary.scalar('cross_entropy', loss)

    # Training op
    if mode == learn.ModeKeys.TRAIN:
        #trainOp = tf.train.MomentumOptimizer(lR,mom).minimize(loss,global_step = tf.contrib.framework.get_global_step())
        trainOp = tf.train.AdamOptimizer(learning_rate=lR,beta1=0.9,beta2 = 0.999,epsilon=1e-08,use_locking=False,name='Adam').minimize(loss,global_step = tf.contrib.framework.get_global_step())
        #trainOp = tf.contrib.layers.optimize_loss(
        #loss = loss,
        #global_step = tf.contrib.framework.get_global_step(),
        #learning_rate = lR,
        #optimizer = "SGD")
    
    # Gen. Pred.
    predictions = {
        "classes": tf.argmax(
        input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name = "softmaxTensor")
    }


    # attach summaries for tensorboad https://www.tensorflow.org/get_started/summaries_and_tensorboard

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss, train_op=trainOp)
init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.

#sess = tf.Session()
#sess.run(init)
def main(unused_argv):
    # Load the training data
   

    myData = np.load('./outdoors0to12/out012Imgs.npy')
    if(iRGB == 1):
        myData = myData[:,:,:,0:]
    myLabels = np.load('./outdoors0to12/out012Tgts.npy')
    


    if(1):
	    np.random.seed(mySeed)
	    np.random.shuffle(myData)
	    np.random.seed(mySeed)
	    np.random.shuffle(myLabels)

    print("data shape: ", np.shape(myData), " labels shape: ", np.shape(myLabels), "data mean: ", np.mean(myData))#,"data mean (FFT): ", np.mean(myData[:,:,:,1]))
    
    
    nSamples = np.shape(myLabels)[0]
    

    # Save 10% for the evaluations data set. 
    evalSamples = round(0.1*nSamples)
    testSamples = round(0.2*nSamples)

    # group and normalize the datasets
    trainData = np.array(myData[testSamples+1:nSamples,:],dtype="float32")
    trainLabels = np.array(myLabels[testSamples+1:nSamples,:],dtype="float32")
    trainData = (trainData - np.min(trainData) )/ np.max(trainData - np.min(trainData)) 
    

    evalData = np.array(myData[0:evalSamples,:],dtype="float32")
    evalLabels = np.array(myLabels[0:evalSamples,:],dtype="float32")				
    evalData = (evalData - np.min(evalData) )/ np.max(evalData - np.min(evalData)) 
    
    testData = np.array(myData[evalSamples+1:testSamples,:],dtype="float32")
    testLabels = np.array(myLabels[evalSamples+1:testSamples,:],dtype="float32")           
    testData = (testData - np.min(testData) )/ np.max(testData - np.min(testData)) 
    
    print("labels shape (training): ", np.shape(trainLabels)," (evaluation): ", np.shape(evalLabels))
    print("mean value for evaluation labels (loaded coin-flip score): ", np.mean(evalLabels))
    sTime = time.time()
    # Create estimator
    MTClassifier = learn.Estimator(model_fn = cNNMTModel,
                                   model_dir = "./outdoors0to12ModelRGB/model",
                                   config=tf.contrib.learn.RunConfig(save_checkpoints_secs=50))
    
    # Metrics for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                           prediction_key="classes")
              }
   
    # set up logging
    tensors_to_log = {"probabilities": "softmaxTensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                                              every_n_iter = 10000)
    validationMetrics = {
    "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                           prediction_key="classes"),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

    #evalua during training and stop early if necessary
    validationMonitor = tf.contrib.learn.monitors.ValidationMonitor(
        evalData,
        evalLabels,
        every_n_steps=50,
        metrics=validationMetrics,
        early_stopping_metric="accuracy",
        early_stopping_metric_minimize=False,
        early_stopping_rounds=32000)


    # Train Model 
    MTClassifier.fit(x=trainData,
                    y=trainLabels,
                    batch_size = batchSize,
                    steps = maxSteps,
                    monitors = [validationMonitor])
    
    
    print(np.mean(evalLabels))
    print("elapsed time: ",time.time()-sTime)
    # Evaluate model and display results
    evalResults = MTClassifier.evaluate(x=evalData,
                                          y=evalLabels,
                                          metrics=metrics)
    print("Final cross-validation results", evalResults)
    testResults = MTClassifier.evaluate(x=testData,
                                          y=testLabels,
                                          metrics=metrics)
    print("final results with test data (not seen during training, building of model):",testResults)
    if (0): #uncomment the following to get classification examples 
        np.random.seed(16)
        plt.figure()
        
        for c in range(1,10):
            plt.subplot(3,3,c)
            myDemo = int(round(np.random.random() * 63))
            plt.imshow(testData[myDemo],cmap="gray")
            for ck in range(0,12):
                if(testLabels[myDemo] == ck):
                    plt.title("Digit recog as " + str(ck))
        
        plt.figure()
        for c in range(1,10):
            plt.subplot(3,3,c)
            myDemo = int(round(np.random.random() * 63))
            plt.imshow(testData[myDemo],cmap="gray")
            for ck in range(0,12):
                if(testLabels[myDemo] == ck):
                    plt.title("Digit recog as " + str(ck))
            
                
        plt.show()

if __name__ == "__main__":
    tf.app.run()


