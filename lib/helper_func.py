"""
Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

import tensorflow as tf
import pandas as pd
import sklearn
import numpy as np
import math
import random
import os

                
"""Helper functions"""
# Define functions for initializing variables and standard layers
# For now, this seems superfluous, but in extending the code to many more 
# layers, this will keep our code more readable

def train_valid_data(input_data, input_target, valid_ratio):
    '''
    A helper function to split the data into training, validation and testing sets
    :param input_data: 4D tensor (samples x channels x height x width)
    :param input_target: 2D tensor for the target labels of the samples-
    :param valid_ratio: ratio of validation samples from full (or training) samples
    :param test_ratio: ratio of test samples from full samples
    :param test_set: if there is a need to split testing set
    :return: 4D tensors (data_train, data_valid, and/or data_test) 
             2D tensors (target_train, target_valid, or target_test)
    '''

    # determine the index of training, validation and testing sets
    # randomly sample a fixed ratio of control and patients to create testing set
    num_subj = input_target.shape[0]
    
    num_control = np.sum(input_target[:,0])
    num_patient = np.sum(input_target[:,1])
    control_index_all = [i for i,aa in enumerate(input_target[:,0]) if aa == 1]
#    control_valid_index = random.sample(control_index_all, int(math.floor(valid_ratio*num_control)))
    control_valid_index = control_index_all[0:int(math.floor(valid_ratio*num_control))] # special case for subjets with multiple scans arranged side-by-side
#    print('Control_valid_index: ', control_valid_index)
    
    patient_index_all = [i for i,aa in enumerate(input_target[:,1]) if aa == 1] 
#    patient_valid_index = random.sample(patient_index_all, int(math.floor(valid_ratio*num_patient)))
    patient_valid_index = patient_index_all[0:int(math.floor(valid_ratio*num_patient))] # special case for subjets with multiple scans arranged side-by-side

    # validation set
    data_valid_index = np.concatenate((control_valid_index, patient_valid_index), axis=0)
    data_valid = input_data[data_valid_index]
    target_valid = input_target[data_valid_index]
    
    # randomly sample a fixed ratio of control and patients to create training and validation sets
    data_train_index = np.delete(np.arange(num_subj), data_valid_index)    
    data_train = input_data[data_train_index]
    target_train = input_target[data_train_index]
    
    print('Number of control: ', num_control)
    print('Number of patient: ', num_patient)
    
#    return Bunch(data_train=data_train, target_train=target_train, data_valid=data_valid, target_valid=target_valid) # early prediction                   
    return data_train, target_train, data_valid, target_valid # early prediction                   

def train_valid_data_comb(input_data1, input_data2, input_target, valid_ratio):
    '''
    A helper function to split the data into training, validation and testing sets
    :param input_data: 4D tensor (samples x channels x height x width)
    :param input_target: 2D tensor for the target labels of the samples-
    :param valid_ratio: ratio of validation samples from full (or training) samples
    :param test_ratio: ratio of test samples from full samples
    :param test_set: if there is a need to split testing set
    :return: 4D tensors (data_train, data_valid, and/or data_test) 
             2D tensors (target_train, target_valid, or target_test)
    '''

    # determine the index of training, validation and testing sets
    # randomly sample a fixed ratio of control and patients to create testing set
    num_subj = input_target.shape[0]
    
    num_control = np.sum(input_target[:,0])
    num_patient = np.sum(input_target[:,1])
    control_index_all = [i for i,aa in enumerate(input_target[:,0]) if aa == 1]
    control_valid_index = random.sample(control_index_all, int(math.floor(valid_ratio*num_control)))
    patient_index_all = [i for i,aa in enumerate(input_target[:,1]) if aa == 1] 
    patient_valid_index = random.sample(patient_index_all, int(math.floor(valid_ratio*num_patient)))

    # validation set
    data_valid_index = np.concatenate((control_valid_index, patient_valid_index), axis=0)
    data_valid1 = input_data1[data_valid_index]
    data_valid2 = input_data2[data_valid_index]
    target_valid = input_target[data_valid_index]
    
    # randomly sample a fixed ratio of control and patients to create training and validation sets
    data_train_index = np.delete(np.arange(num_subj), data_valid_index)    
    data_train1 = input_data1[data_train_index]
    data_train2 = input_data2[data_train_index]
    target_train = input_target[data_train_index]
    
    print('Number of control: ', num_control)
    print('Number of patient: ', num_patient)
    return data_train1, data_train2, target_train, data_valid1, data_valid2, target_valid # early prediction           

# mixup: data augmentation using weighted sum of 2 images (not restricted to the same class)
def mixup_image(data, labels, alpha, batch_size):
    class_num = np.unique(labels)
    one_hot_labels = np.eye(class_num)[labels]  # one hot coding

    # mixup implementation:
    # Note that for larger images, it's more efficient to do mixup on GPUs (i.e. in the graph)
    weight = np.random.beta(alpha, alpha, batch_size) # generate weight for "batch_size number of samples
    x_weight = weight.reshape(batch_size, 1, 1, 1)
    y_weight = weight.reshape(batch_size, 1)
    index = np.random.permutation(batch_size)

    x1, x2 = data, data[index]
    x = x1 * x_weight + x2 * (1 - x_weight)
    y1, y2 = one_hot_labels, one_hot_labels[index]
    y = y1 * y_weight + y2 * (1 - y_weight)
    return [x, y]

# mixup: data augmentation using weighted sum of 2 samples (not restricted to the same class)
def mixup_data(data, labels, alpha, batch_size):
    if alpha > 0.:
        weight = np.random.beta(alpha, alpha, batch_size)
    else:
        weight = np.ones(batch_size)

    # mixup implementation:
    # Note that for larger images, it's more efficient to do mixup on GPUs (i.e. in the graph)
    x_weight = weight.reshape(batch_size, 1)
    y_weight = weight.reshape(batch_size, 1)
    index = np.random.permutation(batch_size)

    x1, x2 = data, data[index]
    x_mixed = x1 * x_weight + x2 * (1 - x_weight)
    y1, y2 = labels, labels[index]
    y12 = y1 * y_weight + y2 * (1 - y_weight)
    y_mixed = y12
    x = np.concatenate((data, x_mixed), axis=0)      # combine the original and mixed samples
    y = np.concatenate((labels, y_mixed), axis=0)
    return [x, y]
        

def top_k_error_tf(labels, predictions, prevalence, k):
    '''
    Calculate the top-k error in tensorflow format
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    '''
    cm = tf.confusion_matrix(labels, predictions)

    # error 
    zero = tf.constant(0, dtype=tf.int32)
    neum_err = tf.placeholder(tf.int32, shape=[], name="condition")
    neum_err = tf.add(cm[1][0], cm[0][1])
    den_err1 = tf.add(cm[0][0], cm[1][1])
    den_err2 = tf.add(cm[1][0], cm[0][1])
    deno_err = tf.add(den_err1, den_err2)
    error = tf.cond( tf.equal(neum_err, zero), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(neum_err, deno_err) )
    
    # sensitivity
    zero = tf.constant(0, dtype=tf.int32)
    deno_sen = tf.placeholder(tf.int32, shape=[], name="condition")
    deno_sen = tf.add(cm[1][0], cm[1][1])
    sensitivity = tf.cond( tf.equal(deno_sen, zero), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(cm[1][1], deno_sen) )
        
    # specificity
    deno_spe = tf.placeholder(tf.int32, shape=[], name="condition")
    deno_spe = tf.add(cm[0][0], cm[0][1])
    specificity = tf.cond( tf.equal(deno_spe, zero), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(cm[0][0], deno_spe) )
    
    # precision
    deno_pre = tf.placeholder(tf.int32, shape=[], name="condition")
    deno_pre = tf.add(cm[1][1], cm[0][1])
    precision = tf.cond( tf.equal(deno_pre, zero), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(cm[1][1], deno_pre) )

    # fscore
    zero_float = tf.constant(0, dtype=tf.float64)
    nume = tf.placeholder(tf.float64, shape=[], name="condition")
    nume = tf.multiply(sensitivity, precision)
    f1 = tf.cond(tf.equal(nume, zero_float), lambda: tf.constant(0, dtype=tf.float64), 
                 lambda: tf.divide( tf.multiply(sensitivity, precision), tf.add(sensitivity, precision) ))
    fscore = tf.to_float( tf.multiply(f1, 2.0) )

    # ppv
    nume_ppv = tf.placeholder(tf.float64, shape=[], name="condition")
    nume_ppv = tf.multiply(sensitivity, prevalence)
    deno_ppv = tf.placeholder(tf.float64, shape=[], name="condition")
    deno_ppv = tf.multiply(1.0-specificity, 1.0-prevalence)
    ppv = tf.cond( tf.equal(tf.add(nume_ppv, deno_ppv), zero_float), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(nume_ppv, tf.add(nume_ppv, deno_ppv)) )

    # npv
    nume_npv = tf.placeholder(tf.float64, shape=[], name="condition")
    nume_npv = tf.multiply(specificity, 1.0-prevalence)
    deno_npv = tf.placeholder(tf.float64, shape=[], name="condition")
    deno_npv = tf.multiply(1.0-sensitivity, prevalence)
    npv = tf.cond( tf.equal(tf.add(nume_npv, deno_npv), zero_float), lambda: tf.constant(0, dtype=tf.float64), lambda: tf.divide(nume_npv, tf.add(nume_npv, deno_npv)) )
    
    # G-mean (geometric mean) = squared root of (sensitivity x specificity)
    gmean = tf.sqrt(tf.multiply(sensitivity, specificity))
    
    acc = 1. - error
    
    return acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean, predictions

def top_k_error(labels, predictions, prevalence, k):
    '''
    Calculate the top-k error in numpy format
    :param predictions: 2D array with shape [batch_size, num_labels]
    :param labels: 1D array with shape [batch_size, 1]
    :param k: int
    :return: array with shape [1]
    '''
    cm = sklearn.metrics.confusion_matrix(labels, predictions)

    # error 
    nume_err = np.add(cm[1][0], cm[0][1])
    den_err1 = np.add(cm[0][0], cm[1][1])
    den_err2 = np.add(cm[1][0], cm[0][1])
    deno_err = np.add(den_err1, den_err2)
    if nume_err == 0:
        error = 0.
    else:
        error = np.divide(nume_err, 1.*deno_err)
#    acc = 1. - error

    # accuracy
    acc = sklearn.metrics.accuracy_score(labels, predictions) 
    
    # sensitivity
    deno_sen = np.add(cm[1][0], cm[1][1])
    if deno_sen == 0:
        sensitivity = 0.
    else:
        sensitivity = np.divide(cm[1][1], 1.*deno_sen) 
        
    # specificity
    deno_spe = np.add(cm[0][0], cm[0][1])
    if deno_spe == 0:
        specificity = 0.
    else:
        specificity = np.divide(cm[0][0], 1.*deno_spe)
    
    # precision
    deno_pre = np.add(cm[1][1], cm[0][1])
    if deno_pre == 0:
        precision = 0.
    else:
        precision = np.divide(cm[1][1], 1.*deno_pre)

    # fscore
    nume = np.multiply(sensitivity, precision)
    if nume == 0.:
        fscore = 0.
    else:
        fscore = 2. * np.divide( np.multiply(sensitivity, precision), 1.*np.add(sensitivity, precision) )

    # ppv
    nume_ppv = np.multiply(sensitivity, prevalence)
    deno_ppv = np.multiply(1.0-specificity, 1.0-prevalence)
    if np.add(nume_ppv, deno_ppv) == 0:
        ppv = 0.
    else:
        ppv = np.divide(nume_ppv, np.add(nume_ppv, 1.*deno_ppv)) 

    # npv
    nume_npv = np.multiply(specificity, 1.0-prevalence)
    deno_npv = np.multiply(1.0-sensitivity, prevalence)
    if np.add(nume_npv, deno_npv) == 0:
        npv = 0.
    else:
        npv = np.divide(nume_npv, np.add(nume_npv, 1.*deno_npv)) 
    
    # G-mean (geometric mean) = squared root of (sensitivity x specificity)
    gmean = np.sqrt(np.multiply(sensitivity, specificity))
    
    return acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean

def generate_train_batch(train_data, train_labels, batch_size, patient_ratio, random_batch_sampling_train=False):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''

    if random_batch_sampling_train is True:
        # Construct mini-batch by random sampling of subjects from whole dataset
        random_idx = random.sample(np.arange(0, train_labels.shape[0]), batch_size)       
        batch_data = train_data[random_idx, ...]
        batch_label = train_labels[random_idx]
    else:
        control_index_all = [i for i,aa in enumerate(train_labels) if aa == 0]
        control_train_index = random.sample(control_index_all, int(math.floor((1.0-patient_ratio)*batch_size)))
        patient_index_all = [i for i,aa in enumerate(train_labels) if aa == 1] 
        patient_train_index = random.sample(patient_index_all, int(math.ceil(patient_ratio*batch_size)))

        # training batch set
        data_train_index = np.concatenate((control_train_index, patient_train_index), axis=0)
        batch_data = train_data[data_train_index, ...]
        batch_label = train_labels[data_train_index]

    return batch_data, batch_label

def downsample_with_function(b, down_factor, func=np.nanmean):
    pad_size = np.int(math.ceil(float(b.shape[1])/down_factor)*down_factor - b.shape[1])
    c = np.append(b[0,:], np.zeros(pad_size)*np.NaN)
    b_down = np.zeros([b.shape[0], np.int(c.shape[0]/down_factor)])
    for i in range(b.shape[0]):
        b_padded = np.append(b[i,:], np.zeros(pad_size)*np.NaN)
        b_down[i,:] = func(b_padded.reshape(-1,down_factor), axis=1)
    print('Shape of b_down: ', b_down.shape)
    return b_down

# define bias
def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, 1, 1], padding='SAME')

def max_pool_1d(x, pool_factor):
  return tf.nn.max_pool(x, ksize=[1, pool_factor, 1, 1],
                        strides=[1, pool_factor, 1, 1], padding='SAME')

def avg_pool_1d(x, pool_factor):
  return tf.nn.avg_pool(x, ksize=[1, pool_factor, 1, 1],
                        strides=[1, pool_factor, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def save_to_csv(idx, acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean, dir_name, dataset, mode='w', header=False):
  df = {'aaidx':idx, 'accuracy':acc, 'fscore':fscore, 'sensitivity':sensitivity, 'specificity':specificity, 
                      'precision':precision, 'ppv':ppv, 'npv':npv, 'gmean':gmean}
  df = pd.DataFrame(data=df, index=[0])
  df.to_csv(dir_name, mode=mode, header=header)
  

def restore_nn(dir_name, folder): # restore a pretrained dnn model
    pretrained_ckt = os.path.abspath(os.path.join(dir_name, folder))
    filename = tf.train.latest_checkpoint(pretrained_ckt)
    graph = tf.Graph()
    with graph.as_default():
        new_saver = tf.train.import_meta_graph(filename + '.meta')    

    sess = tf.Session(graph=graph)    
    with sess.as_default():
        with graph.as_default():
            aa = new_saver.restore(sess, filename)
            return aa

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def print_classification_performance(dataset, prevalence, loss, acc, fscore, sensitivity, specificity, precision, ppv, npv, gmean):
  print( '----------------------------')
  print( 'Classification performance for %s: ' % dataset)
  print( 'Validation prevalence = %.4f' % prevalence)
  print( 'Validation loss = %.4f' % loss)
  print( 'Validation top1 accuracy = %.4f' % acc)
  print( 'Validation top1 fscore = %.4f' % fscore)
  print( 'Validation top1 sensitivity = %.4f' % sensitivity)
  print( 'Validation top1 specificity = %.4f' % specificity)
  print( 'Validation top1 precision = %.4f' % precision)
  print( 'Validation top1 ppv = %.4f' % ppv)
  print( 'Validation top1 npv = %.4f' % npv)
  print( 'Validation top1 gmean = %.4f' % gmean)
  print( '----------------------------')
      