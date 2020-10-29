"""
Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

import tensorflow as tf
import pandas as pd
from datetime import datetime
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil

from lib import helper_func as hf
from lib import graph


# Common methods for all models


class base_model(object):
    
    def __init__(self):
        self.regularizers = []
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            
            batch_data = np.zeros((self.batch_size, data.shape[1]))
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin] = tmp_data
            feed_dict = {self.ph_data: batch_data, self.ph_dropout: 1}
            
            # Compute loss if labels are given.
            if labels is not None:
                if labels.ndim == 1:
                    batch_labels = np.zeros(self.batch_size)
                    batch_labels[:end-begin] = labels[begin:end]
                    feed_dict[self.ph_labels] = batch_labels
                    batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                else:
                    batch_labels = np.zeros((self.batch_size, labels.shape[1]))
                    batch_labels[:end-begin,:] = labels[begin:end,:]
                    feed_dict[self.ph_labels] = batch_labels
                    batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
#                    batch_pred = tf.argmax(batch_pred, axis=1)

                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss * self.batch_size / size
#            return predictions, loss / (1. * size)
        else:
            return predictions
        
    def evaluate(self, data, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
#        t_process, t_wall = time.process_time(), time.time() # version Python 3
        t_process, t_wall = time.clock(), time.time() # version Python 2
        predictions, loss = self.predict(data, labels, sess)
        #print(predictions)
        if labels.ndim > 1:
            labels = np.copy(labels[:,1])
        ncorrects = sum(predictions == labels)
        prevalence = np.sum(labels)*1.0 / labels.shape[0]
        accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean = self.top_k_error(labels, predictions, prevalence, 1)
        accuracy = 100 * accuracy
        f1 = 100 * f1
        string = 'accuracy: {:.2f} ({:d} / {:d}), f1 (weighted): {:.2f}, loss: {:.2e}'.format(
                accuracy, ncorrects, len(labels), f1, loss)
        if sess is None:
#            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall) # version Python 3
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.clock()-t_process, time.time()-t_wall) # version Python 2
        return string, loss, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, predictions

    def fit(self, train_data, train_labels, val_data, val_labels, alpha, learning_rate=0.1, num_epochs=20, decay_steps=None, finetune=False, finetune_fc=False, patient_ratio=0.5):
        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  
        config.gpu_options.per_process_gpu_memory_fraction=0.9
#        config = tf.ConfigProto(tf.ConfigProto(allow_soft_placement=True, log_device_placement=True), gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45), device_count = {'GPU': 1})
        sess = tf.Session(graph=self.graph, config=config)

        
        sess.run(self.op_init)
        
        ckpt = tf.train.get_checkpoint_state(self._get_path('checkpoints'))
#        
        # Load the latest model
        if finetune:
            # Restore from check point
            shutil.copytree(self._get_path('checkpoints'), self._get_path('checkpoints_orig'))
            path = os.path.join(self._get_path('checkpoints'), 'model')
            self.op_saver.restore(sess, ckpt.model_checkpoint_path)
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.decay_steps = decay_steps
            self.finetune_fc = finetune_fc
            
        else:
            shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
            shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
            os.makedirs(self._get_path('checkpoints'))
            path = os.path.join(self._get_path('checkpoints'), 'model')
            self.finetune_fc = finetune_fc
        
        # Start the queue runners
        tf.train.start_queue_runners(sess = sess)

        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)

        # Training.
        max_gmean = -1.0
        
        step_list = []
        train_accuracy_list = []
        train_fscore_list = []
        train_sensitivity_list = []
        train_specificity_list = []
        train_precision_list = []
        train_ppv_list = []
        train_npv_list = []
        train_gmean_list = []
        
        val_loss_list = []
        val_accuracy_list = []
        val_fscore_list = []
        val_sensitivity_list = []
        val_specificity_list = []
        val_precision_list = []
        val_ppv_list = []
        val_npv_list = []
        val_gmean_list = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if self.random_batch_sampling_train==False:
                if alpha > 0.:
                    train_labels_1d = np.copy(train_labels[:,1])
                    batch_data, batch_labels_1d = hf.generate_train_batch(train_data, train_labels_1d, self.batch_size/2, patient_ratio, random_batch_sampling_train=False)
                    class_num = np.unique(batch_labels_1d).shape[0]
                    batch_labels = np.eye(class_num)[batch_labels_1d]
                else:
                    batch_data, batch_labels = hf.generate_train_batch(train_data, train_labels, self.batch_size, patient_ratio, random_batch_sampling_train=False)
            else:
                if len(indices) < self.batch_size:
                    indices.extend(np.random.permutation(train_data.shape[0]))
                idx = [indices.popleft() for i in range(self.batch_size)]
                batch_data, batch_labels = train_data[idx,:], train_labels[idx]
            
            # Perform mixup (data augmentation) for training mini-batch 
            if alpha > 0.:
                batch_data, batch_labels = hf.mixup_data(batch_data, batch_labels, alpha, self.batch_size/2)
                
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
                
            feed_dict = {self.ph_data: batch_data, self.ph_labels: batch_labels, self.ph_dropout: self.dropout, self.ph_lr: self.learning_rate} # new learning rate for pretraining
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps or step == 1: # or nFlag==1:
                epoch = step * self.batch_size / train_data.shape[0]
                start_time = time.time()

                string, loss_train, acc_train, fscore_train, sensitivity_train, specificity_train, precision_train, ppv_train, npv_train, gmean_train, predictions = self.evaluate(train_data, train_labels, sess)
                string, loss_valid, acc_valid, fscore_valid, sensitivity_valid, specificity_valid, precision_valid, ppv_valid, npv_valid, gmean_valid, predictions = self.evaluate(val_data, val_labels, sess)

            # Save model parameters (for evaluation) with the higest gnean
                gmean_valid_new = min(gmean_valid, gmean_train)
                if gmean_valid_new > max_gmean:    
                    max_gmean = gmean_valid_new
                    self.op_saver.save(sess, path, write_state=True, global_step=step)
                else:
                    max_gmean = max_gmean*0.99995
                # Check performance of the training set
                step_list.append(step)
                train_accuracy_list.append(acc_train)
                train_fscore_list.append(fscore_train)
                train_sensitivity_list.append(sensitivity_train)
                train_specificity_list.append(specificity_train)
                train_precision_list.append(precision_train)
                train_ppv_list.append(ppv_train)
                train_npv_list.append(npv_train)
                train_gmean_list.append(gmean_train)
                
                duration = time.time() - start_time
                
                train_summ = tf.Summary()
                train_summ.value.add(tag='train_accuracy', simple_value=acc_train)
                train_summ.value.add(tag='train_fscore', simple_value=fscore_train)
                train_summ.value.add(tag='train_sensitivity', simple_value=sensitivity_train)
                train_summ.value.add(tag='train_specificity', simple_value=specificity_train)
                train_summ.value.add(tag='train_precision', simple_value=precision_train)
                train_summ.value.add(tag='train_ppv', simple_value=ppv_train)
                train_summ.value.add(tag='train_npv', simple_value=npv_train)
                train_summ.value.add(tag='train_gmean', simple_value=gmean_train)
                writer.add_summary(train_summ, step)
                writer.flush()
                

                # Check performance of the validation set
                val_loss_list.append(loss_valid)
                val_accuracy_list.append(acc_valid)
                val_fscore_list.append(fscore_valid)
                val_sensitivity_list.append(sensitivity_valid)
                val_specificity_list.append(specificity_valid)
                val_precision_list.append(precision_valid)
                val_ppv_list.append(ppv_valid)
                val_npv_list.append(npv_valid)
                val_gmean_list.append(gmean_valid)
      
                
                valid_summ = tf.Summary()
                valid_summ.value.add(tag='valid_accuracy', simple_value=acc_valid)
                valid_summ.value.add(tag='valid_fscore', simple_value=fscore_valid)
                valid_summ.value.add(tag='valid_sensitivity', simple_value=sensitivity_valid)
                valid_summ.value.add(tag='valid_specificity', simple_value=specificity_valid)
                valid_summ.value.add(tag='valid_precision', simple_value=precision_valid)
                valid_summ.value.add(tag='valid_ppv', simple_value=ppv_valid)
                valid_summ.value.add(tag='valid_npv', simple_value=npv_valid)
                valid_summ.value.add(tag='valid_gmean', simple_value=gmean_valid)
                writer.add_summary(valid_summ, step)
                writer.flush()
      
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                if batch_labels.ndim == 1:
                    prevalence = np.sum(batch_labels)*1.0/batch_labels.shape[0]
                else:
                    prevalence = np.sum(batch_labels[:,1]>0.5)*1.0/(1.*batch_labels.shape[0])
      
                print( '%s: ' % datetime.now())
                print('step: {} / {} (epoch: {:.2f} / {}):'.format(step, num_steps, int(epoch), self.num_epochs))
                print('learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                format_str =( '%.1f examples/sec; %.3f ' 'sec/batch')
                print( format_str % (examples_per_sec, sec_per_batch))
                print( 'Prevalence = %.4f' % prevalence)
                print( 'Train loss = %.4f' % loss_train)
                print( 'Train top1 accuracy = %.4f' % acc_train)
                print( 'Train top1 fscore = %.4f' % fscore_train)
                print( 'Train top1 sensitivity = %.4f' % sensitivity_train)
                print( 'Train top1 specificity = %.4f' % specificity_train)
                print( 'Train top1 precision = %.4f' % precision_train)
                print( 'Train top1 ppv = %.4f' % ppv_train)
                print( 'Train top1 npv = %.4f' % npv_train)
                print( 'Train top1 gmean = %.4f' % gmean_train)
                print( '*******')
                print( 'Validation loss = %.4f' % loss_valid)
                print( 'Validation top1 accuracy = %.4f' % acc_valid)
                print( 'Validation top1 fscore = %.4f' % fscore_valid)
                print( 'Validation top1 sensitivity = %.4f' % sensitivity_valid)
                print( 'Validation top1 specificity = %.4f' % specificity_valid)
                print( 'Validation top1 precision = %.4f' % precision_valid)
                print( 'Validation top1 ppv = %.4f' % ppv_valid)
                print( 'Validation top1 npv = %.4f' % npv_valid)
                print( 'Validation top1 gmean = %.4f' % gmean_valid)
                print( '----------------------------')
      
                df = pd.DataFrame(data={'step':step_list, 'train_accuracy':train_accuracy_list, 'train_fscore':train_fscore_list, 'train_sensitivity':train_sensitivity_list, 
                      'train_specificity':train_specificity_list, 'train_precision':train_precision_list, 'train_ppv':train_ppv_list, 'train_npv':train_npv_list,
                      'validation_accuracy': val_accuracy_list, 'validation_fscore': val_fscore_list, 'validation_sensitivity':val_sensitivity_list, 'validation_specificity':val_specificity_list, 
                      'validation_precision':val_precision_list, 'validation_ppv':val_ppv_list, 'validation_npv':val_npv_list, 'validation_gmean':val_gmean_list})
                df.to_csv(self._get_path('') + 'performance.csv')

        print('Validation accuracy: Peak = {:.2f}, Mean = {:.2f}'.format(max(val_accuracy_list), np.mean(val_accuracy_list[-10:])))
        print('Validation geometric accuracy: Peak = {:.2f}, Mean = {:.2f}'.format(max(val_gmean_list), np.mean(val_gmean_list[-10:])))
        writer.close()
        sess.close()
        
        return val_loss_list, val_accuracy_list, val_fscore_list, val_sensitivity_list, val_specificity_list, val_precision_list, val_ppv_list, val_npv_list, val_gmean_list


    def top_k_error(self, labels, predictions, prevalence, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
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
        acc = 1. - error
        
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

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0, C):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0), 'data')
                if C == 1:
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                else:    
                    self.ph_labels = tf.placeholder(tf.int32, (self.batch_size, C), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_lr = tf.placeholder(tf.float32, (), 'learning_rate')
#                self.ph_fc = tf.placeholder(tf.bool, (), 'finetune_fc')

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.ph_lr,
                    self.decay_steps, self.decay_rate, self.momentum, self.finetune_fc) # changed for finetuning with different learning rate
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and https://github.com/DrSleep/tensorflow-deeplab-resnet/issues/83biases.
            self.op_init = tf.global_variables_initializer()
            
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=2)
        
        self.graph.finalize()
    
    def inference(self, data, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        self.fc_data = self._inference(data, dropout)
        #        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            logits = self.fc(self.fc_data, self.M[-1], relu=False)
#        logits = tf.Print(logits, [logits], message="Logits: ")
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            prediction = tf.argmax(logits, axis=1)
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                print("Shape of labels", labels.get_shape().as_list())
                print("Shape of logits", logits.get_shape().as_list())
                if labels.get_shape().as_list() == logits.get_shape().as_list():
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#                    cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="Cross entropy: ")
                else:
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(cross_entropy)
#                cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="Mean cross entropy: ")
#                print('Cross entropy: ', cross_entropy)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    # for the case where 2 losses from 2 data to combine
    def loss_cbn2(self, logits1, logits2, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.to_int64(labels)
                cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels)
                cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labels)
                cross_entropy = tf.reduce_mean(tf.add(cross_entropy1, cross_entropy2))
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9, finetune_fc=False):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.build_graph
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                if type(decay_steps) is list:
                    learning_rate = tf.train.piecewise_constant(global_step, decay_steps, learning_rate, name= 'LearningRate')
                else:
                    learning_rate = tf.train.exponential_decay(
                            learning_rate, global_step, decay_steps, decay_rate, staircase=True)    
            
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            
            if finetune_fc == True:
                fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')
                logits_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'logits')
                train_vars = []
                train_vars.append(fc_vars)
                train_vars.append(logits_vars)
                print(train_vars)
            else:
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print(train_vars)

            grads = optimizer.compute_gradients(loss, var_list=train_vars) 
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        return os.path.join(self.dir_name, folder)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var
    def _weight_variable_fc_liucq(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.01)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable_fc_liucq(self, shape, regularization=True):
        initial = tf.zeros_initializer
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Fully connected

# Convolutional 

def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.

    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis


    
class cgcnn_liucq(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Sparse Matrix. Size (7*Levels) x M, Levels = 7.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
#        dir_name: Name for directories (summaries and model parameters).
        dir_name: Name for save directory (including summaries and model parameters).
    """
    def __init__(self, LA, LP, C, F, K, p, M, datetype='both', _inference='_inference_both_different_para', filter='chebyLiucqLR', brelu='b1relu', pool='avg_poolLiucqLR', finetune=False, finetune_fc=False,
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name='', random_batch_sampling_train=False, decay_factor=0.1):
#        super().__init__() # version Python 3
        super(cgcnn_liucq, self).__init__() # version Python 2
        
        print('Size LA: ', len(LA))
        print('Size LP: ', len(LP))
        print('Size F: ', len(F))
        print('Size K: ', len(K))
        print('Size p: ', len(p))
        
        # Verify the consistency w.r.t. the number of layers.
#        assert len(L) >= len(F) == len(K) == len(p)
        assert len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        assert len(LP) >= len(F)  # Enough coarsening levels for pool sizes.
                
        # Keep the useful Laplacians only. May be zero.
        self.LA=LA
        self.LP=LP
        self.datatype=datetype
        M_0 = LA[0][0].shape[0]
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, self.LA[i][0].shape[0], F[i], p[i], self.LA[i][0].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, self.LA[i][0].shape[0], F[i], self.LA[i][0].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else self.LA[-1][0].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.F, self.K, self.p, self.M = F, K, p, M
        self.finetune = finetune
        self.finetune_fc = finetune_fc
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self._inference = getattr(self, _inference)
        self.random_batch_sampling_train = random_batch_sampling_train

        self.decay_rate, self.momentum = decay_rate, momentum
        self.decay_factor = decay_factor
        
        if type(decay_steps) is list == True:
            self.decay_steps = decay_steps
        else:
            self.decay_steps = decay_steps
        
        print ('decay_steps = ', self.decay_steps)
        print ('dir_name = ', self.dir_name)
        
        #if filter is 'chebyLiucqLR' or filter is 'chebyLiucqLR2':
        if self.datatype is 'both':
            M_0 = M_0*2
        # Build the computational graph.
        self.build_graph(M_0, C)
        
    def filter_in_fourier(self, x, L, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout

    def fourier(self, x, L, Fout, K):
        assert K == L.shape[0]  # artificial but useful to compute number of parameters
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        _, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, Fout, Fin], regularization=False)
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def spline(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # M x M
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        #B = bspline_basis(K, len(lamb), degree=3)  # M x K
        B = tf.constant(B, dtype=tf.float32)
        # Weights
        W = self._weight_variable([K, Fout*Fin], regularization=False)
        W = tf.matmul(B, W)  # M x Fout*Fin
        W = tf.reshape(W, [M, Fout, Fin])
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    def chebyshev2(self, x, L, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.
        
        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(L, x, K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout
    def chebyLiucq(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N
        x = tf.expand_dims(x1, 0)  # 1 x M x Fin*N
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N
            x = concat(x, x1) #(i+1) x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyLiucqLR(self, x, L, Fout, K):
        N, M2, Fin = x.get_shape()
        N, M2, Fin, M, N2 = int(N), int(M2), int(Fin), int(M2/2), int(N*2)
        x0 = tf.reshape(x, [N, M, 2, Fin])
        x0 = tf.transpose(x0, perm=[1, 3, 0, 2])  # M x Fin x N x 2
        x0 = tf.reshape(x0, [M, Fin*N2])     # M x Fin*N2
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N2
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N2       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N2
        x = tf.expand_dims(x1, 0)  # 1 x M x Fin*N2
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N2
            x = concat(x, x1) #(i+1) x M x Fin*N2
        x = tf.reshape(x, [K, M, Fin, N2])  # K x M x Fin x N2
        x = tf.transpose(x, perm=[3,1,2,0])  # N2 x M x Fin x K
        x = tf.reshape(x, [N2*M, Fin*K])  # N2*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N2*M x Fout
        x = tf.reshape(x, [N, 2, M, Fout])
        x = tf.transpose(x, perm=[0,2,1,3])
        return tf.reshape(x, [N, M2, Fout])  # N x M x Fout
    def chebyLiucqLR_sub(self, x, L, Fout, K, W):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N
        x = tf.expand_dims(x1, 0)  # 1 x M x Fin*N
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M x Fin*N
            x = concat(x, x1) #(i+1) x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        #W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout
    def chebyLiucqLR_different_para(self, x, L, Fout, K):
        N, M2, Fin = x.get_shape()
        N, M2, Fin, M = int(N), int(M2), int(Fin), int(int(M2)/2)
        xList = tf.split(x,[M,M], axis=1)
        W = self._weight_variable([Fin*K*2, Fout], regularization=False)
        wList = tf.split(W,[Fin*K,Fin*K], axis=0)
        x  = self.chebyLiucqLR_sub(xList[0], L, Fout, K, wList[0])
        x1 = self.chebyLiucqLR_sub(xList[1], L, Fout, K, wList[1])
        return tf.concat([x,x1], axis=1)
    def chebyLiucqLR_same_para(self, x, L, Fout, K):
        N, M2, Fin = x.get_shape()
        N, M2, Fin, M = int(N), int(M2), int(Fin), int(int(M2)/2)
        xList = tf.split(x,[M,M], axis=1)
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x  = self.chebyLiucqLR_sub(xList[0], L, Fout, K, W)
        x1 = self.chebyLiucqLR_sub(xList[1], L, Fout, K, W)
        return tf.concat([x,x1], axis=1)
    def chebyLiucqLR2_One(self, x, Fout):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        W = self._weight_variable([Fin,Fout], regularization=False)
        x = tf.reshape(x, [N*M, Fin])  # N*M x Fin
        x = tf.matmul(x, W)  #  N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x
    def avg_poolLiucq(self, x, L, K):
        """Average pooling of size p. Should be a power of 2."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
        x = tf.expand_dims(x1, 0)  # 1 x M1 x Fin*N
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
            x = concat(x, x1) #(i+1) x M1 x Fin*N
        P, M1, F = x.get_shape()
        x = tf.reshape(x, [P, M1, Fin, N])  # K x M1 x Fin x N
        x = tf.transpose(x, perm=[3,0,1,2])  # N x K x M1 x Fin
        x = tf.nn.avg_pool(x, ksize=[1,K,1,1], strides=[1,K,1,1], padding='SAME')
        return tf.squeeze(x, [1])  # N x M1 x F
    def avg_poolLiucqLR(self, x, L, K):
        """Average pooling of size p. Should be a power of 2."""
        N, M2, Fin = x.get_shape()
        N, M2, Fin, M, N2 = int(N), int(M2), int(Fin), int(M2/2), int(N*2)
        x0 = tf.reshape(x, [N, M, 2, Fin])
        x0 = tf.transpose(x0, perm=[1, 3, 0, 2])  # M x Fin x N x 2
        x0 = tf.reshape(x0, [M, Fin*N2])     # M x Fin*N2
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N2
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N2       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N2
        x = tf.expand_dims(x1, 0)  # 1 x M1 x Fin*N2
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N2
            x = concat(x, x1) #(i+1) x M1 x Fin*N2
        P, M1, F = x.get_shape()
        x = tf.reshape(x, [P, M1, Fin, N2])  # K x M1 x Fin x N2
        x = tf.transpose(x, perm=[3,0,1,2])  # N2 x K x M1 x Fin
        x = tf.nn.avg_pool(x, ksize=[1,K,1,1], strides=[1,K,1,1], padding='SAME')
        x = tf.squeeze(x, [1])  # N2 x M1 x F
        x = tf.reshape(x, [N, 2, M1, Fin])
        x = tf.transpose(x, perm=[0,2,1,3])
        return tf.reshape(x, [N, M1*2, Fin])  # N x M1*2 x Fin
    def avg_poolLiucqLR_sub(self, x, L, K):
        """Average pooling of size p. Should be a power of 2."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
        x = tf.expand_dims(x1, 0)  # 1 x M1 x Fin*N
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
            x = concat(x, x1) #(i+1) x M1 x Fin*N
        P, M1, F = x.get_shape()
        x = tf.reshape(x, [P, M1, Fin, N])  # K x M1 x Fin x N
        x = tf.transpose(x, perm=[3,0,1,2])  # N x K x M1 x Fin
        x = tf.nn.avg_pool(x, ksize=[1,K,1,1], strides=[1,K,1,1], padding='SAME')
        return tf.squeeze(x, [1])  # N x M1 x F
    def avg_poolLiucqLR2(self, x, L, K):
        N, M2, Fin = x.get_shape()
        M = int(int(M2)/2)
        xList = tf.split(x,[M,M], axis=1)
        x = self.avg_poolLiucqLR_sub(xList[0], L, K)
        x1 = self.avg_poolLiucqLR_sub(xList[1], L, K)
        return tf.concat([x,x1], axis=1)
    def max_poolLiucq(self, x, L, K):
        """Max pooling of size p. Should be a power of 2."""
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N       
        # Transform
        indices = np.column_stack((L[0].row, L[0].col))
        LL = tf.SparseTensor(indices, L[0].data, L[0].shape)
        LL = tf.sparse_reorder(LL)
        x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
        x = tf.expand_dims(x1, 0)  # 1 x M1 x Fin*N
        for i in range(1,K):
            indices = np.column_stack((L[i].row, L[i].col))
            LL = tf.SparseTensor(indices, L[i].data, L[i].shape)
            LL = tf.sparse_reorder(LL)
            x1 = tf.sparse_tensor_dense_matmul(LL, x0) # M1 x Fin*N
            x = concat(x, x1) #(i+1) x M1 x Fin*N
        P, M1, F = x.get_shape()
        x = tf.reshape(x, [P, M1, Fin, N])  # K x M1 x Fin x N
        x = tf.transpose(x, perm=[3,0,1,2])  # N x K x M1 x Fin
        x = tf.nn.max_pool(x, ksize=[1,K,1,1], strides=[1,K,1,1], padding='SAME')
        return tf.squeeze(x, [1])  # N x M1 x F

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def fc_liucq(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable_fc_liucq([int(Min), Mout], regularization=True)
        b = self._bias_variable_fc_liucq([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x
    
    
#params['filter']         = 'chebyLiucq'
#params['brelu']          = 'b1relu'
#params['pool']           = 'avg_poolLiucq'
#params['_inference']         = '_inference_single(self, x, dropout)'        
    def _inference_single(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(1):#B3
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        for i in range(1, len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.LA[i], self.F[i], 7)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
#        # Logits linear layer, i.e. softmax without normalization.
#        with tf.variable_scope('logits'):
#            x = self.fc(x, self.M[-1], relu=False)
        return x    
#params['filter']         = 'chebyLiucqLR_same_para'
#params['brelu']          = 'b1relu'
#params['pool']           = 'avg_poolLiucqLR2'
#params['_inference']         = '_inference_both_same_para' 
    def _inference_both_same_para(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(1):#B3
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        for i in range(1,len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    #x = self.chebyLiucqLR(x, self.LA[i], self.F[i], 7)
                    x = self.filter(x, self.LA[i], self.F[i], 7)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
                    #x = self.avg_poolLiucqLR(x, self.LP[i], 7)
                    #x = self.avg_poolLiucqLR2(x, self.LP[i], 7)
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
#        # Logits linear layer, i.e. softmax without normalization.
#        with tf.variable_scope('logits'):
#            x = self.fc(x, self.M[-1], relu=False)
        return x
#params['filter']         = 'chebyLiucqLR_same_para'
#params['brelu']          = 'b1relu'
#params['pool']           = 'avg_poolLiucqLR2'
#params['_inference']         = '_inference_both_same_para_multi' 
    def _inference_both_same_para_multi(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(1):#B3
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        for i in range(1,len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.LA[i], self.F[i], 7)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
            with tf.variable_scope('conv{}'.format(i+1+len(self.p))):        
                with tf.name_scope('filter'):
                    x = self.filter(x, self.LA[i], self.F[i], 7)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        return x   
#_inference_both_different_para(self, x, dropout):
#params['filter']         = 'chebyLiucq'
#params['brelu']          = 'b1relu'
#params['pool']           = 'avg_poolLiucq'
#params['_inference']         = '_inference_both_different_para(self, x, dropout)        
    def _inference_both_different_para2(self, x, dropout):
        # Graph convolutional layers.
        N, M2 = x.get_shape()
        M = int(M2/2)
        xList = tf.split(x,[M,M], axis=1)
        for ik in range(2): #B2
            x = tf.expand_dims(xList[ik], 2)  # N x M x F=1
            i = 0
            with tf.variable_scope('conv{}'.format(ik*2*len(self.p)+i+1)):
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
            for i in range(1, len(self.p)):#B3
                with tf.variable_scope('conv{}'.format(ik*2*len(self.p)+i+1)):
                    with tf.name_scope('filter'):
                        #x = self.chebyLiucq(x, self.LA[i], self.F[i], 7)
                        x = self.filter(x, self.LA[i], self.F[i], 7)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.LP[i], 7)
                    #    x = self.avg_poolLiucq(x, self.LP[i], 7)
                    #    x = self.max_poolLiucq(x, self.LP[i], 7)
                        
            #E3
        # Fully connected hidden layers.
            N, M, F = x.get_shape()
            x = tf.reshape(x, [int(N), int(M*F)])  # N x M
            for i,M in enumerate(self.M[:-1]):
                with tf.variable_scope('fc{}'.format(ik*2+i+1)):
                    x = self.fc(x, M)
                    x = tf.nn.dropout(x, dropout)
            if ik == 0:
                x_ = x
            else:
                x_ = tf.concat([x_,x], axis=1)
        #E2
#        # Logits linear layer, i.e. softmax without normalization.
#        with tf.variable_scope('logits'):
#            x = self.fc(x_, self.M[-1], relu=False)
        return x_
    
#params['filter']         = 'chebyLiucqLR_different_para'
#params['brelu']          = 'b1relu'
#params['pool']           = 'avg_poolLiucqLR2'
#params['_inference']         = '_inference_both_different_para' 
    def _inference_both_different_para(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(1):#B3
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
        for i in range(1,len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    #x = self.chebyLiucqLR(x, self.LA[i], self.F[i], 7)
                    x = self.filter(x, self.LA[i], self.F[i], 7)
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.LP[i], 7)
                    #x = self.avg_poolLiucqLR(x, self.LP[i], 7)
                    #x = self.avg_poolLiucqLR2(x, self.LP[i], 7)
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
#        # Logits linear layer, i.e. softmax without normalization.
#        with tf.variable_scope('logits'):
#            x = self.fc(x, self.M[-1], relu=False)
        return x


