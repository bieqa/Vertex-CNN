"""
Created on Dec 6 2018

Perform geometric cnn for dementia classification using cortical thickness data from 
ADNI dataset. Using ten-fold to defind train and test data sets.
@author: Chaoqiang Liu
Revision: 1.0 

Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""
"""
Input files      : REG_NEIGHBOR_MAT = '../regular/regular_neighborhood_matrix_7.mat'
                   REG_thick_MAT   =  '../regular/adni_thickness_age_gender_regression_regular.mat'
Output directory : save_dir_group0 = '../corthick_age_cn_ad'
"""

import numpy as np
import pandas as pd

import os, time
import datetime

from lib import helper_func as hf
from lib import load_data as lda
from lib import ADNI2_data_subset as adni
from lib import models_gvcnn as models

os.environ["CUDA_VISIBLE_DEVICES"]="0" #GPU1   Test_new

# ~~~~~~~~~~~~~~~~~~~~~~~~ Data type and save folder ~~~~~~~~~~~~~~~~~~~~~~~~ #
dataset = 'adni'

groups = ['cn_ad', 'cn_emci', 'cn_lmci', 'lmci_ad', 'emci_lmci', 'emci_ad']
C0ids  = [0, 0, 0, 2, 1, 1]
C1ids  = [3, 1, 2, 3, 2, 3]

groupid = 0

randomstate = 0
foldIDS = 10

group     = groups[groupid]
C0ID      = C0ids[groupid]
C1ID      = C1ids[groupid]

params = dict()
params['num_epochs']     = 30
DATADIR = '../regular/'
#DATADIR = '/home/projects/11001444/bieliucq/regular/'

avg_pool = True
#avg_pool = False # maximum pooling
"""Hyperparameters"""

# Perform data augmentation using mixup
alpha = 0.  # without mixup
#alpha = 0.001  # with mixup


   
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Graph CNN parameters
test_ratio = 0.15
valid_ratio = 0.25       # portion fof validation data from the original training data
input_norm = True       # Do you want z-score input normalization?
random_batch_sampling_train = False
patient_ratio = .5


batch_size               = 64 

if alpha == 0.:
    params['batch_size'] = batch_size
else:
    params['batch_size'] = 2*batch_size
params['eval_frequency'] = 10

# Building blocks.

params['filter']         = 'chebyLiucqLR_same_para'
params['brelu']          = 'b1relu'
params['pool']           = 'avg_poolLiucqLR2'
params['_inference']         = '_inference_both_same_para'
#params['_inference']         = '_inference_both_same_para_multi'

datatype = 'both' #'left', 'right'
params['datetype']     = datatype

path = os.path.dirname(os.path.realpath(__file__))
save_dir_group0 = os.path.join(path, 'corthick_age_' + group)
szDatetime = datetime.datetime.now().strftime('%s')
save_dir_group  = os.path.join(save_dir_group0, 'state'+str(randomstate)+'_'+szDatetime)


# Number of classes.
C = 2
# Architecture.
params['p']              = [7, 7, 7, 7] # Pooling sizes.
params['F']              = [8, 16, 32, 64]  # Number of graph convolutional filters.
params['K']              = [7, 7, 7, 7]  # 
params['M']              = [512, C]  # Output dimensionality of fully connected layers. 

params['regularization'] = 5e-4
params['dropout']        = 1
params['decay_rate']     = 1#0.5
params['momentum']       = 0#0.9
params['decay_steps']    = 40
params['learning_rate']  = 1e-3



# ~~~~~~~~~~~~~~~~~~~ Training, validation and test data ~~~~~~~~~~~~~~~~~~~~ #
print('Loading data ...')

REG_NEIGHBOR_MAT = DATADIR + 'regular_neighborhood_matrix_7' + '.mat'
LA, LP = lda.load_regular_neighbor_sparse_matrix_fromfile(REG_NEIGHBOR_MAT)


REG_thick_MAT = DATADIR + 'adni_thickness_age_gender_regression_regular.mat'

Label, SubjID, SubjIDIndex, thick_select, Age, Gender, thick_base, Age_base, Gender_base =\
    adni.adni_get_subset(REG_thick_MAT, C0ID, C1ID, 1)
C0, C1 = adni.adni_set_class_define(Label, SubjIDIndex)

Labelfilename   = os.path.join(save_dir_group, dataset + '_' + group + '_label.csv')
Accuaryfilename = os.path.join(save_dir_group, dataset + '_' + group + '_performance.csv')
Resultfilename  = os.path.join(save_dir_group0, dataset + '_' + group + '_performance_tenfold.csv')

all_start_time = time.time()

for foldID in range(foldIDS):
    save_dir = os.path.join(save_dir_group, str(foldID))
    params['dir_name']       = save_dir
    
    TrainIdx, TestIdx, TrainSet, TestSet = adni.adni_tenfold(Label, SubjIDIndex, C0, C1, randomstate, foldID)
    input_train_data, data_test, input_train_target, input_test_target =  adni.adni_get_subdata_only(Label, thick_select, TrainSet, TestSet)

        
    input_test_target = np.transpose(np.vstack([np.transpose(1-input_test_target), np.transpose(input_test_target)]))
    input_train_target = np.transpose(np.vstack([np.transpose(1-input_train_target), np.transpose(input_train_target)]))

    print('Sphere shape for train data: ', input_train_data.shape)
    print('Sphere shape for test data: ', data_test.shape)

    if alpha > 0.:
        target_test = np.copy(input_test_target)
    else:
        target_test = np.copy(input_test_target[:,1])
    target_test = target_test.astype(np.uint8)
    if patient_ratio < 0:
        num_control = np.sum(input_train_target[:,0])
        num_patient = np.sum(input_train_target[:,1])
        patient_ratio = num_patient/float(num_patient+num_control)
# Create training and validation sets
    data_train, target_train, data_valid, target_valid = hf.train_valid_data(input_data=input_train_data, input_target=input_train_target, valid_ratio=valid_ratio)
    data_train = data_train.astype(np.float32)
    data_valid = data_valid.astype(np.float32)
    del input_train_data

    if alpha > 0.:
        target_train = np.copy(target_train)
        target_valid = np.copy(target_valid)
    else:
        target_train = np.copy(target_train[:,1])
        target_valid = np.copy(target_valid[:,1])
    target_train = target_train.astype(np.uint8)
    target_valid = target_valid.astype(np.uint8)

    print('Input label shape: ', target_train.shape)

    d       = data_train.shape[1]       # Dimensionality.
    n_train = data_train.shape[0]       # Number of train samples.

    params['decay_steps']    = (np.multiply(params['decay_steps'], n_train / params['batch_size'])).astype(int)
    
    c = np.unique(target_train)   # Number of feature communities (classes).
    
    print('Class imbalance: ', np.unique(target_train, return_counts=True)[1])
    print('Dimensionality: ', d)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#

# Training and validation 0.8
    if alpha > 0.:
        model = models.cgcnn_liucq(LA, LP, C, **params)
    else:
        model = models.cgcnn_liucq(LA, LP, 1, **params)
    val_loss_list, val_accuracy_list, val_fscore_list, val_sensitivity_list, val_specificity_list, val_precision_list, val_ppv_list, val_npv_list, val_gmean_list = model.fit(data_train, target_train, data_valid, target_valid, alpha, patient_ratio=patient_ratio)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Evaluate performance on test set
    prevalence = np.sum(target_test)*1.0 / target_test.shape[0]

    res, loss, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, predictions = model.evaluate(data_test, target_test)
    hf.save_to_csv(szDatetime, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, Accuaryfilename, dataset, mode='a+', header=(foldID==0))
    hf.print_classification_performance(dataset, prevalence, loss, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean)
    
    if alpha > 0.:
        d = {'Idx':TestSet, 'Actual':target_test[:,1], 'Predicted':predictions}
    else:
        d = {'Idx':TestSet, 'Actual':target_test, 'Predicted':predictions}
    df_label = pd.DataFrame(data=d, dtype=np.uint16)
    df_label.to_csv(os.path.join(save_dir, dataset + '_' + group + '_label.csv'))

    df_labelgroup = pd.DataFrame(data=d)
    df_labelgroup.to_csv(Labelfilename, mode='a+', header=(foldID==0))
    del model
    print('processed fold ID:', foldID)
V = pd.read_csv(Labelfilename)
labels = np.array(V.Actual)
predictions = np.array(V.Predicted)
prevalence = np.sum(labels)*1.0 / len(labels)
accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean = hf.top_k_error(labels, predictions, prevalence, 1)
hf.save_to_csv(szDatetime, accuracy, f1, sensitivity, specificity, precision, ppv, npv, gmean, Resultfilename, dataset, mode='a+', header=True)

all_time = float(time.time() - all_start_time)
format_str = 'whole time: %.3f, epoch time: %.3f'
print (format_str % (all_time, all_time/params['num_epochs']/foldIDS))

