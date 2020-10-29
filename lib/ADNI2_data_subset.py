#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:31:09 2019

@author: chaoqiang

Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

import numpy as np
import h5py
from sklearn.utils import shuffle

def adni_generate_unique_SubjId(SubjID):
    num = 0
    name=[]
    SubjIDIndex = np.array(range(len(SubjID)))
    SubjID_unique = []
    for k in range(len(SubjID)):
        if SubjID[k] != name:
            num = num+1
            name = SubjID[k]
            SubjIDIndex[k] = num-1
            SubjID_unique.append(name)
        else:
            SubjIDIndex[k] = num-1
    return SubjIDIndex
def adni_generate_unique_SubjIdIndex(SubjID):
    num = 0
    name=-1
    SubjIDIndex = np.array(range(SubjID.shape[0]))
    SubjID_unique = []
    for k in range(len(SubjID)):
        if SubjID[k] != name:
            num = num+1
            name = SubjID[k]
            SubjIDIndex[k] = num-1
            SubjID_unique.append(name)
        else:
            SubjIDIndex[k] = num-1
    return SubjIDIndex            

def adni_get_subset(filename, label1, label2, readdata):
    f = h5py.File(filename, 'r')
    print(list(f.keys()))
    Label = np.transpose(np.array(f.get('Label')).astype(int))[0]
    Age   = np.array(f.get('Age'))
    Gender= np.array(f.get('Gender'))

    if readdata > 0:
        lthick = np.array(f.get('lthick_regular'))
        rthick = np.array(f.get('rthick_regular'))
        thickdata = np.concatenate([lthick, rthick], axis=1)
        del lthick, rthick
    else:
        thickdata = []
        
    SubjID=[]
    for column in f['SubjID']:
        row_data = ''.join(map(chr, f[column[0]][:]))
        SubjID.append(row_data)
    
    select_index = [i for i,aa in enumerate(Label) if aa == label1 or aa == label2]
    base_index   = [i for i,aa in enumerate(Label) if aa < label1]

    SubjIDIndex_in_whole = adni_generate_unique_SubjId(SubjID)
    SubjIDIndex_sub_in_whole = SubjIDIndex_in_whole[select_index]
    SubjIDIndex_sub_index = adni_generate_unique_SubjIdIndex(SubjIDIndex_sub_in_whole)
    Label_s = Label[select_index]
    Age_s   = Age[select_index]
    Gender_s= Gender[select_index]
    if readdata > 0:
        thick_s = thickdata[select_index]
    if len(base_index)>0:
        if readdata >0:
            thick_base = thickdata[base_index]
        else:
            thick_base = []
        Age_base = Age[base_index]
        Gender_base = Gender[base_index]
    else:
        thick_base  = []
        Age_base    = []
        Gender_base = []
    Label_s[Label_s==label1] = 0
    Label_s[Label_s==label2] = 1
    return Label_s, SubjIDIndex_sub_in_whole, SubjIDIndex_sub_index, \
        thick_s, Age_s, Gender_s, thick_base, Age_base, Gender_base

def adni_set_class_define(Label, SubjidIndex):
    N = np.max(SubjidIndex)+1
    LabelSum = np.zeros(N, dtype=float)
    LabelNum = np.zeros(N, dtype=float)
    for i in range(len(Label)):
        LabelSum[SubjidIndex[i]] += Label[i]
        LabelNum[SubjidIndex[i]] += 1
    S = LabelSum/LabelNum
    C0 = np.array(np.where(S<0.5))[0]
    C1 = np.array(np.where(S>=0.5))[0]

    return C0, C1
def adni_tenfold(Label, SubjIDIndex, C0, C1, state, idx):
    if state < 0:
        C0 = shuffle(C0)
        C1 = shuffle(C1)
    else:
        C0 = shuffle(C0, random_state=state)
        C1 = shuffle(C1, random_state=state)
    NC0 = len(C0)
    PC0 = np.floor(NC0/10+0.8).astype(int)
    NC1 = len(C1)
    PC1 = np.floor(NC1/10+0.8).astype(int)
    if idx == 9:
        TestIdx  = np.concatenate((C0[PC0*9:NC0], C1[PC1*9:NC1]))
        TrainIdx = np.concatenate((C0[0:PC0*9], C1[0:PC1*9])) 
    else:
        TestIdx  = np.concatenate((C0[PC0*idx:PC0*(idx+1)], C1[PC1*idx:PC1*(idx+1)]))
        if idx == 0:
            TrainIdx = np.concatenate((C0[PC0:NC0], C1[PC1:NC1]))
        else:
            TrainIdx = np.concatenate((C0[0:PC0*(idx-1)], C0[PC0*(idx+1):NC0], 
                                          C1[0:PC1*(idx-1)], C1[PC1*(idx+1):NC1]))
    TrainSet = [i for i,aa in enumerate(SubjIDIndex) if aa in TrainIdx]
    TestSet = [i for i,aa in enumerate(SubjIDIndex) if aa in TestIdx]
    return TrainIdx, TestIdx, TrainSet, TestSet
 
def adni_get_subdata_only(Label, thick, TrainSet, TestSet):
    Label_test = Label[TestSet]
    Label_train = Label[TrainSet]
    thick_test = thick[TestSet]
    thick_train = thick[TrainSet]

    return thick_train, thick_test, Label_train, Label_test

