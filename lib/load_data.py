"""
Copyright (C) 2019,
CFA Lab and Dept. of Biomedical Engineering, National University of Singapore. 
All rights reserved.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import xrange

import h5py
import scipy.io as sio
from scipy.sparse import coo_matrix


def load_regular_neighbor_sparse_matrix_fromfile(filename):#B1
    data = sio.loadmat(filename)
    aRegMatrixData=data['sparseMatrixs']
    nLevel = aRegMatrixData.shape[0] #shape is (nLevel,7,3)
    pMatrixs = []
    
    for kk in xrange(nLevel):#B2
        k = nLevel-1-kk
        kshape=aRegMatrixData[k,0][0].shape[0]
        pp=[]
        for jj in xrange(7):#B3 
            row = np.asarray(aRegMatrixData[k,jj][0]-1, dtype=np.int64)[:,0]
            col = np.asarray(aRegMatrixData[k,jj][1]-1, dtype=np.int64)[:,0]
            values = np.asarray(aRegMatrixData[k,jj][2], dtype=np.float32)[:,0]
            xx = coo_matrix((values,(row,col)), shape=(kshape,kshape))
            pp.append(xx)
        #E3
        pMatrixs.append(pp)
    #E2

    pPoolMatrixs = []
    for k in xrange(nLevel-1):#B2
        kshape1 = pMatrixs[k][0].shape[0]
        kshape0 = pMatrixs[k+1][0].shape[0]
        pp=[]
        for jj in xrange(7):#B3
            xx = pMatrixs[k][jj]
            M = len(pMatrixs[k+1][jj].row)
            row = xx.row[0:M]
            col = xx.col[0:M]
            values = xx.data[0:M]
            xx = coo_matrix((values,(row,col)), shape=(kshape0,kshape1))
            pp.append(xx)
        #E3
        pPoolMatrixs.append(pp)
    #E2
    
    return pMatrixs, pPoolMatrixs
#E1


#  Properties:
#     Properties.Writable: false                                                                      
#         TransferMatrixs: [3-D         cell]       7X7X3 (MxN) matrixs                                                  
#              label_test: [266x2       double]                                                       
#             label_train: [1286x2      double]                                                       
#     lthick_regular_test: [163842x     double]                                                       
#    lthick_regular_train: [163842x     double]                                                       
#     rthick_regular_test: [163842x     double]                                                       
#    rthick_regular_train: [163842x     double]                                                       
#             subjID_test: [266x1       cell]                                                         
#            subjID_train: [1286x1      cell]  
def load_regular_thickness_data_fromfile(filename):#B1
    f = h5py.File(filename, 'r')
    label_test = np.array(f.get('label_test'))
    label_train = np.array(f.get('label_train'))
    lthick_regular_test = np.array(f.get('lthick_regular_test'))
    lthick_regular_train = np.array(f.get('lthick_regular_train'))
    rthick_regular_test = np.array(f.get('rthick_regular_test'))
    rthick_regular_train = np.array(f.get('rthick_regular_train'))
    return label_test, label_train, lthick_regular_test, lthick_regular_train, rthick_regular_test, rthick_regular_train
#E1
            