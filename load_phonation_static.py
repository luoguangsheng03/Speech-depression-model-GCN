# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:30:18 2023
This file is used to load phonation_static features
@author: Jun Ye
"""
import numpy as np

import h5py

file = h5py.File('phonation_static.h5', 'r')

from utils import replace_nan, get_dataset_num
def load_features_phonation(file):
    '''
    Load Phonation_static file's features

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
        h5 file openned by h5py

    Returns
    -------
    feature_array : TYPE 2-d array M*N
        DESCRIPTION. 
        M is the patient idx
        N is the features length

    '''
    dataset_list =  list( file.values() ) 
    data_subject_id = {} 

    count, max_axis1, max_axis2 = get_dataset_num(file) 
    #feature_array = np.zeros( (count,max_axis1,  max_axis2) )  
    feature_array = np.zeros( (count,  max_axis2) )    
    for i, dataset in enumerate(dataset_list): 
         data_feature = dataset_list[i][()]
         new_features, deleted_rows = replace_nan(data_feature)
         #feature_array[i, : len(data_feature)] = data_feature
         feature_array[i] = new_features.mean(axis = 0)
         data_subject_id[i] = dataset_list[i].name[6:] 
         
    feature_array = np.delete(feature_array, [537, 754, 1107], axis = 0)
    return feature_array
     #data_subject_id[i] = dataset_list[i].name[1:]
feature_array_phonation = load_features_phonation(file)


def load_phon():
    global feature_array_phonation
    return feature_array_phonation
file.close()