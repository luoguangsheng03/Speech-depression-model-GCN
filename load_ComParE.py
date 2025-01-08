# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:12:10 2023
This file is to load data of "ComParE" features
@author: Jun Ye
"""
import numpy as np
import pandas as pd
import h5py

# open h5 file
file = h5py.File('ComParE.h5', 'r')



"""
This section is to Get attn data,

data_dict: attn data
data_subject_id: the id of each subject

output: csv format of attn data
"""
def load_attn(file):
    '''
    Load attn of ComParE

    Parameters
    ----------
    file : TYPE h5 file opened by h5py
        DESCRIPTION.

    Returns
    -------
    attn_array : TYPE 2-d array of attn, M*N
        DESCRIPTION.
        M is patients index
        N is attn length of this patient

    '''
    dataset_list =  list( file['attn'].values() ) 
    data_dict = {}
    data_subject_id = {}    
       
    for i, dataset in enumerate(dataset_list): 
        data = dataset_list[i][()]
        data_dict[i] = data
        data_subject_id[i] = dataset_list[i].name[6:]  
    
    df_attn = pd.DataFrame.from_dict(data_dict,  orient='index') 
    attn_array = df_attn.to_numpy()
    #df_subject_id = pd.DataFrame.from_dict(data_subject_id,  orient='index')
    #df_attn.insert(0, 'subject_id', df_subject_id)
    #df_attn.to_csv('attn.csv', index=False)
    return attn_array

attn_array = load_attn(file)




def load_features(file):
    '''
    Load features of ComParE

    Parameters
    ----------
    file : TYPE 
        DESCRIPTION.
        h5 file openned by h5py

    Returns
    -------
    feature_array : TYPE 3-d ndarray, M*N*Z
        DESCRIPTION.
        M is patient index
        N is channels of features
        Z is lengths of features

    '''
    dataset_list =  list( file['feature'].values() ) 
    
    num = len(attn_array)
    num_features = len(attn_array[0])
    length_features =  len(dataset_list[0][()][0])  
    feature_array = np.zeros( (num, num_features, length_features) )  
        
    for i, dataset in enumerate(dataset_list): 
          data_feature = dataset_list[i][()]
          feature_array[i,  : len(data_feature)] = data_feature
    return feature_array
      #data_subject_id[i] = dataset_list[i].name[6:]
feature_array = load_features(file)



def load_ComP():
    '''
    Load ComparE FILE
    Returns
    -------
    attn_array : TYPE
        DESCRIPTION.
    feature_array : TYPE
        DESCRIPTION.

    '''
    global attn_array
    global feature_array
    return attn_array, feature_array    
file.close()