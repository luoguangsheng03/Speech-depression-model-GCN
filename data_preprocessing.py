# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:33:13 2023

@author: Jun Ye
"""
import pandas as pd
import torch

# In[1] 
'''
This section is to load the features from file ComParE.h5

Preprocessing Operation:
    1. Select/Pooling Channels of features
    2. Filt the single values (whose absolute values are over 100)
'''
from load_ComParE import load_ComP
attn_array, feature_array = load_ComP()
from utils import get_features, filt_singlevalue
features = get_features(attn_array, feature_array, threshold = 0.5)
features = filt_singlevalue(features)
# In[2] 
'''
This section is to load the features from file phonation_static.h5

Preprocessing Operation:
    1. replace nan with mean of columns/ get rid of nan if they are over half
    2. Filt the single values (whose absolute values are over 100)
'''
from load_phonation_static import load_phon
features_phon = load_phon()
# In[3]
'''
This section is to load labels file and calculate adjacency matrix

Preprocessing Operation:
    1. Extract labels of patient with depression and without depression
    2. Delete Subjects with Outliers, Index = 754 
    3. Calculate adjacency matrix with behaviors informations
'''
data_label = pd.read_csv('label.csv')
from utils import create_adjacency_matrix

label_matrix = data_label['label'].drop(index = 754).to_numpy()
data_label = data_label.drop(columns = ['label','patient_id', 'duration'])
data_label = data_label.drop(index = 754)
label_array = data_label.to_numpy().T
adj = create_adjacency_matrix(label_array)  

# In[4]
'''
Normalize adj and features
Note. The method getting adjacency matrix contains I matrix and symmetrical
'''
from utils import normalize
adj = normalize(adj)
#features = np.append(features, feature_array_phonation, axis =1)
features = normalize(features)
#features_phon = normalize(feature_array_phonation)

# In[5]
'''
DataSet Split
# Split dataset into training set, validation set and test set by
# extracting data samples in a round-robin fashion.
'''
idx_train = range(0, 1120, 2)
idx_val = range(1, 1120, 2)
idx_test = range(2, 1120, 2)

'''
Turn data into tensor format
'''
features = torch.FloatTensor(features)
features_phon = torch.FloatTensor(features_phon)
labels = torch.FloatTensor(label_matrix)
adj = torch.FloatTensor(adj)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
