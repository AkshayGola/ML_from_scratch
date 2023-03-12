#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
import math


# In[2]:


# function to calculate gain ratio
def gain_ratio(Y, feature, level_entropy):
    
    # parameter 'feature' is the numpy array of feature, whose gain_ratio is being calculated to decide whether to split upon it or not
    # parameter 'Y' is output
    # parameter 'level_entropy'  is entropy of the current level, which has called function 'gain_ratio' to decide which feature to split on
    # variable 's_i' stores the value of split info 
    
    s_i = split_info(feature)
    
    if s_i == 0: 
        return(math.inf)
    
    # variable N stores total number of sample points in numpy array of the feature
    N = len(feature)
    
    # variable 'info_req' stores the value of weighed entropy in case decision tree splits on the feature
    info_req = 0
    
    # variable 'feature_classes' is numpy array of all possible values in feature  
    feature_classes = np.array(list(set(feature)))
    for i in range(len(feature_classes)):
        
        # variable 'frac' stores the values of weight of a particular class in feature
        frac = len(feature[feature == feature_classes[i]])/N
        
        # variable 'frac_entropy' stores the entropy/information requires of a paricular class in the feature
        frac_entropy = entropy(Y[feature == feature_classes[i]])

        info_req = info_req + frac * frac_entropy
    

    gain_ratio = (level_entropy - info_req)/s_i
    
    return(gain_ratio)


# In[3]:


# function to calculate split information
def split_info(feature):
    
    # 'feature' is numpy array of a selected feature, whose split information is being claculated
    
    # variable 'N' stores total number sample points in feature
    N = len(feature)
    s_i = 0
    feature_classes = np.array(list(set(feature)))
    for i in range(len(feature_classes)):
        frac = len(feature[feature == feature_classes[i]])/N
        s_i = s_i - frac * math.log2(frac) 
        
    return(s_i)


# In[4]:


# function to calculate entropy
def entropy(Y):
    
    # 'Y' is numpy array of output
    
    # variable e stores value of entropy
    e = 0
    
    # varaible 'n_Y' is total number of sample points in output 'Y'
    n_Y = len(Y)
    
    # classes is numpy array which stores total number of possible values of output 'Y' 
    classes = np.array(list(set(Y)))
    for i in range(len(classes)):

        # variable 'n_class' stores total number of sample points belonging to a particular class
        n_class = len(Y[Y == classes[i]])

        # variable 'prob' stores probabality of occurance of a particular class in output 'Y'
        prob = (n_class/n_Y)

        e = e - prob * math.log2(prob) 

    return e


# In[5]:


# recursive function to implement decision tree
def decision_tree(level,X,Y,Y_classes, features):
    
    # variable 'level' is an integer value of the level at which decision tree is further splitting
    # variable 'X' is dictionary which has as feature names as keys, and numpy array of corresponding feature as values
    # variable 'Y_classes' includes all possible values of output 'Y'
    # variable 'Y' is numpy array of ouput
    # variable 'feature' is numpy array of valid features, yet to be split upon
    
    print('Level', level)
    
    for i in range(len(Y_classes)):
        print('Count of', Y_classes[i],'=',len(Y[Y == Y_classes[i]]))
    
    level_entropy = entropy(Y)
    print('Current Entropy  is =',level_entropy)
    
    if level_entropy == 0:
        print('Reached leaf Node')
        return
    elif len(features) == 0:
        return
    else:
        
        # variable 'max_gain_ratio' stores the value of maximum gain ratio amongst all features
        max_gain_ratio = -1
        
        # variable 'feature_split' stores the feature, whose gain ratio is maximum on being split upon 
        feature_split = ''
        
        
        # loop for iterating over all valid features to find maximum gain ratio and corresponding feature
        for i in range(len(features)):
            feature_gain_ratio = gain_ratio(Y, X[features[i]], level_entropy)
            if max_gain_ratio == -1:
                max_gain_ratio = feature_gain_ratio
                feature_split = features[i]
            elif feature_gain_ratio > max_gain_ratio:
                max_gain_ratio = feature_gain_ratio
                feature_split = features[i]
            else:
                pass
        
        print('Splitting on feature', feature_split ,'with gain ratio', max_gain_ratio)
        

        # variable 'feature_values' includes all possible values of the selected feature to split upon 
        feature_values = np.array(list(set(X[feature_split])))
        
        # loop to iterate over all classes of selected feature to split upon
        for i in range(len(feature_values)):
            
            # variable 'X_split' is a dictionary which includes feature names as keys and split features as its values
            X_split = {}
            
            # loop to iterate over all features to split the features accarding to selected feature to split upon 
            for j in range(len(features)):
                X_split[ features[j] ] = X[features[j]][X[feature_split] == feature_values[i]]
                
            # variable 'Y_split' is the numpy array, which only includes output only corresponding to particular class of selected feature to split upon   
            Y_split = Y[X[feature_split] == feature_values[i]]
            
            # variable 'feature_split' is the numpy array of valid features (it does not include already split upon features), passed to the next level
            features_split = np.delete(features, np.where(feature_split))
            
            print('')
            decision_tree(level + 1, X_split ,Y_split ,Y_classes , features_split)


# In[6]:


# Below function converts continuous data to categorical data
def cont_to_categor(cont_data):
    cont_data = np.array(cont_data)
    
    cat_data = np.zeros(cont_data.shape)
    
    for i in range(cont_data.shape[0]):
        for j in range(cont_data.shape[1]):

            if cont_data[i,j] < np.mean(cont_data[:,j]):
                cat_data[i,j] = 0
            else:
                cat_data[i,j] = 1
    return cat_data


# In[7]:


iris = datasets.load_iris()

features = iris['feature_names']

cat_data = cont_to_categor(iris['data'])
X = {}

for i in range(len(iris['feature_names'])):
    X[iris['feature_names'][i]] = np.array(cat_data[:,i])
      
Y = iris['target']

Y_classes = np.array(list(set(Y)))


# calling decision tree function
decision_tree(0,X,Y,Y_classes, features)


# In[8]:


# OR dataset

df = pd.read_csv('or.csv')
df.head()

df_X = df[df.columns[:-1]]

features = list(df_X.columns)
df_X = np.array(df_X)
df_Y = np.array(df['X1 OR X2'])

X = {}

for i in range(len(features)):
    X[features[i]] = np.array(df_X[:,i])
    
Y = df_Y

Y_classes = np.array(list(set(Y)))
decision_tree(0,X,Y,Y_classes, features)


# In[ ]:




