'''
NTHU EE Machine Learning HW2
Author: 
Student ID: 
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse

def get_MLR_Weights(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

def get_BLR_Weights(X, Y, lamb_da):
    return np.linalg.inv(lamb_da*np.eye(X.shape[1]) + X.T @ X) @ X.T @ Y

def construct_feature_vec(features, O1, O2):
    # features為 n*3的float array, n為資料個數
    feature_vec = np.ones((features.shape[0], O1*O2+2))
    X1 = features[:,0] # (n,) float array
    X2 = features[:,1] # (n,) float array
    X3 = features[:,2] # (n,) float array

    X1_max = np.max(X1)
    X2_max = np.max(X2)
    X1_min = np.min(X1)
    X2_min = np.min(X2)

    s1 = (X1_max - X1_min)/(O1 - 1) #scalar
    s2 = (X2_max - X2_min)/(O2 - 1) #scalar

    for i in range(1, O1+1, 1):
        for j in range(1, O2+1, 1):
            u_i = s1 * (i - 1) + X1_min #scalar
            u_j = s2 * (j - 1) + X2_min #scalar

            k = O2 * (i - 1) + j

            phi_of_X = np.exp( -((X1 - u_i)**2) / (2*s1**2) - ((X2 - u_j)**2) / (2*s2**2) ) # (n,)
            feature_vec[:, k-1] = phi_of_X # 一直行
    
    feature_vec[:, -2] = X3
    return feature_vec

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    # train_data 有feature + data (300*4) float array
    # test_data_feature 只有feature (100*3) float array
    
    data_train_feature = train_data[:, :3] # (n*3) float array
    data_train_label = train_data[:, -1] # (n,) float array

    training_feature = construct_feature_vec(data_train_feature, O1, O2) # training_feature:(n* (O1*O2+2))
    
    Model_Weights = get_BLR_Weights(training_feature, data_train_label, 0.01)

    #start to predict
    predict_list = []
    testing_feature = construct_feature_vec(test_data_feature, O1, O2) # testing_feature:(n* (O1*O2+2))
    for i in range(0, test_data_feature.shape[0], 1):
        predict_list.append(testing_feature[i, :] @ Model_Weights)

    return np.array(predict_list)

# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    # train_data 有feature + data (300*4) float array
    # test_data_feature 只有feature (100*3) float array
    
    data_train_feature = train_data[:, :3] # (n*3) float array
    data_train_label = train_data[:, -1] # (n,) float array

    training_feature = construct_feature_vec(data_train_feature, O1, O2) # training_feature:(n* (O1*O2+2))
    
    Model_Weights = get_MLR_Weights(training_feature, data_train_label)

    #start to predict
    predict_list = []
    testing_feature = construct_feature_vec(test_data_feature, O1, O2) # testing_feature:(n* (O1*O2+2))
    for i in range(0, test_data_feature.shape[0], 1):
        predict_list.append(testing_feature[i, :] @ Model_Weights)

    return np.array(predict_list)


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=2)
    parser.add_argument('-O2', '--O_2', type=int, default=2)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy() # 有feature + data (100*4) float array
    data_test_feature = data_test[:, :3] # (100*3) float array
    data_test_label = data_test[:, 3] # (100,) float array

    data_train_feature = data_train[:, :3] # (n*3) float array
    data_train_label = data_train[:, 3] # (n,) float array
    

    predict_MLR_for_testing = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_BLR_for_testing = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    predict_MLR_for_training = MLR(data_train, data_train_feature, O1=O_1, O2=O_2)
    predict_BLR_for_training = BLR(data_train, data_train_feature, O1=O_1, O2=O_2)

    print('Testing data set : MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR_for_testing, data_test_label), e2=CalMSE(predict_MLR_for_testing, data_test_label)))

    print('Training data set : MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR_for_training, data_train_label), e2=CalMSE(predict_MLR_for_training, data_train_label)))


if __name__ == '__main__':
    main()
    