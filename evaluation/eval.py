import pandas as pd
import numpy as np

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

import multiprocessing
from joblib import Parallel, delayed
from tqdm.autonotebook import trange

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import utils, model_selection, metrics

from datetime import datetime, timedelta
from typing import List
import lightgbm as lgb
import glob2
import os 

import sys
sys.path.insert(1, '/home/fefespinola/ETHZ_Fall_2020/flirt-1')
import flirt.simple

def get_features_per_subject(path, window_length):
    features = flirt.simple.get_features_for_empatica_archive(zip_file_path = path,
                                      window_length = window_length,
                                      window_step_size = 0.25,
                                      hrv_features = True,
                                      eda_features = True,
                                      acc_features = True,
                                      bvp_features = True,
                                      temp_features = True,
                                      debug = True)
    return features

def find_label_timestamps(csv_path, StartingTime):

    ID = csv_path.split('/', 3)[2]
    df_timestamp = pd.read_csv(glob2.glob('project_data/WESAD/' + ID + '/*quest.csv')[0], delimiter = ';', header = 1).iloc[:2, :].dropna(axis = 1)
    print('===================================')
    print('Printing the timestamp for {0}'.format(ID))
    print('===================================')
    print(df_timestamp.head())
    
    # Start/End of experiment periods
    print('\nStart of the baseline: ' + str(df_timestamp['Base'][0]))
    print('End of the baseline: ' + str(df_timestamp['Base'][1]))
    print('Start of the fun: ' + str(df_timestamp['Fun'][0]))
    print('End of the fun: ' + str(df_timestamp['Fun'][1]))
    print('Start of the stress: ' + str(df_timestamp['TSST'][0]))
    print('End of the stress: ' + str(df_timestamp['TSST'][1]))
    
    # Get start and end time and assign label into a dict
    lab_dict = {'Base':0, 'TSST':1, 'Fun':2}
    labels_times_dict = {}
    for mode in df_timestamp.columns.tolist():
        print('mode', mode)
        if mode=='Base' or mode=='Fun' or mode=='TSST':
            labels_times_dict[mode] = [StartingTime + timedelta(minutes = int(str(df_timestamp[mode][0]).split(".")[0]))+ timedelta                                         
                                    (seconds = int(str(df_timestamp[mode][0]).split(".")[1])), 
                                    StartingTime + timedelta(minutes = int(str(df_timestamp[mode][1]).split(".")[0])) + timedelta                                           
                                    (seconds = int(str(df_timestamp[mode][1]).split(".")[1])), lab_dict[mode]]
            
            #labels_times_dict[mode] = [StartingTime + timedelta(minutes = float(df_timestamp[mode][0])), 
            #                      StartingTime + timedelta(minutes = float(df_timestamp[mode][1])), lab_dict[mode]]
        
    return labels_times_dict

def render_metric(eval_results, metric_name):
    ax = lgb.plot_metric(evals_result, metric=metric_name, figsize=(10, 5))
    #plt.show()
    plt.savefig('/home/fefespinola/ETHZ_Fall_2020/plots/render_metric_eda_ekf.png')

def render_plot_importance(gbm, importance_type, max_features=10, ignore_zero=True, precision=3):
    ax = lgb.plot_importance(gbm, importance_type=importance_type,
                             max_num_features=max_features,
                             ignore_zero=ignore_zero, figsize=(12, 8),
                             precision=precision)
    #plt.show()
    plt.savefig('/home/fefespinola/ETHZ_Fall_2020/plots/feature_importance_eda_ekf.png')

def __get_subset_features(df_all, feature_name: str, eda_method:str='lpf'):
    
    if feature_name=='physio':
        small_df = df_all.loc[:, df_all.columns.str.startswith('hrv')&df_all.columns.str.startswith   ('eda')&df_all.columns.str.startswith('bvp')&df_all.columns.str.startswith('temp')]
        filename = 'features_all_' + features_name +'_' + eda_method + '_feat.csv'
    else:
        small_df = df_all.loc[:, df_all.columns.str.startswith(feature_name)]
        if feature_name=='eda':
            filename = 'features_all_' + features_name +'_' + eda_method + '_feat.csv'
        else:
            filename = 'features_all_' + features_name + '_feat.csv'
    small_df.to_csv(filename)

def __get_train_valid_data(df_all, cv_subject):
    scaler = StandardScaler()

    #training data
    X_train = df_all.loc[df_all['ID']!=cv_subject]  # 500 entities, each contains 10 features
    X_train = X_train.iloc[:, 1:len(df_all.columns)-1]
    X_train = X_train.replace([np.nan, np.inf, -np.inf], np.nan)
    #for col in X_train.columns.tolist():
    #    median = np.median(X_train[col].loc[X_train[col]!=np.nan])
    #    X_train[col].replace([np.nan], median, inplace=True)
    
    #X_train = X_train.dropna(axis=1)
    X_train = scaler.fit_transform(X_train)
    y_train = df_all.loc[df_all['ID']!=cv_subject, ['label']]  # binary target
    train_data = lgb.Dataset(X_train, label=y_train)

    #validation data
    X_test = df_all.loc[df_all['ID']==cv_subject]  # 500 entities, each contains 10 features
    X_test = X_test.iloc[:, 1:len(df_all.columns)-1]
    X_test = X_test.replace([np.nan, np.inf, -np.inf], np.nan)
    #for col in X_test.columns.tolist():
    #    median = np.median(X_test[col].loc[X_test[col]!=np.nan])
    #    X_test[col].replace([np.nan], median, inplace=True)
    
    #X_test = X_test.dropna(axis=1)
    X_test = scaler.transform(X_test)
    y_test = df_all.loc[df_all['ID']==cv_subject, ['label']]   # binary target
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    return X_train, y_train, train_data, X_test, y_test, test_data
 
def main():
    os.chdir('/home/fefespinola/ETHZ_Fall_2020/') #local directory where the script is
    df_all = pd.DataFrame(None)
    #relevant_features = pd.DataFrame(None)
    File_Path = glob2.glob('project_data/WESAD/**/*_readme.txt', recursive=True)
    window_length = 60 # in seconds
    window_shift = 0.25 # in seconds
    for subject_path in File_Path:
        print(subject_path)
        print(subject_path.split('/', 3)[2])
        ID = subject_path.split('/', 3)[2]
        zip_path = glob2.glob('project_data/WESAD/' + ID + '/*_Data.zip')[0]
        print(zip_path)
        features = get_features_per_subject(zip_path, window_length)
        features.index.name = 'timedata'
        StartingTime = features.index[0]
        print(features)
        labels_times = find_label_timestamps(subject_path, StartingTime)
        relevant_features = features.loc[
            ((features.index >= labels_times['Base'][0]) & (features.index <= labels_times['Base'][1])) 
            | ((features.index >= labels_times['Fun'][0]) & (features.index <= labels_times['Fun'][1])) 
            | ((features.index >= labels_times['TSST'][0]) & (features.index <= labels_times['TSST'][1]))]

        relevant_features.insert(0, 'ID', ID)
        relevant_features['label'] = np.zeros(len(relevant_features))
        relevant_features.loc[(relevant_features.index>=labels_times['Fun'][0]) &
                                (relevant_features.index<=labels_times['Fun'][1]), 'label'] = labels_times['Fun'][2]
        relevant_features.loc[(relevant_features.index>=labels_times['TSST'][0]) & 
                            (relevant_features.index<=labels_times['TSST'][1]), 'label'] = labels_times['TSST'][2]

        # concatenate all subjects and add IDs
        df_all = pd.concat((df_all, relevant_features))

    return df_all

if __name__ == '__main__':
    df_all = main()
    df_all.to_csv('/home/fefespinola/ETHZ_Fall_2020/features_all_all_lr_0_1_feat.csv')
    df_all = pd.read_csv('/home/fefespinola/ETHZ_Fall_2020/features_all_all_lr_0_1_feat.csv')
    df_all.set_index('timedata', inplace=True)
    print(df_all.ID)
    
    ### for binary classification uncomment line below 
    #df_all['label'].replace(2, 0, inplace=True)
    #print('===== BINARY ====')

    print('---start classification---')

    df = df_all.replace([np.inf, -np.inf], np.nan) # np.inf leads to problems with some techniques

    # Clean columns that contain a lot of nan values 
    print(len(df), len(df.columns))
    df = df.dropna(axis=1, thresh=int(len(df)*0.8))
    print(len(df), len(df.columns))
    print('Columns dropped: ', df_all.drop(df.columns, axis=1).columns.values)

    stats = []

    cv = model_selection.LeaveOneGroupOut()

    X = df.drop(columns=['label', 'ID'])
    y = df['label'].astype('int')
    groups = df['ID']
    print("running %d-fold CV..." % (cv.get_n_splits(X, y, groups)))

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(15, 15))

    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #### for binary classification uncomment line below
        #params = {'objective': 'binary', 'is_unbalance': True}
        params = {'objective': 'multiclass', 'is_unbalance': True}
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        stats.append({
            'f1': metrics.f1_score(y_test, y_pred, average="macro")})
            
        
        print(metrics.classification_report(y_test, y_pred))

    stats = pd.DataFrame(stats)
    print(stats.f1.mean())

    '''
    #parameters
    param = {'metric': 'auc_mu', 'learning_rate': 0.01, 'num_leaves': 60, 'max_depth': 9, 'is_unbalance':True,
        'verbose': 1, 'objective':'multiclass', 'num_class':3, 'lambda_l1': 30, 'force_col_wise':True,
        'feature_fraction': 0.6, 'predict_contrib':True, 'min_data_in_leaf': 100, 'max_bin': 400, 'bagging_fraction':0.6}

    subjects = ['S2', 'S3', 'S4', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

    ##### Start Classification
    f1_tot = 0
    f1_dict = {}
    num_iterations = 550
    evals_tot = np.zeros((num_iterations,))
    train_tot = np.zeros((num_iterations,))
    #data with validation set
    for subj in subjects:
        # get data
        print('===Training for LOSO ', subj, '===')
        _, _, train_data, X_test, y_test, test_data = __get_train_valid_data(df_all, subj)
    
        evals_result = {}  # to record eval results for plotting

        #train normally
        bst_norm = lgb.train(param, train_data, num_boost_round=num_iterations, valid_sets=[train_data, test_data],                        
                                    evals_result=evals_result, verbose_eval=10)
        
        train_tot = np.add(train_tot, evals_result['training']['auc_mu'][:])
        evals_tot = np.add(evals_tot, evals_result['valid_1']['auc_mu'][:])
        #best_iteration = np.argmax(evals_result['valid_1']['auc_mu'][100:]) + 1
        #print ('best iter', best_iteration)
        print('n_features', bst_norm.num_feature())
        print('feature contrib', bst_norm.feature_importance(), np.shape(bst_norm.feature_importance()))
        y_pred = bst_norm.predict(X_test)#, num_iteration=best_iteration)
        y_pred = np.argmax(y_pred, axis=1)

        f1_metric = f1_score(y_test, y_pred, average='macro')
        f1_dict[subj] = f1_metric
        print(f1_dict)
        f1_tot = f1_tot + f1_metric
        print(confusion_matrix(y_test, y_pred))


    f1_tot = f1_tot/len(subjects)
    print(f1_tot)
    evals_tot_dict = {'training':{'auc_mu':train_tot/len(subjects)}, 'valid_1':{'auc_mu':evals_tot/len(subjects)}}
    render_metric(evals_tot_dict, param['metric'])
    render_plot_importance(bst_norm, importance_type='split')
    '''
