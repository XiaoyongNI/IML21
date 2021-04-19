import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import svm
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.model_selection import KFold

from scipy.special import expit
from imblearn.over_sampling import SMOTE
from collections import Counter
from score_submission import get_score

def DataLoad(trainx_path='train_features.csv', trainy_path='train_labels.csv', testx_path='test_features.csv'):
    print("Loading Data")

    train_features = pd.read_csv(trainx_path)
    train_labels = pd.read_csv(trainy_path)
    test_features = pd.read_csv(testx_path)

    # print(train_features.shape)
    # print(train_labels.shape)

    return train_features, train_labels, test_features

def data_imputation(data):
    results = np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]//12):
        for j in range(data.shape[1]):
            tmp = data[12*i:12*(i+1),j]
            tmp_pd = pd.DataFrame(tmp).interpolate(method = 'linear',limit_direction = 'both').replace(np.nan,0)
            results[12*i:12*(i+1),j] = tmp_pd.values.reshape((12,),order = 'C')
    return results

def Preprocessing(xdata):
    print("Preprocessing")

    # transform to ndarry
    xarray = xdata.values

    # get pid ID
    pidID = xarray[np.arange(0, len(xdata.index), 12), 0]

    # remove 'pid' and 'time'
    xarray = xarray[:, 2:]

    # feature imputation
    # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp_mean.fit(xarray)
    # data_fitted = imp_mean.transform(xarray)
    # data_fitted = data_imputation(xarray)

    # transform data dim to (pid# * 12d) and impute
    x_transformed = []
    for i in range(xarray.shape[0] // 12):
        pid_i = xarray[i * 12:(i + 1) * 12, :]
        # impute nan with median value of that pid i
        median_pidi = np.nanmedian(pid_i, axis=0)
        index = np.where(np.isnan(pid_i))
        pid_i[index] = np.take(median_pidi, index[1])
        # transform data dim
        pid_i = np.reshape(pid_i,-1)
        x_transformed.append(pid_i)

    # extract mean and divide by std
    processer = RobustScaler()
    x_transformed = processer.fit_transform(x_transformed)

    return np.array(x_transformed), pidID

def CV_model_eval(model, x, y, evalType):
    print("Evaluating model")

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_mean = 0
    train_mean = 0

    for train_index, cv_index in kf.split(x):
        model_eval = clone(model)
        x_train = x[train_index]
        y_train = y[train_index]
        x_cv = x[cv_index]
        y_cv = y[cv_index]

        model_eval.fit(x_train, y_train)
        result_cv = model_eval.predict(x_cv)
        result_train = model_eval.predict(x_train)

        if evalType == 'classification':
            score_cv = roc_auc_score(result_cv, y_cv)
            score_train = roc_auc_score(result_train, y_train)
        elif evalType == 'regression':
            score_cv = r2_score(result_cv, y_cv)
            score_train = r2_score(result_train, y_train)

        cv_mean += score_cv
        train_mean += score_train

    cv_mean /= k_folds
    train_mean /= k_folds
    print("Validation Set Score: {}".format(cv_mean))
    print("Training Set Score: {}".format(train_mean))



def main():
    # Choices
    oversample_dataset_sub1 = False
    oversample_dataset_sub2 = True
    model_sub1 = 'svm' 
    model_sub2 = 'xgboost'
    model_sub3 = 'ridgeCV'

    # Data loading
    trainx_path = 'train_features.csv'
    trainy_path = 'train_labels.csv'
    y_pred_save_path = 'train_labels_pred.csv'
    testx_path = 'test_features.csv'
    y_test_save_path = 'test_labels.csv'
    train_features, train_labels, test_features = DataLoad(trainx_path='train_features.csv', trainy_path='train_labels.csv', testx_path='test_features.csv')   
    
    yarray = train_labels.values   
    yarray = yarray[:, 1:] # remove 'pid'

    # Preprocesing
    trainx_processed, pidID_train = Preprocessing(train_features)
    testx_processed, pidID_test = Preprocessing(test_features)

    # print(trainx_processed.shape)

    # Pre-defined pd dataframe for saving predicting results
    test_labels = pd.DataFrame()
    train_labels_pred = pd.DataFrame()

    for i in range(0, train_labels.columns.shape[0] - 1):
        
        labelName = train_labels.columns[i + 1]
        yarray_i = yarray[:, i]
        
        # first 10 labels for Subtask 1
        if 0 <= i <= 9: 
            # Do over-sampling to combat imbalanced dataset
            if oversample_dataset_sub1:
                trainx_processed, yarray_i = over_sampling(trainx_processed, yarray_i, oversample_dataset)          
            
            if model_sub1 == 'svm':
                clf = svm.SVC(gamma='auto', class_weight='balanced')
            elif model_sub1 == 'xgboost':
                # imbalanced classification: scale_pos_weight=count(negative examples)/count(Positive examples)
                count = Counter(yarray_i)              
                scale_pos_weight_val = count[0] / count[1]
                clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_val)
            
            model1 = clf.fit(trainx_processed, yarray_i)
            CV_model_eval(model1, trainx_processed, yarray_i, 'classification')

            if model_sub1 == 'svm':
                testy_score = model1.decision_function(testx_processed)
                trainy_score = model1.decision_function(trainx_processed)
                trainy = []
                testy = []
                for i in range(0, len(testy_score)):
                    testy.append(expit(testy_score[i])) # tranform to range [0,1]
                for i in range(0, len(trainy_score)):
                    trainy.append(expit(trainy_score[i])) # tranform to range [0,1]
                testy = np.array(testy)
                trainy = np.array(trainy)
            elif model_sub1 == 'xgboost':
                test_score = model1.predict_proba(testx_processed)
                testy = test_score[:, 1] 
                train_score = model1.predict_proba(trainx_processed)
                trainy = train_score[:, 1]   
            
            test_labels['pid'] = pidID_test
            test_labels['{}'.format(labelName)] = testy
            train_labels_pred['pid'] = pidID_train
            train_labels_pred['{}'.format(labelName)] = trainy

        
        # label 11 for Subtask 2
        if i == 10:
            # Do over-sampling to combat imbalanced dataset
            if oversample_dataset_sub2:
                trainx_processed, yarray_i = over_sampling(trainx_processed, yarray_i, oversample_dataset) 

            if model_sub2 == 'svm':
                clf = svm.SVC(gamma='auto', class_weight='balanced')
            elif model_sub2 == 'xgboost':
                # imbalanced classification: scale_pos_weight=count(negative examples)/count(Positive examples)
                count = Counter(yarray_i)              
                scale_pos_weight_val = count[0] / count[1]
                clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_val)
            
            model2 = clf.fit(trainx_processed, yarray_i)
            CV_model_eval(model2, trainx_processed, yarray_i, 'classification')

            if model_sub2 == 'svm':
                testy_score = model2.decision_function(testx_processed)
                trainy_score = model2.decision_function(trainx_processed)
                trainy = []
                testy = []
                for i in range(0, len(testy_score)):
                    testy.append(expit(testy_score[i])) # tranform to range [0,1]
                for i in range(0, len(trainy_score)):
                    trainy.append(expit(trainy_score[i])) # tranform to range [0,1]
                testy = np.array(testy)
                trainy = np.array(trainy)
            elif model_sub2 == 'xgboost':
                test_score = model2.predict_proba(testx_processed)
                testy = test_score[:, 1] 
                train_score = model2.predict_proba(trainx_processed)
                trainy = train_score[:, 1]   
            
            test_labels['pid'] = pidID_test
            test_labels['{}'.format(labelName)] = testy
            train_labels_pred['pid'] = pidID_train
            train_labels_pred['{}'.format(labelName)] = trainy

        # Subtask 3
        """
        Reference: https://github.com/adodd202/HousingPricesML
        """ 
        else: 
            if model_sub3 == 'ridgeCV':
                reg = RidgeCV(alphas=(0.01,1,100), fit_intercept=False, cv=5)                    
            elif model_sub3 == 'xgboost':
                reg = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                       learning_rate=0.05, max_depth=3,
                                       min_child_weight=1.7817, n_estimators=2200,
                                       reg_alpha=0.4640, reg_lambda=0.8571,
                                       subsample=0.5213, silent=1,
                                       random_state=7, nthread=-1)

            model3 = reg.fit(trainx_processed, yarray_i)
            CV_model_eval(model3, trainx_processed, yarray_i, 'regression')
    
            testy = model3.predict(testx_processed)
            test_labels['pid'] = pidID_test
            test_labels['{}'.format(labelName)] = testy

            trainy = model3.predict(trainx_processed)
            train_labels_pred['pid'] = pidID_train
            train_labels_pred['{}'.format(labelName)] = trainy

    test_labels.to_csv(y_test_save_path, index=True)
    train_labels_pred.to_csv(y_pred_save_path, index=True)

    get_score(pd.read_csv(y_path),pd.read_csv(y_pred_save_path))

    test_labels.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')

if __name__ == '__main__':
    main()







