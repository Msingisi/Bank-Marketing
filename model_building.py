# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:33:09 2024

@author: mzing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib 
import streamlit as st

training = pd.read_csv('dataset/train.csv')
testing = pd.read_csv('dataset/test.csv')
full_data = pd.concat([training, testing], axis=0) 
full_data = full_data.sample(frac=1).reset_index(drop=True)

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train, test = data_split(full_data, 0.2)

train_copy = train.copy()
test_copy = test.copy()

####################### Classes used to preprocess the data ##############################

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=None):
        self.feat_with_outliers = feat_with_outliers or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = set(self.feat_with_outliers) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        # Calculate 25% and 75% quantiles
        Q1 = X[self.feat_with_outliers].quantile(0.25)
        Q3 = X[self.feat_with_outliers].quantile(0.75)
        IQR = Q3 - Q1

        # Keep data within 3 IQR
        X = X[~((X[self.feat_with_outliers] < (Q1 - 3 * IQR)) | (X[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
        return X
    
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, col_with_skewness=None):
        self.col_with_skewness = col_with_skewness or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = set(self.col_with_skewness) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        X[self.col_with_skewness] = np.cbrt(X[self.col_with_skewness])
        return X
    
class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_num_enc=['default','housing','loan']):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_num_enc).issubset(df.columns)):
            # Change 0 to No and 1 to Yes for all the features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:'Yes',0:'No'})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_imputed_ft=None, median_imputed_ft=None):
        self.mode_imputed_ft = mode_imputed_ft or []
        self.median_imputed_ft = median_imputed_ft or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = set(self.mode_imputed_ft + self.median_imputed_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        # Drop missing values in the target feature
        X.dropna(inplace=True, axis=0, subset=['y'])

        # Impute missing values with mode
        for ft in self.mode_imputed_ft:
            the_mode = X[ft].mode()[0]
            X[ft] = X[ft].fillna(the_mode)

        # Impute missing values with median
        for ft in self.median_imputed_ft:
            the_median = X[ft].median()
            X[ft] = X[ft].fillna(the_median)

        return X
    
class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=None):
        self.one_hot_enc_ft = one_hot_enc_ft or []
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.one_hot_enc.fit(X[self.one_hot_enc_ft])
        self.feat_names_one_hot_enc = self.one_hot_enc.get_feature_names_out(self.one_hot_enc_ft)
        return self

    def transform(self, X):
        missing_features = set(self.one_hot_enc_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        one_hot_enc_df = pd.DataFrame(self.one_hot_enc.transform(X[self.one_hot_enc_ft]).toarray(), columns=self.feat_names_one_hot_enc, index=X.index)
        rest_of_features = [ft for ft in X.columns if ft not in self.one_hot_enc_ft]
        df_concat = pd.concat([one_hot_enc_df, X[rest_of_features]], axis=1)
        return df_concat
    
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        self.feature_to_drop = feature_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = set(self.feature_to_drop) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        return X.drop(columns=self.feature_to_drop)
    
class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=None):
        self.min_max_scaler_ft = min_max_scaler_ft or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = set(self.min_max_scaler_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        min_max_enc = MinMaxScaler()
        X[self.min_max_scaler_ft] = min_max_enc.fit_transform(X[self.min_max_scaler_ft])
        return X
    
class Oversample(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'y' in df.columns:
            # smote function to oversample the minority class to fix the imbalance data
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'y'],df['y'])
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("y is not in the dataframe")
            return df
        
# Create the pipeline
pipeline = Pipeline([
    ('missing value imputer', MissingValueImputer(mode_imputed_ft=['job', 'education', 'contact', 'poutcome'])),
    ('outlier remover', OutlierRemover(feat_with_outliers=['age', 'balance'])),
    ('skewness transformer', SkewnessHandler(col_with_skewness=['age', 'balance', 'duration'])),
    ('binning_num_to_yn', BinningNumToYN()),
    ('drop features', DropFeatures(feature_to_drop=['month', 'day_of_week', 'duration_bins', 'campaign', 'pdays', 'previous'])),
    ('one hot encoding', OneHotWithFeatNames(one_hot_enc_ft=['job', 'marital', 'education', 'contact', 'poutcome', 'default','housing','loan'])),
    ('min max scaler', MinMaxWithFeatNames(min_max_scaler_ft=['age', 'balance', 'duration'])),
    ('oversample', Oversample())
])

# Apply the pipeline to the DataFrame
train_pipe_prep = pipeline.fit_transform(train_copy)
test_pipe_prep = pipeline.fit_transform(test_copy)


# split the train data into X and y (target)
X_train, y_train = train_pipe_prep.loc[:, train_pipe_prep.columns != 'y'], train_pipe_prep['y'].astype('int64')

# split the test data into X and y (target)
X_test, y_test = test_pipe_prep.loc[:, test_pipe_prep.columns != 'y'], test_pipe_prep['y'].astype('int64')


# initialize Random forest regression  with default parameters
bagging_model=BaggingClassifier(n_estimators=15,max_samples=1.0, max_features=0.6, 
                                bootstrap_features=True, bootstrap=True)

# fit model with training data
bagging_model.fit(X_train, y_train)

# make predictions
prediction=bagging_model.predict(X_test)

print('Model Accuracy:', accuracy_score(prediction, y_test))


joblib.dump(bagging_model, "bagging.sav")