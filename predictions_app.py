# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:29:58 2024

@author: mzing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, feat_with_num_enc=['default', 'housing', 'loan']):
        self.feat_with_num_enc = feat_with_num_enc

    def fit(self, df):
        return self

    def transform(self, df):
        if set(self.feat_with_num_enc).issubset(df.columns):
            # Change 0 to No and 1 to Yes for all the features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df.loc[:, ft] = df[ft].map({1: 'Yes', 0: 'No'})
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

############################# Streamlit interface ############################

st.write("""
# Bank Marketing Prediction App
This app predicts if the client will subscribe to a term deposit. Just fill in the customer details and click on the Predict button.
""")


# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select customer age', value=30,
                      min_value=18, max_value=100, step=1)


# Job dropdown
st.write("""
## Job
""")
jobs = ['Blue-Collar', 'Management', 'Technician', 'Admin', 'Services', 'Retired', 'Self-Employed', 'Entrepreneur', 'Unemployed', 'Housemaid', 'Student']
input_job = st.selectbox(
    'Select customer job', jobs)


# Mariral status input
st.write("""
## Marital status
""")
input_marital_status = st.radio('Select customer marital status',['Married','Single', 'Divorced'], index=0)


# Education input
st.write("""
## Education
""")
input_education = st.radio('Select customer education',['Secondary','Tertiary', 'Primary'], index=0)


# default input
st.write("""
## Default
""")
input_default = st.radio('Has Credit in Default?',['Yes','No'], index=0)


# Balance
st.write("""
## Balance
""")
customer_balance = st.number_input('Average Yearly Balance', value=1000)


# Housing input
st.write("""
## Housing
""")
input_housing = st.radio('Has Housing Loan?',['Yes','No'], index=0)


# Loan input
st.write("""
## Loan
""")
input_loan = st.radio('Has Personal Loan?',['Yes','No'], index=0)


# contact input
st.write("""
## Contact Method
""")
input_contact = st.radio('Contact Communication Type',['Cellular','Telephone'], index=0)



# Duration
st.write("""
## Duration 
""")
customer_duration = st.number_input('Last Contact Duration (seconds)', value=100)



# outcome input
st.write("""
## Previous Outcome
""")
input_outcome = st.radio('Outcome of Previous Campaign',['Failure','Other', 'Success'], index=0)


st.markdown('##')
st.markdown('##')


# list of all the inputs
profile_to_predict = [input_age, input_job, input_marital_status, input_education, input_default, customer_balance,
                      input_housing, input_loan, input_contact, 0, '', customer_duration,
                      0, 0, 0, input_outcome, 0, '']


# Convert to dataframe with column names
profile_to_predict_df = pd.DataFrame(
    [profile_to_predict], columns=train_copy.columns)


train_copy_with_profile_to_pred = pd.concat([train_copy,profile_to_predict_df],ignore_index=True)


train_copy_prep = pipeline.fit_transform(train_copy)

test_copy_prep = pipeline.fit_transform(test_copy)

X_train_copy_prep = train_copy_prep.iloc[:,:-1]

y_train_copy_prep = train_copy_prep.iloc[:,-1]


X_test_copy_prep = test_copy_prep.iloc[:,:-1]


y_test_copy_prep = test_copy_prep.iloc[:,-1]



train_copy_with_profile_to_pred = pipeline.fit_transform(train_copy_with_profile_to_pred)

profile_to_pred_prep = train_copy_with_profile_to_pred.iloc[-1:,:-1]


# Function to load the model
def load_model():
    try:
        model = joblib.load('bagging_model.sav')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
# Predict button
if st.button('Predict'):
    with st.spinner('Predicting...'):
        loaded_model = load_model()
        if loaded_model:
            prediction = loaded_model.predict(profile_to_pred_prep)
            prediction_proba = loaded_model.predict_proba(profile_to_pred_prep)
            formatted_prediction = 'The client will subscribe to the term deposit.' if prediction[0] == 1 else 'The client will not subscribe to the term deposit.'
            probability = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            st.markdown(f"**Prediction: {formatted_prediction}**", unsafe_allow_html=True)
            st.markdown(f"**Probability of this prediction: {probability:.2%}**", unsafe_allow_html=True)
            st.markdown(f"**Client Profile Summary:**")
            st.markdown(f"- **Age**: {input_age}")
            st.markdown(f"- **Job**: {input_job}")
            st.markdown(f"- **Marital Status**: {input_marital_status}")
            st.markdown(f"- **Education**: {input_education}")
            st.markdown(f"- **Credit Default**: {input_default}")
            st.markdown(f"- **Average Yearly Balance**: €{customer_balance:,.2f}")
            st.markdown(f"- **Housing Loan**: {input_housing}")
            st.markdown(f"- **Personal Loan**: {input_loan}")
            st.markdown(f"- **Contact Communication Type**: {input_contact}")
            st.markdown(f"- **Duration of Last Contact**: {customer_duration} seconds")
            st.markdown(f"- **Outcome of Previous Campaign**: {input_outcome}")
            
            # Display balloons if the client will subscribe
            if prediction[0] == 1:
                st.balloons()