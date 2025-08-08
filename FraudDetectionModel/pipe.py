import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt


C = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
       'account_incoming_debt_vs_paid_0_24m', 'account_status',
       'account_worst_status_0_3m', 'account_worst_status_12_24m',
       'account_worst_status_3_6m', 'account_worst_status_6_12m', 'age',
       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'merchant_category',
       'merchant_group', 'has_paid', 'max_paid_inv_0_12m',
       'max_paid_inv_0_24m', 'name_in_email',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
       'num_arch_ok_12_24m', 'num_arch_rem_0_12m',
       'num_arch_written_off_0_12m', 'num_arch_written_off_12_24m',
       'num_unpaid_bills', 'status_last_archived_0_24m',
       'status_2nd_last_archived_0_24m', 'status_3rd_last_archived_0_24m',
       'status_max_archived_0_6_months', 'status_max_archived_0_12_months',
       'status_max_archived_0_24_months', 'recovery_debt',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv']

COLS = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
       'account_days_in_rem_12_24m', 'account_days_in_term_12_24m',
       'account_status', 'account_worst_status_0_3m',
       'account_worst_status_12_24m', 'account_worst_status_3_6m',
       'account_worst_status_6_12m', 'age', 'avg_payment_span_0_12m',
       'merchant_category', 'merchant_group', 'has_paid', 'max_paid_inv_0_12m',
       'max_paid_inv_0_24m', 'name_in_email',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
       'num_arch_ok_12_24m', 'num_unpaid_bills', 'status_last_archived_0_24m',
       'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
       'status_max_archived_0_24_months', 'recovery_debt',
       'sum_paid_inv_0_12m', 'time_hours']

CAT_COLS = ['account_status', 'account_worst_status_0_3m',
       'account_worst_status_12_24m', 'account_worst_status_3_6m',
       'account_worst_status_6_12m', 'merchant_category', 'merchant_group',
        'name_in_email', 'status_last_archived_0_24m',
       'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
       'status_max_archived_0_24_months']

NUM_COLS = ['account_amount_added_12_24m', 'account_days_in_dc_12_24m',
        'account_days_in_rem_12_24m', 'account_days_in_term_12_24m', 'age',
        'avg_payment_span_0_12m', 'has_paid', 'max_paid_inv_0_12m',
        'max_paid_inv_0_24m', 'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_dc_0_12m', 'num_arch_dc_12_24m', 'num_arch_ok_0_12m',
       'num_arch_ok_12_24m', 'num_unpaid_bills', 'recovery_debt',
       'sum_paid_inv_0_12m', 'time_hours']

def get_data():
    raw_data = 'data/dataset.csv'
    df = pd.read_csv(raw_data, sep=';', index_col='uuid')
    predict = df[df['default'].isnull()].copy()
    df.drop(index=predict.index, inplace=True)
    return df, predict

def clean_data(df):
    df.drop_duplicates(inplace=True)
    y = df['default']
    df.drop(columns='default', inplace=True)
    return df, y

def split_data(df, y):
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_pipeline(X, model):
    numeric_transformer = Pipeline(steps=[
        ('m_imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())]) #NUM_COLS (median is zero for cols account_days)

    categorical_transformer = Pipeline(steps=[
        ('z_imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))]) #CAT_COLS

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, NUM_COLS),
            ('cat', categorical_transformer, CAT_COLS)])

    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
    return pipe

# CV = GridSearchCV(pipeline, parameters, scoring = 'mean_absolute_error', n_jobs= 1)

def train(X_train, y_train, pipeline):
    return pipeline.fit(X_train, y_train)

def predict_result(X, pipe):
    if 'uuid' in X.columns:
        result = X[['uuid']].copy()
    else:
        result = pd.DataFrame(index=X.index, columns=['pd'])
    X_ =pd.DataFrame(X, columns=C)
    result['pd'] = pipe.predict_proba(X_)[:, 1]
    return result
