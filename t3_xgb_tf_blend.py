import pandas as pd
import numpy as np

import xgboost as xgb
from scipy.sparse import csr_matrix, hstack

from t3_feature_extraction import FeatureExtractor, preprocess
from t3_utils import target_transform, target_inverse_transform, xgb_eval
from t3_tfidf import get_tfidf

from sklearn.model_selection import train_test_split


df = pd.read_csv('../raw_data/train.csv', delimiter=';', index_col=0)
df = preprocess(df)

df_test = pd.read_csv('../raw_data/test.csv', delimiter=';', index_col=0)
df_test = preprocess(df_test)

sample_subm = pd.read_csv('../raw_data/t3_sample_submission.csv',
                          delimiter=';', index_col=0)
df = df.sort_index()
df_test = df_test.sort_index()

y = df['item_views']

y = y.sort_index()

df_train, df_val, y_train, y_val = (
    train_test_split(df, y, test_size=0.2, random_state=100))

df_train.sort_index(inplace=True)
df_val.sort_index(inplace=True)
y_train = y_train.sort_index()
y_val = y_val.sort_index()

num_boost_round = 600

n_features_list = [
    6000,
    6000,
    7000,
    9000,
    8000
]


params_list = [
    {
        'objective': 'reg:linear',
        'silent': 1,
        'eta': 0.025,
        'nthread': 8,
        'max_depth': 45,
        'min_child_weight': 80,
    },
    {
        'objective': 'reg:linear',
        'silent': 1,
        'eta': 0.025,
        'nthread': 8,
        'max_depth': 25,
        'min_child_weight': 0,
    },
    {
        'objective': 'reg:linear',
        'silent': 1,
        'eta': 0.025,
        'nthread': 8,
        'max_depth': 35,
        'min_child_weight': 140,
    },
    # {
    #     'objective': 'reg:linear',
    #     'silent': 1,
    #     'eta': 0.05,
    #     'nthread': 8,
    #     'max_depth': 20,
    #     'min_child_weight': 280,
    # },
    {
        'objective': 'reg:linear',
        'silent': 1,
        'eta': 0.025,
        'nthread': 8,
        'max_depth': 45,
        'min_child_weight': 40,
    }
]

oof = []
pred = []
oof.append(y_val)

for n_features, params in zip(n_features_list, params_list):

    # VAL

    # tfidf = get_tfidf(n_features=n_features)
    # tfidf.fit(df_train.title)
    # X_tfidf = tfidf.transform(df_train.title)
    # X_tfidf_val = tfidf.transform(df_val.title)

    fe = FeatureExtractor()
    # fe.fit(df_train, df_val)
    # X_train = fe.transform(df_train)
    # X_val = fe.transform(df_val)

    # X_train = hstack([csr_matrix(X_train.values), X_tfidf])
    # X_val = hstack([csr_matrix(X_val.values), X_tfidf_val])

    # dtrain = xgb.DMatrix(X_train, target_transform(y_train))
    # dval = xgb.DMatrix(X_val, y_val)
    # watchlist = [(dval, 'val')]

    # xgbm = xgb.train(
    #     params, dtrain, num_boost_round,
    #     evals=watchlist, early_stopping_rounds=10,
    #     verbose_eval=False, feval=xgb_eval)

    # best_iter = xgbm.best_iteration
    # y_pred = target_inverse_transform(xgbm.predict(dval))

    # oof.append(y_pred)

    # PRED

    tfidf = get_tfidf(n_features=n_features)
    tfidf.fit(df.title)
    X_tfidf = tfidf.transform(df.title)
    X_tfidf_test = tfidf.transform(df_test.title)

    fe.fit(df, df_test)
    X = fe.transform(df)
    X_test = fe.transform(df_test)

    X = hstack([csr_matrix(X.values), X_tfidf])
    X_test = hstack([csr_matrix(X_test.values), X_tfidf_test])

    dtrain = xgb.DMatrix(X, target_transform(y))
    dtest = xgb.DMatrix(X_test)

    xgbm = xgb.train(params, dtrain, num_boost_round)

    best_iter = xgbm.best_iteration
    y_pred = target_inverse_transform(xgbm.predict(dtest))

    pred.append(y_pred)

# columns = ['y_true', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5']
# oof_dict = {col: values for col, values in zip(columns, oof)}
# oof_df = pd.DataFrame(oof_dict)

columns = ['y_1', 'y_2', 'y_3', 'y_4']
pred_dict = {col: values for col, values in zip(columns, pred)}
pred_df = pd.DataFrame(pred_dict)

# oof_df.to_csv('../own_data/xgb_blend_oof.csv', index=False)
pred_df.to_csv('../own_data/xgb_blend_pred.csv', index=False)

y_pred = pred_df.mean(axis=1).values

sample_subm = pd.read_csv('../raw_data/t3_sample_submission.csv',
                          delimiter=';', index_col=0)

sample_subm['item_views '] = y_pred.astype(np.uint64)
sample_subm.to_csv('../submissions/t3_0902_xgb_blend_2.csv', sep=';')
