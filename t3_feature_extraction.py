# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer

from t3_utils import target_transform


class FeatureExtractor(object):
    cat_cols = [
        'owner_type',
        'category',
        'subcategory',
        'param1',
        'param2',
        'param3',
        'region',
        ]

    cont_cols = [
        'price',
        'title_len',
        'param1_price_diff',
        'subcategory_price_diff',
        # 'first_capital',
        'has_symbols',

    ]

    other_cols = [
        'hour',
        'day',
    ]

    def __init__(self, scale=False):
        self.le_dict = {}

        self.agg = Aggregates()

        self.scale = scale
        self.ss = StandardScaler()
        self.scale_cols = []
        self.means = {}

    def fit(self, X_train, X_test):

        X = X_train.copy()
        X['item_views'] = target_transform(X['item_views'])

        if self.scale:
            X['price'] = np.log(X['price'] + 1)

        self.agg.fit(X)

        X = pd.concat([X_train, X_test])

        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.le_dict[col] = le

        if self.scale:
            X = self.agg.transform(X)
            self.scale_cols = self.cont_cols + self.agg.cols
            self.ss.fit(X.loc[:, self.scale_cols].fillna(0))

    def transform(self, X):

        if self.scale:
            X['price'] = np.log(X['price'] + 1)
        X = self.agg.transform(X)

        X['param1_price_diff'] = ((X['price'] - X['param1_price_median']) /
                                  X['param1_price_median'])
        X['param1_price_diff'].replace([np.inf, -np.inf], 1, inplace=True)
        X['param1_price_diff'].fillna(1, inplace=True)
        X['subcategory_price_diff'] = (
            (X['price'] - X['subcategory_price_median']) /
            X['subcategory_price_median'])
        X['subcategory_price_diff'].replace([np.inf, -np.inf], 1, inplace=True)
        X['subcategory_price_diff'].fillna(1, inplace=True)
        # X['category_price_diff'] = (
        #     (X['price'] - X['category_price_mean']) /
        #     X['category_price_mean'])

        # X['first_capital'] = X['title'].apply(lambda t: t[0].isupper()).astype(np.int)
        X['has_symbols'] = X['title'].apply(has_symbols)
        X['title_len'] = X['title'].apply(lambda t: len(t))

        for col in self.cat_cols:
            le = self.le_dict[col]
            X.loc[:, col] = le.transform(X.loc[:, col])

        if self.scale:
            X.loc[:, self.scale_cols] = (
                self.ss.transform(X.loc[:, self.scale_cols]))

        X = X.loc[:, self.cont_cols + self.cat_cols +
                  self.agg.cols + self.other_cols]

        return X

    # def fit_transform(self, X):
    #     self.fit(X)
    #     return self.transform(X)


def transform_ohe(X, X_test):
    X = X.copy()
    X_test = X_test.copy()

    fe = FeatureExtractor(scale=False)
    fe.fit(X, X_test)
    X = fe.transform(X)
    X_test = fe.transform(X_test)

    cat_cols = [
            'owner_type',
            'category',
            'subcategory',
            'param1',
            'param2',
            'param3',
            'region'
        ]

    cat_indices = [i for i in range(X.shape[1]) if X.columns[i] in cat_cols]

    ohe = OneHotEncoder(categorical_features=cat_indices, sparse=True)


    ohe.fit(pd.concat([X, X_test]))
    ohe_cols = ohe.transform(X)
    ohe_cols_test = ohe.transform(X_test)

    return ohe_cols, ohe_cols_test, ohe.feature_indices_


class Aggregates(object):
    def __init__(self):
        self.aggrs = {}
        self.cols = []

    def add_aggr(self, X, col, func, nan_repl_col=None, agg_col=None):
        if not agg_col:
            agg_col = 'item_views'
        new_col = col + '_' + agg_col + '_' + func.__name__
        self.aggrs[new_col] = (
            col,
            X.groupby([col])[agg_col].agg({new_col: func}),
            nan_repl_col)

    def fit(self, X):

        self.overall_mean = X['item_views'].mean()

        # self.add_aggr(X, 'region', np.median)
        self.add_aggr(X, 'region', np.mean)
        self.add_aggr(X, 'region', np.size)
        # self.add_aggr(X, 'param1', np.median)
        self.add_aggr(X, 'param1', np.mean)
        self.add_aggr(X, 'param1', np.size)
        # self.add_aggr(X, 'subcategory', np.median)
        self.add_aggr(X, 'subcategory', np.mean)
        self.add_aggr(X, 'subcategory', np.size)
        # self.add_aggr(X, 'category', np.median)
        self.add_aggr(X, 'category', np.mean)
        # self.add_aggr(X, 'hour', np.median)
        self.add_aggr(X, 'hour', np.mean)
        self.add_aggr(X, 'hour', np.size)

        self.add_aggr(X, 'subcategory', np.median, agg_col='price')
        self.add_aggr(X, 'category', np.median, agg_col='price')
        self.add_aggr(X, 'param1', np.median, agg_col='price')

        # self.add_aggr(X, 'region_param1', np.mean,
        #               nan_repl_col='param1_item_views_mean')
        # self.add_aggr(X, 'region_param1_hour', np.mean,
        #               nan_repl_col='param1_views_mean')
        # self.add_aggr(X, 'region_subcategory', np.std,
        #               nan_repl_col='subcategory_views_mean')

        # self.add_aggr(X, 'region_hour', np.mean,
        #               nan_repl_col='hour_item_views_mean')
        # self.add_aggr(X, 'region_param1_param2', np.mean,
        #               nan_repl_col='param1_views_median')
        # self.add_aggr(X, 'region_param1_param2_param3', np.mean) # self.add_aggr(X, 'region_hour_param1', np.mean)

        self.cols = self.aggrs.keys()

    def transform(self, X):
        X = X.copy()
        for col, aggr in self.aggrs.iteritems():
            X = X.merge(aggr[1], left_on=aggr[0], how='left', right_index=True)
        # fill na
        for col, aggr in self.aggrs.iteritems():
            nan_repl_col = aggr[2]
            if nan_repl_col:
                X.loc[:, col].fillna(X.loc[:, nan_repl_col], inplace=True)

        print (X.loc[:, self.cols].isnull().any(axis=1)).sum()
        X.loc[:, self.cols] = X.loc[:, self.cols].fillna(0)
        X = X.sort_index()
        return X


def preprocess(X):
    X['start_time'] = pd.to_datetime(X['start_time'], errors='raise')
    X['hour'] = X['start_time'].apply(lambda x: int(x.hour) / 4)
    X['day'] = X['start_time'].apply(lambda x: x.day)

    text_columns = ['title', 'region', 'param1', 'param2', 'param3',
                    'subcategory', 'category']
    X.loc[:, text_columns] = (X.loc[:, text_columns]
                              .fillna('')
                              .applymap(lambda t: t.decode('utf-8')))

    region_dict = get_region_dict()
    X['region_2'] = X.region.apply(lambda r: region_dict[r] if r in
                              region_dict.keys() else r)


    X['region_param1'] = X.region_2 + ' ' + X.param1
    X['region_param1_hour'] = (X.region + ' ' + X.param1 +
                               ' ' + X.hour.astype(str))
    X['region_param1_param2'] = (X.region + ' ' + X.param1 + ' ' + X.param2)
    X['region_param1_param2_param3'] = (
        X.region + ' ' + X.param1 + ' ' + X.param2 + ' ' +
        X.param3)
    X['region_subcategory'] = X.region + ' ' + X.subcategory
    X['region_hour'] = X.region + ' ' + X.hour.astype(str)
    X['region_hour_param1'] = (X.region + ' ' + X.hour.astype(str) +
                               ' ' + X.param1)

    X['text'] = (X['title'] + ' ' + X['param1'] + ' ' +
                 X['param2'] + ' ' + X['param3'])

    X = X.fillna('NODATA')

    return X


def get_region_dict():
    region_groups = [
        [u'Ингушетия', u'Карачаево-Черкесия', u'Адыгея', u'Дагестан', u'Северная Осетия', u'Кабардино-Балкария', u'Чеченская республика'],
        [u'Чукотский АО', u'Магаданская область', u'Саха (Якутия)', u'Камчатский край'],
        [u'Приморский край', u'Сахалинская область', u'Еврейская АО', u'Амурская область'],
        [u'Ставропольский край', u'Калмыкия'],
        [u'Республика Алтай', u'Алтайский край', u'Тыва'],
        [u'Бурятия', u'Хакасия', u'Забайкальский край', u'Ханты-Мансийский АО'],
        [u'Ямало-Ненецкий АО', u'Ненецкий АО', u'Коми', u'Архангельская область'],
        [u'Новгородская область', u'Псковская область'],
        [u'Башкортостан', u'Мордовия', u'Марий Эл'],
        [u'Курская область', u'Брянская область', u'Орловская область'],
        [u'Свердловская область', u'Курганская область'],
        [u'Ярославская область', u'Костромская область'],
        [u'Ленинградская область', u'Санкт-Петербург']
    ]

    region_dict = {}
    for i, group in enumerate(region_groups):
        for reg in group:
            region_dict[reg] = u'group_{}'.format(i)

    return region_dict

symbols = u'!@#$%^&*"№;%:"`<>'


def has_symbols(s):
    if any(x in s for x in symbols):
        return 1
    else:
        return 0
