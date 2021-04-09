import os
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB
import random

cwd = os.path.dirname(__file__)

def get_categorical_columns(df):
    return list(df.select_dtypes(include=['object']).columns.values)

def get_numerical_columns(df):
    return list(df.select_dtypes(include=[np.number]).columns.values)

def get_binary_columns(df):
    return list(df.select_dtypes(include=[bool]).columns.values)


def standardize_num_cols(df_train, df_test):
    # copy of datasets
    df_train_stand = df_train.copy()
    df_test_stand = df_test.copy()
    # combine both so that ranges of values are based on both train and test
    df_full = pd.concat((df_train, df_test))

    # numerical features
    num_cols = get_numerical_columns(df_full)

    # apply standardization on numerical features
    for i in num_cols:
        # fit on training data column
        scale = StandardScaler().fit(df_full[[i]])

        # transform the training data column
        df_train_stand[i] = scale.transform(df_train_stand[[i]])

        # transform the testing data column
        df_test_stand[i] = scale.transform(df_test_stand[[i]])
    return df_train_stand, df_test_stand


def one_hot_encode_cat_cols(df_train, df_test):
    # copy of datasets and label train and test datasets
    df_train_ohe = df_train.copy()
    df_train_ohe['train'] = 1
    df_test_ohe = df_test.copy()
    df_test_ohe['train'] = 0
    # combine both so that OHE includes all values from both train and test
    df_full = pd.concat((df_train_ohe, df_test_ohe))

    # categorical features
    cat_cols = get_categorical_columns(df_full)
    df_full_ohe = pd.get_dummies(df_full, columns=cat_cols, drop_first=False)

    df_train_ohe = df_full_ohe[df_full_ohe['train'] == 1]
    df_train_ohe = df_train_ohe.drop(columns=['train'])
    df_test_ohe = df_full_ohe[df_full_ohe['train'] == 0]
    df_test_ohe = df_test_ohe.drop(columns=['train'])
    return df_train_ohe, df_test_ohe


def get_cosine_matrix(df_train, df_test):
    # Standardize and one hot encode features
    ds_train_stand, ds_test_stand = standardize_num_cols(df_train, df_test)
    ds_train_transformed, ds_test_transformed = one_hot_encode_cat_cols(
        ds_train_stand, ds_test_stand)

    # calculate cosine similarity
    cos = cosine_distances(ds_test_transformed, ds_train_transformed)
    cos_df = pd.DataFrame(cos, index=ds_test_transformed.index,
                          columns=ds_train_transformed.index)
    cos_result_df = cos_df.idxmin(axis=1)

    return cos_result_df


def get_trained_datasets():
    path_to_ds = os.path.join(cwd, 'ds.csv')
    ds_train = pd.read_csv(path_to_ds)
    # ds_train = ds_train.set_index('uea_ucr')
    # ds_train = ds_train.drop(columns=['dataset', 'in_both'])
    return ds_train

def get_train_test_ds(ds, num_test_ds):
    ds_names = ds['dataset'].unique().tolist()
    ds_test_names = random.sample(ds_names, num_test_ds)
    ds_train_names = list(set(ds_names).difference(set(ds_test_names)))

    ds_train = ds[ds['dataset'].isin(ds_train_names)].copy()
    ds_test = ds[ds['dataset'].isin(ds_test_names)].copy()
    return ds_train, ds_test

def prepare_training_dataset(ds_train, ds_test, alg_ds):
    cat_encoder = OrdinalEncoder()
    nb_cat = CategoricalNB()
    nb_gaus = GaussianNB()
    nb_brn = BernoulliNB()

    df_train = ds_train.join(alg_ds, on='uea_ucr', how='inner')
    X_train = df_train.drop(columns=['best_algorithm'])
    X_train_cat = X_train[get_categorical_columns(X_train)]
    X_train_cat_encoded = pd.DataFrame(cat_encoder.fit_transform(
        X_train_cat), index=X_train_cat.index, columns=X_train_cat.columns)
    X_train_num = X_train[get_numerical_columns(X_train)]
    X_train_bool = X_train[get_binary_columns(X_train)]
    y_train = df_train['best_algorithm']

    nb_cat.fit(X_train_cat_encoded, y_train)
    nb_gaus.fit(X_train_num, y_train)
    nb_brn.fit(X_train_bool, y_train)

    df_test = ds_test.join(alg_ds, on='uea_ucr', how='inner')
    X_test = df_test.drop(columns=['best_algorithm'])
    X_test_cat = X_test[get_categorical_columns(X_test)]
    X_test_cat_encoded = pd.DataFrame(cat_encoder.fit_transform(
        X_test_cat), index=X_test_cat.index, columns=X_test_cat.columns)
    X_test_num = X_test[get_numerical_columns(X_test)]
    X_test_bool = X_test[get_binary_columns(X_test)]
    y_test = df_test['best_algorithm']

    categorical_probas = nb_cat.predict_proba(X_test_cat_encoded)
    gaussian_probas = nb_gaus.predict_proba(X_test_num)
    bernoulli_probas = nb_brn.predict_proba(X_test_bool)

    import IPython
    IPython.embed()


if __name__ == '__main__':

    ds = get_trained_datasets()
    ds_train, ds_test = get_train_test_ds(ds,2)

    # similarity between datasets
    cos_result_df = get_cosine_matrix(ds_train, ds_test)

    # prepare_training_dataset(ds_train, ds_test, alg_ds)
