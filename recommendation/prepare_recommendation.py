import IPython
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import spatial
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    path_to_ds = os.path.join(cwd, 'recommendation_data.csv')
    ds_train = pd.read_csv(path_to_ds)
    # ds_train = ds_train.set_index('uea_ucr')
    # ds_train = ds_train.drop(columns=['dataset', 'in_both'])
    return ds_train

def get_train_test_ds(ds, num_test_ds):
    ds_names = ds['dataset'].unique().tolist()
    ds_test_names = random.sample(ds_names, num_test_ds)
    ds_train_names = list(set(ds_names).difference(set(ds_test_names)))

    ds_train = ds[ds['dataset'].isin(ds_train_names)].copy()
    ds_train.set_index('dataset')
    ds_test = ds[ds['dataset'].isin(ds_test_names)].copy()
    ds_test.set_index('dataset')
    return ds_train, ds_test

def prepare_training_dataset(ds_train, ds_test, alg_ds):
    cat_encoder = OrdinalEncoder()
    nb_cat = CategoricalNB()
    nb_gaus = GaussianNB()
    nb_brn = BernoulliNB()

    df_train = ds_train.join(alg_ds, on='dataset', how='inner')
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

    IPython.embed()


if __name__ == '__main__':

    df = get_trained_datasets()
    df_train, df_test = get_train_test_ds(df,6)
    df_train.set_index(['dataset'],inplace= True)
    df_test.set_index(['dataset'],inplace= True)
    df_train_ohe, df_test_ohe = one_hot_encode_cat_cols(df_train, df_test)
    df_train_std, df_test_std = standardize_num_cols(df_train_ohe, df_test_ohe)

    ## ohe + std
    # df_full = pd.concat((df_train_std, df_test_std))
    # y_train = df_train_std['harmonic_mean']
    # X_train = df_train_std.drop('harmonic_mean', axis= 1)
    # y_test = df_test_std['harmonic_mean']
    # X_test = df_test_std.drop('harmonic_mean', axis= 1)

    ## ohe only
    df_full = pd.concat((df_train_ohe, df_test_ohe))
    y_train = df_train_ohe['harmonic_mean']
    X_train = df_train_ohe.drop('harmonic_mean', axis= 1)
    y_test = df_test_ohe['harmonic_mean']
    X_test = df_test_ohe.drop('harmonic_mean', axis= 1)

    plt.subplots(figsize=(20,15))
    sns.heatmap(df_full.corr(),annot=True,lw=1)

    df_train_10 = df_train[df_train['revealed_pct'] == 10]
    df_test_10 = df_test[df_test['revealed_pct'] == 10]
    df_train_10_ohe, df_test_10_ohe = one_hot_encode_cat_cols(df_train_10, df_test_10)
    df_train_10_std, df_test_10_std = standardize_num_cols(df_train_10_ohe, df_test_10_ohe)
    df_10_full = pd.concat((df_train_10_ohe, df_test_10_ohe))
    y_train = df_train_10_ohe['harmonic_mean']
    X_train = df_train_10_ohe.drop('harmonic_mean', axis= 1)
    y_test = df_test_10_ohe['harmonic_mean']
    X_test = df_test_10_ohe.drop('harmonic_mean', axis= 1)

    # plt.subplots(figsize=(20,15))
    # sns.heatmap(df_10_full.corr(),annot=True,lw=1)

    x_train_plot = list(range(1, X_train.shape[0] + 1))
    x_test_plot = list(range(1, X_test.shape[0] + 1))

    # multiple linear regression
    import numpy as np
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    r_sq = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean absolute error for linear regression degree : {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"mean squared error for linear regression degree : {mse}")
    sns.scatterplot(x_test_plot, y_test)
    sns.scatterplot(x_test_plot, y_pred)


    import statsmodels.api as sm
    X_train_Sm= sm.add_constant(X_train)
    X_train_Sm= sm.add_constant(X_train)
    ls=sm.OLS(y_train,X_train_Sm).fit()
    print(ls.summary())

    # polynomial regression
    from sklearn.preprocessing import PolynomialFeatures
    transformer_2 = PolynomialFeatures(degree=2, include_bias=True)
    transformer_3 = PolynomialFeatures(degree=3, include_bias=True)
    transformer_4 = PolynomialFeatures(degree=4, include_bias=True)
    transformer_5 = PolynomialFeatures(degree=5, include_bias=True)
    
    transformer_2.fit(X_train)
    transformer_3.fit(X_train)

    X_train_poly = transformer_2.transform(X_train)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    X_test_poly = transformer_2.transform(X_test)
    r_sq = model.score(X_test_poly, y_test)
    y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean absolute error for polynomial regression degree 2 : {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"mean squared error for polynomial regression degree 2 : {mse}")
    sns.scatterplot(x_test_plot, y_test)
    sns.scatterplot(x_test_plot, y_pred)

    X_train_poly = transformer_3.transform(X_train)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    X_test_poly = transformer_3.transform(X_test)
    r_sq = model.score(X_test_poly, y_test)
    y_pred = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean absolute error for polynomial regression degree 3 : {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"mean squared error for polynomial regression degree 3 : {mse}")
    sns.scatterplot(x_test_plot, y_test)
    sns.scatterplot(x_test_plot, y_pred)

    # Fitting Regression Trees to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean absolute error for regression tree : {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"mean squared error for regression tree : {mse}")
    sns.scatterplot(x_test_plot, y_test)
    sns.scatterplot(x_test_plot, y_pred)

    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"mean absolute error for random forest : {mae}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"mean squared error for random forest : {mse}")
    sns.scatterplot(x_test_plot, y_test)
    sns.scatterplot(x_test_plot, y_pred)


    # # import export_graphviz
    # from sklearn.tree import export_graphviz 
    # # export the decision tree to a tree.dot file
    # # for visualizing the plot easily anywhere
    # export_graphviz(regressor, out_file ='tree.dot')

    # export_graphviz(regressor, out_file ='tree.dot',
    #             feature_names =['Production Cost']) 



    IPython.embed()

    # similarity between datasets
    # cos_result_df = get_cosine_matrix(ds_train, ds_test)

    # prepare_training_dataset(ds_train, ds_test, alg_ds)


    plt.savefig('image_name', dpi=200)
    plt.clf()
    plt.cla()
    plt.close()


    import numpy as np
    import pandas as pd
    from scipy import stats
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # applying various transformations
    # log = np.log(df_10_full.copy()) # log 
    # log_corr = log.corr()
    # log_heatmap = log_corr[log_corr['harmonic_mean'].notnull()][['harmonic_mean']]
    # sns.heatmap(log_heatmap,annot=True,lw=1)

    # sqrt = np.sqrt(df_10_full.copy()) # square root
    # sqrt_corr = sqrt.corr()
    # sqrt_heatmap = sqrt_corr[sqrt_corr['harmonic_mean'].notnull()][['harmonic_mean']]
    # sns.heatmap(sqrt_heatmap,annot=True,lw=1)

    # boxcox, _ = stats.boxcox(df_10_full.copy()) # boxcox
    # boxcox_corr = boxcox.corr()
    # boxcox_heatmap = boxcox_corr[boxcox_corr['harmonic_mean'].notnull()][['harmonic_mean']]
    # sns.heatmap(boxcox_heatmap,annot=True,lw=1)


    # x = df["GrLivArea"].copy() # original data



    # df_1 = pd.concat((df_train, df_test))