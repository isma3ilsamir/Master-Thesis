import IPython
import sys
import os

import scipy as sp
MAIN_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
list_dirs = next(os.walk(MAIN_DIRECTORY))[1]
curr_folder =  os.path.dirname(os.path.realpath(__file__))
seed_folder = None
for dir in list_dirs:
    dir = os.path.join(MAIN_DIRECTORY, dir)
    if dir not in sys.path:
        sys.path.insert(0, dir)

import joblib
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import confusion_matrix, accuracy_score, hamming_loss, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.tree import export_graphviz
from dtreeviz.trees import dtreeviz
from subprocess import call
from pickle import dump, load
from sklearn.model_selection import GridSearchCV

from prepare_analysis import rank_on_col, rank_relative_to_dummy, get_pivot



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

        # export scaler
        export_folder = os.path.join(seed_folder, 'scaler')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        dump(scale, open(os.path.join(
                export_folder, f'scaler_{i}.pkl'), 'wb'))
    return df_train_stand, df_test_stand

def standardize_num_cols_new_dataset(df_train):
    # export scaler
    import_folder = os.path.join(seed_folder, 'scaler')

    # copy of datasets
    df_train_stand = df_train.copy()

    # numerical features
    num_cols = get_numerical_columns(df_train_stand)

    # load scaler and apply standardization on numerical features
    for i in num_cols:
        # load the scaler
        scale = load(open(os.path.join(import_folder, f'scaler_{i}.pkl'), 'rb'))

        # transform the training data column
        df_train_stand[i] = scale.transform(df_train_stand[[i]])

    return df_train_stand

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

    ohe = OneHotEncoder(sparse= False)
    ohe.fit(df_full[cat_cols])
    ohe_result = ohe.transform(df_full[cat_cols])
    df_ohe_result = pd.DataFrame(ohe_result, columns= ohe.get_feature_names(cat_cols))
    df_full.reset_index(inplace= True)
    df_concat = pd.concat([df_full, df_ohe_result], axis=1)
    df_concat.drop(cat_cols, axis=1, inplace= True)
    df_concat.set_index('dataset', inplace= True)

    df_train_ohe = df_concat[df_concat['train'] == 1]
    df_train_ohe = df_train_ohe.drop(columns=['train'])
    df_test_ohe = df_concat[df_concat['train'] == 0]
    df_test_ohe = df_test_ohe.drop(columns=['train'])

    # export ohe
    export_folder = os.path.join(seed_folder, 'one_hot_encoder')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    dump(ohe, open(os.path.join(
            export_folder, f'one_hot_encoder.pkl'), 'wb'))
    return df_train_ohe, df_test_ohe


def one_hot_encode_cat_cols_new_dataset(df_train):
    # export encoder
    import_folder = os.path.join(seed_folder, 'one_hot_encoder')

    # load the scaler
    ohe = load(open(os.path.join(import_folder, f'one_hot_encoder.pkl'), 'rb'))

    # copy of datasets
    df_train_ohe = df_train.copy()
    # categorical features
    cat_cols = get_categorical_columns(df_train_ohe)

    ohe_result = ohe.transform(df_train_ohe[cat_cols])
    df_ohe_result = pd.DataFrame(ohe_result, columns= ohe.get_feature_names(cat_cols))
    df_train_ohe.reset_index(inplace= True)
    df_concat = pd.concat([df_train_ohe, df_ohe_result], axis=1)
    df_concat.drop(cat_cols, axis=1, inplace= True)
    df_concat.set_index('dataset', inplace= True)

    return df_concat


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


def fit_polynomial_regression(X_train, X_test, y_train, y_test, degrees):
    # initialise y_train_pred and y_test_pred matrices to store the train and test predictions
    # each row is a data point, each column a prediction using a polynomial of some degree
    y_train_pred = np.zeros((len(X_train), len(degrees)))
    y_test_pred = np.zeros((len(X_test), len(degrees)))

    x_train_plot = list(range(1, X_train.shape[0] + 1))
    x_test_plot = list(range(1, X_test.shape[0] + 1))

    for i, degree in enumerate(degrees):
        # make pipeline: create features, then feed them to linear_reg model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

        # predict on test and train data
        # store the predictions of each degree in the corresponding column
        y_train_pred[:, i] = model.predict(X_train)
        y_test_pred[:, i] = model.predict(X_test)

    # train data
    plt.scatter(x_train_plot, y_train)
    plt.title("Train data")
    for i, degree in enumerate(degrees):
        plt.scatter(x_train_plot, y_train_pred[:, i], s=15, label=str(degree))
        plt.legend(loc='upper left')
    plt.show()

    plt.scatter(x_test_plot, y_test)
    plt.title("Test data")
    for i, degree in enumerate(degrees):
        plt.scatter(x_test_plot, y_test_pred[:, i], s=15, label=str(degree))
        plt.legend(loc='upper left')
    plt.show()


def get_trained_datasets():
    path_to_ds = os.path.join(curr_folder, 'recommendation_data.csv')
    ds_train = pd.read_csv(path_to_ds)
    # ds_train = ds_train.set_index('uea_ucr')
    # ds_train = ds_train.drop(columns=['dataset', 'in_both'])
    return ds_train


def get_train_test_ds(ds, num_test_ds, seed):
    random.seed(seed)
    ds_names = ds['dataset'].unique().tolist()
    ds_test_names = random.sample(ds_names, num_test_ds)
    ds_train_names = list(set(ds_names).difference(set(ds_test_names)))

    ds_train = ds[ds['dataset'].isin(ds_train_names)].copy()
    ds_train.set_index('dataset')
    ds_test = ds[ds['dataset'].isin(ds_test_names)].copy()
    ds_test.set_index('dataset')
    return ds_train, ds_test


def preprocess_data(df_train, df_test, pct, ohe, std):
    if pct:
        df_train_pct = df_train[df_train['revealed_pct'] == pct]
        df_test_pct = df_test[df_test['revealed_pct'] == pct]
    else:
        df_train_pct = df_train
        df_test_pct = df_test
    y_train = df_train_pct['f_score']
    X_train = df_train_pct.drop('f_score', axis=1)
    y_test = df_test_pct['f_score']
    X_test = df_test_pct.drop('f_score', axis=1)
    if ohe:
        X_train, X_test = one_hot_encode_cat_cols(X_train, X_test)
    if std:
        X_train, X_test = standardize_num_cols(X_train, X_test)
    return X_train, X_test, y_train, y_test


def preprocess_new_data(df_train, pct, ohe, std):
    if pct:
        df_train_pct = df_train[df_train['revealed_pct'] == pct]
    else:
        df_train_pct = df_train
    X_train = df_train_pct.copy()
    if ohe:
        X_train = one_hot_encode_cat_cols_new_dataset(X_train)
    if std:
        X_train = standardize_num_cols_new_dataset(X_train)
    return X_train


def train_random_forest(df_train, df_test, pct, ohe, std, random_state):
    X_train, X_test, y_train, y_test = preprocess_data(
        df_train, df_test, pct, ohe, std)
    # Fitting Random Forest Regression to the dataset
    final_res = {}
    res = {}
    res['X_train'] = X_train
    res['X_test'] = X_test
    res['y_train'] = y_train
    res['y_test'] = y_test

    # Create the parameter grid based on the results of random search
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [2, 3, 4, 5, 10, 50, 100, 250, 500]
    }
    # Create a based model
    rf = RandomForestRegressor(bootstrap= True, oob_score= True, max_features= 'auto', random_state= random_state)

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv = 10,
        n_jobs = -1,
        verbose = 2
        )
    # Fit the random search model
    grid_search.fit(X_train, y_train)

    # regressor = RandomForestRegressor(
    #     n_estimators=2, random_state=0, max_depth=None)
    # regressor.fit(X_train, y_train)
    regressor = grid_search.best_estimator_

    res['RF_Reg'] = regressor
    f_score_pred = regressor.predict(X_test)
    res['f_score_pred'] = f_score_pred
    f_score_pred_train = regressor.predict(X_train)
    res['f_score_pred_train'] = f_score_pred_train
    mae = mean_absolute_error(y_test, f_score_pred)
    res['mae'] = mae
    print(f"MAE for random forest on {pct}%: {mae}")
    mse = mean_squared_error(y_test, f_score_pred)
    res['mse'] = mse
    print(f"MSE for random forest on {pct}%: {mse}")
    r_sq = regressor.score(X_test, y_test)
    res['r_sq'] = r_sq
    print(f"r squared for random forest on {pct}%: {r_sq}")
    rmse = mean_squared_error(y_test, f_score_pred, squared= False)
    res['rmse'] = rmse
    print(f"RMSE for random forest on {pct}%: {rmse}")
    res['train_prediction_df'] = get_prediction_df(
        df_train, f_score_pred_train, pct)
    res['test_prediction_df'] = get_prediction_df(df_test, f_score_pred, pct)
    final_res[pct] = res
    return final_res


def get_prediction_df(df_actual, f_score_pred, pct):
    df_copy = df_actual[df_actual['revealed_pct'] == pct].copy()
    df_copy = df_copy[['classifier', 'revealed_pct']]
    df_copy['f_score_pred'] = f_score_pred
    df_copy.reset_index(inplace=True)
    return df_copy


def get_bad_results(row):
    actual = row['ds_rank_f_score_calc']
    pred = row['ds_rank_f_score_pred_calc']
    return False if actual == pred else True


def export_feature_importance(X_train, RF_reg, pct):
    export_folder = os.path.join(seed_folder, 'feature_importance')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    # features importance
    y = RF_reg.feature_importances_
    # plot
    fig, ax = plt.subplots()
    width = 0.4  # the width of the bars
    ind = np.arange(len(y))  # the x locations for the groups
    ax.barh(ind, y, color='green')
    ax.set_yticks(ind+width/10)
    ax.set_yticklabels(X_train.columns, minor=False)
    fig.set_size_inches(6.5, 4.5, forward=True)
    plt.title(f'Feature importance of RandomForest Classifier on the {pct}%')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(export_folder, f'relative_feature_imp_{pct}pct'),
                dpi=200, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    df_feat_imp = pd.DataFrame(list(zip(X_train.columns, y)), columns=[
                               'Feature', 'Importance'])
    export_csv(df_feat_imp, 'feature_importance', f'feature_imp_{pct}pct')


def export_scatter_plot(df_actual, f_score_pred, pct, image_name, train):
    ds_step = 'Train Data Set' if train else 'Test Data Set'
    data = df_actual[df_actual['revealed_pct'] == pct].copy()
    x_plot = list(range(1, data.shape[0] + 1))
    sns.scatterplot(x=x_plot, y=data['f_score'], hue=data['classifier'])
    sns.scatterplot(x=x_plot, y=f_score_pred, color=".2", marker="+", label='Predicted')
    sns.set_style('ticks')
    plt.legend(bbox_to_anchor=(1.04, 1),
               loc="upper left", title='F-Score Values')
    plt.title(f'Actual vs. Predicted f_score for {pct}% ({ds_step})')
    plt.ylabel('F-Score')
    plt.xlabel(f'Record Number')
    plt.ylim(0, 1.1)
    export_folder = os.path.join(seed_folder, 'scatter_plots')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    plt.savefig(os.path.join(export_folder, image_name),
                dpi=200, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def get_diff_actual_pred(df_actual, f_score_pred, pct):
    res = {}
    df_copy = df_actual[df_actual['revealed_pct'] == pct].copy()
    df_copy = df_copy[['classifier', 'revealed_pct', 'f_score']]
    df_copy['f_score_pred'] = f_score_pred
    df_copy['diff'] = df_copy['f_score_pred'] - df_copy['f_score']
    df_copy['diff_abs'] = df_copy['diff'].abs()
    df_copy.reset_index(inplace=True)
    res[pct] = df_copy
    return res


def get_rank_by_col(df, col):
    df_temp_ranked = rank_on_col(df, col)
    df_temp_ranked = rank_relative_to_dummy(df_temp_ranked, col)
    return df_temp_ranked


def get_pct_ranking_eval(df, pct, train_res):
    step = 'Training' if train_res else 'Testing'
    num_ds = len(df['dataset'].unique())
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        df['ds_rank_f_score_calc'], df['ds_rank_f_score_pred_calc']).ravel()
    # accuracy
    acc = accuracy_score(df['ds_rank_f_score_calc'],
                         df['ds_rank_f_score_pred_calc'])
    # hamming loss
    ham_loss = hamming_loss(df['ds_rank_f_score_calc'],
                            df['ds_rank_f_score_pred_calc'])
    # f1
    f1_sc = f1_score(df['ds_rank_f_score_calc'],
                         df['ds_rank_f_score_pred_calc'])
    # roc auc
    roc_auc = roc_auc_score(df['ds_rank_f_score_calc'],
                         df['ds_rank_f_score_pred_calc'])
    # precision
    precision = precision_score(df['ds_rank_f_score_calc'],
                         df['ds_rank_f_score_pred_calc'])
    # recall
    recall = recall_score(df['ds_rank_f_score_calc'],
                         df['ds_rank_f_score_pred_calc'])

    total = tp + tn + fp + fn

    d = {
        'pct': [pct],
        'step': [step],
        'num_ds': [num_ds],
        'tp': [tp],
        'tn': [tn],
        'fp': [fp],
        'fn': [fn],
        'acc': [acc],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_sc],
        'roc_auc': [roc_auc],
        'ham_loss': [ham_loss],
        'total' : [total]
    }
    res_df = pd.DataFrame(data=d)
    return res_df


def get_model_pct_eval(df_train, df_test, pct, mae, mse, r_sq, rmse):
    num_ds_train = len(df_train['dataset'].unique())
    num_instances_train = df_train.shape[0]
    num_ds_test = len(df_test['dataset'].unique())
    num_instances_test = df_test.shape[0]
    d = {
        'pct': [pct],
        'num_ds_train': [num_ds_train],
        'num_instances_train' : [num_instances_train],
        'num_ds_test': [num_ds_test],
        'num_instances_test' : [num_instances_test],
        'mae': [mae],
        'mse': [mse],
        'r_sq': [r_sq],
        'rmse': [rmse],
    }
    res_df = pd.DataFrame(data=d)
    return res_df


def export_model(RF_Reg, pct):
    export_folder = os.path.join(seed_folder, 'models')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    joblib.dump(RF_Reg, os.path.join(
                export_folder, f'{pct}_pct_random_forest.joblib'))


def load_model(pct):
    import_folder = os.path.join(seed_folder, 'models')
    loaded_model = joblib.load(os.path.join(
        import_folder, f'{pct}_pct_random_forest.joblib'))
    return loaded_model


def export_csv(df, folder_name, file_name):
    export_folder = os.path.join(seed_folder, folder_name)
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    df.to_csv(os.path.join(
        export_folder, f'{file_name}.csv'), index=False)


def export_trees(RF_Reg, X_train, y_train, pct):
    export_folder = os.path.join(seed_folder, 'tree_diagrams')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    # export the decision tree to a tree.dot file
    # for visualizing the plot easily anywhere
    for i in range(len(RF_Reg.estimators_)):
        # viz = dtreeviz(RF_Reg.estimators_[0], X_train, y_train, feature_names=X_train.columns, target_name="Target")
        # viz.save_svg()
        export_graphviz(RF_Reg.estimators_[i],
                        feature_names=X_train.columns,
                        filled=True,
                        rounded=True,
                        out_file= os.path.join(
                            export_folder, f'{pct}_pct_tree_{i}.dot'))
        # Convert to png
        call(['dot', '-Tpng', os.path.join(
                            export_folder, f'{pct}_pct_tree_{i}.dot'), '-o', os.path.join(
                            export_folder, f'{pct}_pct_tree_{i}.png') , '-Gdpi=100'])

def get_new_datasets():
    path_to_ds = os.path.join(curr_folder, 'new_datasets.csv')
    ds = pd.read_csv(path_to_ds)
    return ds

def prepare_new_datasets_results(df_train, f_score_pred, pct):
    df = get_prediction_df(df_train, f_score_pred, pct)
    df = get_rank_by_col(df, 'f_score_pred')
    return df


if __name__ == '__main__':
    evaluate_on_existing_datasets = True
    pcts = [10, 20, 30, 100]
    classifiers = ['ST', 'CBoss', 'TSF', 'PForest', 'WEASEL', 'Dummy']

    if evaluate_on_existing_datasets:
        df = get_trained_datasets()

        seed_folders = []

        for x in range(50):
            seed_folder = os.path.join(curr_folder, f'recommendation_seed_{x}')
            seed_folders.append(seed_folder)

        #     df_train, df_test = get_train_test_ds(df, 15, x)
        #     df_train.set_index(['dataset'], inplace=True)
        #     # df_train.drop(['type'], axis= 1 , inplace= True)
        #     df_test.set_index(['dataset'], inplace=True)
        #     # df_test.drop(['type'], axis= 1 , inplace= True)

        # #     # plt.subplots(figsize=(20,15))
        # #     # sns.heatmap(df_10_full.corr(),annot=True,lw=1)

        #     ######### Fitting Random Forest Regression to the dataset ##########
        #     res_list = []
        #     res_diff_test_list = []
        #     res_diff_train_list = []
        #     res_pct_ranking_eval_list = []
        #     res_pct_fscore_eval_list = []

        #     for p in pcts:
        #         # train random forests
        #         res = train_random_forest(df_train, df_test, p, True, False, x)
        #         res_list.append(res)

        #         # get difference between actual and predicted test data sets
        #         res_diff_test = get_diff_actual_pred(
        #             df_test, res[p]['f_score_pred'], p)

        #         # get difference between actual and predicted train data sets
        #         res_diff_train = get_diff_actual_pred(
        #             df_train, res[p]['f_score_pred_train'], p)

        #         # draw actual vs predited plots
        #         export_scatter_plot(
        #             df_test, res[p]['f_score_pred'], p, f'scatter_test_actual_vs_pred_{p}pct', False)
        #         export_scatter_plot(
        #             df_train, res[p]['f_score_pred_train'], p, f'scatter_train_actual_vs_pred_{p}pct', True)

        #         # rank test ds
        #         res_test_ranked = get_rank_by_col(res_diff_test[p], 'f_score')
        #         res_test_ranked = get_rank_by_col(res_test_ranked, 'f_score_pred')
        #         res_diff_test_list.append(res_test_ranked)
        #         # results of ranking compared to actual
        #         res_pct_ranking_eval_test = get_pct_ranking_eval(
        #             res_test_ranked, p, False)
        #         res_pct_ranking_eval_list.append(res_pct_ranking_eval_test)

        #         # rank train ds
        #         res_train_ranked = get_rank_by_col(res_diff_train[p], 'f_score')
        #         res_train_ranked = get_rank_by_col(
        #             res_train_ranked, 'f_score_pred')
        #         res_diff_train_list.append(res_train_ranked)
        #         # results of ranking compared to actual
        #         res_pct_ranking_eval_train = get_pct_ranking_eval(
        #             res_train_ranked, p, True)
        #         res_pct_ranking_eval_list.append(res_pct_ranking_eval_train)

        #         # export model
        #         export_model(res[p]['RF_Reg'], p)

        #         # export feature importance graph
        #         export_feature_importance(res[p]['X_train'], res[p]['RF_Reg'], p)

        #         # export trees
        #         # export_trees(res[p]['RF_Reg'], res[p]['X_train'], res[p]['y_train'], p)

        #         # model f-score evaluation
        #         res_pct_fscore_eval = get_model_pct_eval(res[p]['train_prediction_df'], res[p]
        #                                                 ['test_prediction_df'], p, res[p]['mae'], res[p]['mse'], res[p]['r_sq'], res[p]['rmse'])
                # res_pct_fscore_eval_list.append(res_pct_fscore_eval)

            # # export fscore and ranking values
            # df_pct_res_train = pd.concat(res_diff_train_list, axis=0, sort=True)
            # df_pct_res_train['step'] = 'Training'
            # df_pct_res_test = pd.concat(res_diff_test_list, axis=0, sort=True)
            # df_pct_res_test['step'] = 'Testing'
            # df_pct_res_all = pd.concat(
            #     [df_pct_res_train, df_pct_res_test], axis=0, sort=True)
            # df_pct_res_all['bad_results'] = df_pct_res_all.apply(get_bad_results, axis=1)
            # export_csv(df_pct_res_all, 'results',
            #         'trained_ds_fscore_ranking_values')

            # # export evaluation metrics for f-score predictions
            # df_pct_fscore_eval = pd.concat(
            #     res_pct_fscore_eval_list, axis=0, sort=True)
            # export_csv(df_pct_fscore_eval, 'results',
            #         'trained_ds_fscore_evaluation')

            # # export evaluation metrics for recommendations based on ranking
            # df_pct_ranking_eval = pd.concat(
            #     res_pct_ranking_eval_list, axis=0, sort=True)
            # export_csv(df_pct_ranking_eval, 'results',
            #         'trained_ds_ranking_evaluation')

        # compare the different models
        fscore_dfs = []
        ranking_dfs = []
        values_dfs = []
        feature_imp_dfs = []
        for i in seed_folders:
            fscore_path = os.path.join(i, 'results', 'trained_ds_fscore_evaluation.csv')
            df = pd.read_csv(fscore_path)
            df['model_seed'] = i.split('_')[-1]
            fscore_dfs.append(df)

            ranking_path = os.path.join(i, 'results', 'trained_ds_ranking_evaluation.csv')
            df = pd.read_csv(ranking_path)
            df['model_seed'] = i.split('_')[-1]
            ranking_dfs.append(df)

            values_path = os.path.join(i, 'results', 'trained_ds_fscore_ranking_values.csv')
            df = pd.read_csv(values_path)
            df['model_seed'] = i.split('_')[-1]
            values_dfs.append(df)

            for p in pcts:
                imp_path = os.path.join(i, 'feature_importance', f'feature_imp_{p}pct.csv')
                df = pd.read_csv(imp_path)
                df['model_seed'] = i.split('_')[-1]
                df['pct'] = p
                feature_imp_dfs.append(df)

        df_fscores_eval = pd.concat(fscore_dfs)
        df_fscores_eval.to_csv(os.path.join(curr_folder, 'fscore_comparison.csv'), index= False)
        
        df_ranking_eval = pd.concat(ranking_dfs)
        df_ranking_eval.to_csv(os.path.join(curr_folder, 'ranking_comparison.csv'), index= False)
        
        df_values_eval = pd.concat(values_dfs)
        df_values_eval.to_csv(os.path.join(curr_folder, 'values_comparison.csv'), index= False)
        
        df_fimp_eval = pd.concat(feature_imp_dfs)
        df_fimp_eval.to_csv(os.path.join(curr_folder, 'feature_importance_comparison.csv'), index= False)
        import IPython
        IPython.embed()

        sns.set(font_scale=3)  # crazy big
        fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharey=False)
        fig.suptitle('Distribution of RMSE by Revealed%')
        # 10
        sns.histplot(ax=axes[0,0], data= df_fscores_eval[df_fscores_eval['pct'] == 10], x="rmse", kde=True)
        axes[0,0].set_title('10% chunk learner')
        axes[0,0].set(xlabel='RMSE', ylabel='Frequency', ylim=(0, 20), xlim=(0, 0.3))
        # 20
        sns.histplot(ax=axes[0,1], data= df_fscores_eval[df_fscores_eval['pct'] == 20], x="rmse", kde=True)
        axes[0,1].set_title('20% chunk learner')
        axes[0,1].set(xlabel='RMSE', ylabel='Frequency', ylim=(0, 20), xlim=(0, 0.3))
        # 30
        sns.histplot(ax=axes[1,0], data= df_fscores_eval[df_fscores_eval['pct'] == 30], x="rmse", kde=True)
        axes[1,0].set_title('30% chunk learner')
        axes[1,0].set(xlabel='RMSE', ylabel='Frequency', ylim=(0, 20), xlim=(0, 0.3))
        # 100
        sns.histplot(ax=axes[1,1], data= df_fscores_eval[df_fscores_eval['pct'] == 100], x="rmse", kde=True)
        axes[1,1].set_title('100% chunk learner')
        axes[1,1].set(xlabel='RMSE', ylabel='Frequency', ylim=(0, 20), xlim=(0, 0.3))
        plt.subplots_adjust(hspace = 0.5)
        plt.savefig(os.path.join(e, f'hist_rmse.jpg'),
                        dpi=200, bbox_inches='tight')

        plt.clf()
        plt.cla()
        plt.close()

        for i in pcts:
            print(f'Results of {i} percent')
            q = df_fscores_eval[df_fscores_eval['pct'] == i]
            w = df_ranking_eval[(df_ranking_eval['pct'] == i) & (df_ranking_eval['step'] == 'Testing')]
            import IPython
            IPython.embed()
            
            e = os.path.join(curr_folder, 'general charts')
            if not os.path.exists(e):
                os.makedirs(e)
            print(q.describe())
            q[['rmse']].boxplot()
            plt.ylim(0, 1)
            plt.savefig(os.path.join(e, f'boxplot_{i}pct_rmse.jpg'),
                dpi=200, bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()
            q[['rmse']].hist()
            plt.xlim(0, 1)
            plt.savefig(os.path.join(e, f'hist_{i}pct_rmse.jpg'),
                dpi=200, bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()

            print(w.describe())
            w[['acc']].boxplot()
            plt.ylim(0, 1)
            plt.savefig(os.path.join(e, f'boxplot_{i}pct_acc.jpg'),
                dpi=200, bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()
            w[['acc']].hist()
            plt.xlim(0, 1)
            plt.savefig(os.path.join(e, f'hist_{i}pct_acc.jpg'),
                dpi=200, bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()


    else:
        df_new_ds = get_new_datasets()
        df_new_ds_extended = pd.concat([df_new_ds['dataset']] * len(pcts) , keys = pcts).reset_index(level = 1, drop = True).rename_axis('revealed_pct').reset_index()
        df_new_ds_extended = pd.concat([df_new_ds_extended] * len(classifiers) , keys = classifiers).reset_index(level = 1, drop = True).rename_axis('classifier').reset_index()
        df_new_ds_extended = df_new_ds_extended.merge(df_new_ds, on=['dataset'], how='left')
        df_new_ds_extended.set_index(['dataset'], inplace= True)

        seed = 0
        seed_folder = os.path.join(curr_folder, f'recommendation_seed_{seed}')

        for p in pcts:
            # load the model
            df_train = df_new_ds_extended[df_new_ds_extended['revealed_pct'] == p]
            loaded_rf = load_model(p)
            X_train = preprocess_new_data(df_train, p, True, False)
            f_score_pred = loaded_rf.predict(X_train)
            res = prepare_new_datasets_results(df_train, f_score_pred, p)
            res.to_csv(os.path.join(seed_folder,'new_datasets_prediction',f'{p}_pct.csv'))
            IPython.embed()

    # x_train_plot = list(range(1, X_train.shape[0] + 1))
    # x_test_plot = list(range(1, X_test.shape[0] + 1))

    # # multiple linear regression
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # r_sq = model.score(X_test, y_test)
    # f_score_pred = model.predict(X_test)
    # mae = mean_absolute_error(y_test, f_score_pred)
    # print(f"mean absolute error for linear regression degree : {mae}")
    # mse = mean_squared_error(y_test, f_score_pred)
    # print(f"mean squared error for linear regression degree : {mse}")
    # sns.scatterplot(x_test_plot, y_test)
    # plt.ylim(0, 1)
    # sns.scatterplot(x_test_plot, f_score_pred)
    # plt.ylim(0, 1)

    # import statsmodels.api as sm
    # X_train_Sm = sm.add_constant(X_train)
    # X_train_Sm = sm.add_constant(X_train)
    # ls = sm.OLS(y_train, X_train_Sm).fit()
    # print(ls.summary())

    # # polynomial regression
    # # fit multiple polynomial features
    # degrees = [1, 2, 3, 6, 10, 20]
    # transformer_2 = PolynomialFeatures(degree=2, include_bias=True)
    # transformer_3 = PolynomialFeatures(degree=3, include_bias=True)

    # transformer_2.fit(X_train)
    # transformer_3.fit(X_train)

    # X_train_poly = transformer_2.transform(X_train)
    # model = LinearRegression()
    # model.fit(X_train_poly, y_train)
    # X_test_poly = transformer_2.transform(X_test)
    # r_sq = model.score(X_test_poly, y_test)
    # f_score_pred = model.predict(X_test_poly)
    # mae = mean_absolute_error(y_test, f_score_pred)
    # print(f"mean absolute error for polynomial regression degree 2 : {mae}")
    # mse = mean_squared_error(y_test, f_score_pred)
    # print(f"mean squared error for polynomial regression degree 2 : {mse}")
    # sns.scatterplot(x_test_plot, y_test)
    # plt.ylim(0, 1)
    # sns.scatterplot(x_test_plot, f_score_pred)
    # plt.ylim(0, 1)

    # X_train_poly = transformer_3.transform(X_train)
    # model = LinearRegression()
    # model.fit(X_train_poly, y_train)
    # X_test_poly = transformer_3.transform(X_test)
    # r_sq = model.score(X_test_poly, y_test)
    # f_score_pred = model.predict(X_test_poly)
    # mae = mean_absolute_error(y_test, f_score_pred)
    # print(f"mean absolute error for polynomial regression degree 3 : {mae}")
    # mse = mean_squared_error(y_test, f_score_pred)
    # print(f"mean squared error for polynomial regression degree 3 : {mse}")
    # sns.scatterplot(x_test_plot, y_test)
    # plt.ylim(0, 1)
    # sns.scatterplot(x_test_plot, f_score_pred)
    # plt.ylim(0, 1)

    # # Fitting Regression Trees to the dataset
    # regressor = DecisionTreeRegressor(random_state=0)
    # regressor.fit(X_train, y_train)
    # f_score_pred = regressor.predict(X_test)
    # mae = mean_absolute_error(y_test, f_score_pred)
    # print(f"mean absolute error for regression tree : {mae}")
    # mse = mean_squared_error(y_test, f_score_pred)
    # print(f"mean squared error for regression tree : {mse}")
    # sns.scatterplot(x_test_plot, y_test)
    # plt.ylim(0, 1)
    # sns.scatterplot(x_test_plot, f_score_pred)
    # plt.ylim(0, 1)

    # import numpy as np
    # import pandas as pd
    # from scipy import stats
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots

    # # applying various transformations
    # # log = np.log(df_10_full.copy()) # log
    # # log_corr = log.corr()
    # # log_heatmap = log_corr[log_corr['f_score'].notnull()][['f_score']]
    # # sns.heatmap(log_heatmap,annot=True,lw=1)

    # # sqrt = np.sqrt(df_10_full.copy()) # square root
    # # sqrt_corr = sqrt.corr()
    # # sqrt_heatmap = sqrt_corr[sqrt_corr['f_score'].notnull()][['f_score']]
    # # sns.heatmap(sqrt_heatmap,annot=True,lw=1)

    # # boxcox, _ = stats.boxcox(df_10_full.copy()) # boxcox
    # # boxcox_corr = boxcox.corr()
    # # boxcox_heatmap = boxcox_corr[boxcox_corr['f_score'].notnull()][['f_score']]
    # # sns.heatmap(boxcox_heatmap,annot=True,lw=1)

    # # import plotly.express as px
    # # from sklearn.decomposition import PCA
    # # pca = PCA(n_components=1)
    # # components = pca.fit_transform(X_train)
    # # q = components.reshape((150,))
    # # w = sns.scatterplot(x=q, y= y_train)
