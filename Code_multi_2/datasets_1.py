import os
import pathlib
import glob
from pyts import datasets
from sktime.utils.data_io import load_from_arff_to_dataframe as load_arff
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_tsf
from sktime.utils.data_io import load_from_ucr_tsv_to_dataframe as load_tsv
from sktime.utils.data_processing import from_long_to_nested, from_nested_to_long
from sktime.transformations.series.impute import Imputer

def lookup_dataset(dataset_name):
    try:
        if dataset_name in datasets.uea_dataset_list():
            print(datasets.uea_dataset_info(dataset_name))
            return  datasets.uea_dataset_info(dataset_name)
        elif dataset_name in datasets.ucr_dataset_list():
            print(datasets.ucr_dataset_info(dataset_name))
            return datasets.ucr_dataset_info(dataset_name)
        else :
            raise Exception(f"Dataset '{dataset_name}' was not found in either UEA or UCR archives. Please choose a valid dataset")
    except Exception as e:
        print(e)
        raise

def get_files_path(dataset_download_folder, dataset_name):
    arff_files_str_list = glob.glob(dataset_download_folder + "/**/*.arff", recursive = True)
    arff_files_path_list= list(map(pathlib.Path,arff_files_str_list))
    arff_parents= [os.path.normcase(path.parent) for path in arff_files_path_list]
    arff_parent_folders= set(arff_parents)

    ts_files_str_list = glob.glob(dataset_download_folder + "/**/*.ts", recursive = True)
    ts_files_path_list= list(map(pathlib.Path,ts_files_str_list))
    ts_parents= [os.path.normcase(path.parent) for path in ts_files_path_list]
    ts_parent_folders= set(ts_parents)

    arff_files_path = None
    ts_files_path = None

    if len(arff_parent_folders) == 1:
        arff_files_path= arff_parent_folders.pop()
        print(f"arff files found in {arff_files_path}")
    elif len(arff_parent_folders) == 0:
        print(f"No arff files found for {dataset_name}")
    else:
        print(f"arff files for {dataset_name} exist in more than one folder")

    if len(ts_parent_folders) == 1:
        ts_files_path= ts_parent_folders.pop()
        print(f"ts files found in {ts_files_path}")
    elif len(ts_parent_folders) == 0:
        print(f"No ts files found for {dataset_name}")
    else:
        print(f"ts files for {dataset_name} exist in more than one folder")

    if not ts_files_path and not arff_files_path:
        raise Exception(f"There is a problem finding data files for data set {dataset_name}")
    return arff_files_path, ts_files_path

def get_test_train_data(dataset_name):
    ds= os.path.normcase(os.path.join(os.getcwd(),'datasets', f'{dataset_name}'))
    if os.path.exists(ds):
        print(f"Dataset '{dataset_name}' was found in the datasets folder")
    else:
        print(f"Dataset '{dataset_name}' was not found in the datasets folder. Will lookup UEA and UCR archives")
        ds = _download_dataset(dataset_name)
    arff_files_path, ts_files_path= get_files_path(ds, dataset_name)
    X_train, X_test, y_train, y_test= _get_test_train_split(arff_files_path, ts_files_path, dataset_name)
    return X_train, X_test, y_train, y_test

def _download_dataset(dataset_name):
    try:
        datasets_folder= os.path.join(os.getcwd(), 'datasets','')
        if not os.path.exists(datasets_folder):
            os.makedirs(datasets_folder)
        ds_lower= str.lower(dataset_name)
        uea_lower= list(map(str.lower, datasets.uea_dataset_list()))
        ucr_lower= list(map(str.lower, datasets.ucr_dataset_list()))
        if ds_lower in uea_lower:
            ds_index= uea_lower.index(ds_lower)
            ds_name= datasets.uea_dataset_list()[ds_index]
            datasets.fetch_uea_dataset(dataset= ds_name, data_home= datasets_folder, use_cache=False)
        elif ds_lower in ucr_lower:
            ds_index= ucr_lower.index(ds_lower)
            ds_name= datasets.ucr_dataset_list()[ds_index]
            datasets.fetch_ucr_dataset(dataset= ds_name, data_home= datasets_folder, use_cache=False)
        else :
            raise Exception(f"Dataset '{dataset_name}' was not found in either UEA or UCR archives. Please choose a valid dataset")
    except Exception as e:
        raise
    return os.path.join(os.getcwd(), 'datasets', ds_name)

def _get_test_train_split(arff_files_path, ts_files_path, dataset_name):
    if arff_files_path:
        print("Loading arff files")
        try:
            X_train, y_train = load_arff(os.path.normcase(os.path.join(arff_files_path, f'{dataset_name}_TRAIN.arff')))
            X_test, y_test = load_arff(os.path.normcase( os.path.join(arff_files_path, f'{dataset_name}_TEST.arff')))
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print("Error loading arff files")  # Try to load using ts format
            print(e)
    elif not ts_files_path:
        raise Exception(f"No ts files to try for data set {dataset_name}")
    elif ts_files_path:
        print("Loading ts files")
        try:
            X_train, y_train = load_tsf(os.path.normcase(os.path.join(ts_files_path, f'{dataset_name}_TRAIN.ts')))
            X_test, y_test = load_tsf(os.path.normcase( os.path.join(ts_files_path, f'{dataset_name}_TEST.ts')))
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print("Error loading ts files")
            raise Exception(e)

def check_for_missing_values(df, ds, clf_name):
    long_format= from_nested_to_long(df)
    idx= long_format[long_format['value'].isnull()][['index','column']]
    if idx.empty:
        print(f"Dataset {ds} doesn't have any missing values")
        pass
    else:
        raise Exception(f"Dataset {ds} has missing values, current framework cannot handle such case")

def handle_missing_values(df, ds, clf_name):
    long_format= from_nested_to_long(df)
    idx= long_format[long_format['value'].isnull()][['index','column']]
    if idx.empty:
        pass
    else:
        idx= idx.drop_duplicates()
        impute_missing_values(idx, df, ds, clf_name)

def impute_missing_values(idx, df, ds, clf_name):
    print(f"Imputing dataset {ds} Missing Values for {clf_name}")
    imputer = Imputer(method="drift")
    for index, row in idx.iterrows():
        i= row['index']
        d= row['column']
        imputed_data= imputer.fit_transform(df[d].iloc[i])
        df[d].iloc[i]= imputed_data
