import os
from pyts import datasets
from sktime.utils.data_io import load_from_arff_to_dataframe as load_arff
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_tsf
from sktime.utils.data_io import load_from_ucr_tsv_to_dataframe as load_tsv

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

def get_test_train_data(dataset_name):
    ds= os.path.normcase(os.path.join(os.getcwd(),'datasets', f'{dataset_name}'))
    if os.path.exists(ds):
        print(f"Dataset '{dataset_name}' was found in the datasets folder")
    else:
        print(f"Dataset '{dataset_name}' was not found in the datasets folder. Will lookup UEA and UCR archives")
        ds = _download_dataset(dataset_name)
    X_train, X_test, y_train, y_test= _get_test_train_split(ds)
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
            datasets.fetch_uea_dataset(dataset= ds_name, data_home= datasets_folder)
        elif ds_lower in ucr_lower:
            ds_index= ucr_lower.index(ds_lower)
            ds_name= datasets.ucr_dataset_list()[ds_index]
            datasets.fetch_ucr_dataset(dataset= ds_name, data_home= datasets_folder)
        else :
            raise Exception(f"Dataset '{dataset_name}' was not found in either UEA or UCR archives. Please choose a valid dataset")
    except Exception as e:
        raise
    return os.path.join(os.getcwd(), 'datasets', ds_name)

def _get_test_train_split(dataset_location):
    dataset = os.path.split(dataset_location)[-1]
    try:
        X_train, y_train = load_arff(os.path.normcase(os.path.join(dataset_location, f'{dataset}_TRAIN.arff')))
        X_test, y_test = load_arff(os.path.normcase( os.path.join(dataset_location, f'{dataset}_TEST.arff')))
    except Exception as e:
        print(e)
        print("Error loading arff files, will try using ts files")  # Try to load using ts format
        pass
    try:
        X_train, y_train = load_tsf(os.path.normcase(os.path.join(dataset_location, f'{dataset}_TRAIN.ts')))
        X_test, y_test = load_tsf(os.path.normcase( os.path.join(dataset_location, f'{dataset}_TEST.ts')))
    except Exception as e:
        print("Error loading both arff and ts files")
        raise
    return X_train, X_test, y_train, y_test