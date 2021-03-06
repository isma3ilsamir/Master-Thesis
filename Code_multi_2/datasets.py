import os
import pathlib
import glob
from pyts import datasets
from sktime.utils.data_io import load_from_arff_to_dataframe as load_arff
from sktime.utils.data_io import load_from_tsfile_to_dataframe as load_tsf
from sktime.utils.data_io import load_from_ucr_tsv_to_dataframe as load_tsv
from sktime.utils.data_processing import from_long_to_nested, from_nested_to_long
from sktime.transformations.series.impute import Imputer
import IPython

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

def get_files_path(dataset_download_folder):
    files_str_list = glob.glob(dataset_download_folder + "/**/*.arff", recursive = True)
    files_path_list= list(map(pathlib.Path,files_str_list))
    parents= [path.parent for path in files_path_list]
    parent_folders= set(parents)
    files_path=None
    if len(parent_folders) == 1:
        files_path= parent_folders.pop()
    else:
        raise Exception("There are more than one folder with arff files")
    return files_path

def get_test_train_data(dataset_name):
    ds= os.path.normcase(os.path.join(os.getcwd(),'datasets', f'{dataset_name}'))
    if os.path.exists(ds):
        print(f"Dataset '{dataset_name}' was found in the datasets folder")
    else:
        print(f"Dataset '{dataset_name}' was not found in the datasets folder. Will lookup UEA and UCR archives")
        ds = _download_dataset(dataset_name)
    files_path= get_files_path(ds)
    files_path= os.path.normcase(files_path)
    X_train, X_test, y_train, y_test= _get_test_train_split(files_path)
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

def _get_test_train_split(dataset_location):
    dataset = os.path.split(dataset_location)[-1]
    try:
        X_train, y_train = load_arff(os.path.normcase(os.path.join(dataset_location, f'{dataset}_TRAIN.arff')))
        X_test, y_test = load_arff(os.path.normcase( os.path.join(dataset_location, f'{dataset}_TEST.arff')))
    except Exception as e:
        print("Error loading arff files, will try using ts files")  # Try to load using ts format
        print(e)
        try:
            X_train, y_train = load_tsf(os.path.normcase(os.path.join(dataset_location, f'{dataset}_TRAIN.ts')))
            X_test, y_test = load_tsf(os.path.normcase( os.path.join(dataset_location, f'{dataset}_TEST.ts')))
        except Exception as e:
            raise Exception("Error loading both arff and ts files")
    return X_train, X_test, y_train, y_test

def check_for_missing_values(df, ds, clf_name):
    long_format= from_nested_to_long(df)
    idx= long_format[long_format['value'].isnull()][['index','column']]
    if idx.empty:
        print(f"Dataset {ds} doesn't have any missing values")
        pass
    else:
        print(f"Dataset {ds} has missing values, current framework cannot handle such case")
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

def check_for_missing_values_1(datasets):
    ds_w_missing= []
    ds_wout_missing= []
    for ds in datasets:
        X_train, _, _, _ = get_test_train_data(ds)
        long_format= from_nested_to_long(X_train)
        idx= long_format[long_format['value'].isnull()][['index','column']]
        if idx.empty:
            ds_wout_missing.append(ds)
        else:
            ds_w_missing.append(ds)
            # print(f"Dataset {ds} has missing values, current framework cannot handle such case")
            # raise Exception(f"Dataset {ds} has missing values, current framework cannot handle such case")
    return ds_wout_missing, ds_w_missing


l = ['StandWalkJump',
'AtrialFibrillation',
'InsectEPGSmallTrain',
'Fungi',
'Chinatown',
'SonyAIBORobotSurface1',
'MoteStrain',
'DodgerLoopGame',
'DodgerLoopWeekend',
'Rock',
'TwoLeadECG',
'ECGFiveDays',
'SonyAIBORobotSurface2',
'Coffee',
'FreezerSmallTrain',
'Ering',
'Beef',
'OliveOil',
'HouseTwenty',
'BasicMotions',
'CinCECGtorso',
'Wine',
'DuckDuckGeese',
'Meat',
'Car',
'Lightning2',
'InsectEPGRegularTrain',
'ItalyPowerDemand',
'Lightning7',
'DodgerLoopDay',
'ECG200',
'Trace',
'ACSF1',
'PigAirwayPressure',
'PigArtPressure',
'PigCVP',
'Plane',
'Cricket',
'Ham',
'Epilepsy',
'Handwriting',
'FreezerRegularTrain',
'RacketSports',
'HandMovementDirection',
'Libras',
'NATOPS',
'PowerCons',
'SelfRegulationSCP2',
'Heartbeat',
'Phoneme',
'Computers',
'EthanolConcentration',
'PEMS-SF',
'SelfRegulationSCP1',
'JapaneseVowels',
'MotorImagery',
'SemgHandGenderCh2',
'FingerMovements',
'Earthquakes',
'EOGHorizontalSignal',
'EOGVerticalSignal',
'LargeKitchenAppliances',
'RefrigerationDevices',
'ScreenType',
'SmallKitchenAppliances',
'SemgHandMovementCh2',
'SemgHandSubjectCh2',
'ECG5000',
'EthanolLevel',
'Strawberry',
'Wafer',
'StarlightCurves',
'MelbournePedestrian',
# 'NonInvasiveFetalECGThorax1',
# 'NonInvasiveFetalECGThorax2',
'UWaveGestureLibrary',
'LSST',
'FordA',
'FordB',
'FaceDetection',
'SpokenArabicDigits',
'ElectricDevices',
# 'InsectWingbeat',
'InsectWingbeatSound']

ds_wout_missing, ds_w_missing = check_for_missing_values_1(l)

IPython.embed()