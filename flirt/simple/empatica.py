import os
import sys
if sys.version_info >= (3, 7):
    import zipfile
else:
    import zipfile37 as zipfile

import pandas as pd

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/fefespinola/ETHZ_Fall_2020/flirt-1')

import flirt
import flirt.reader.empatica
import flirt.eda.preprocessing as eda_preprocess

def get_features_for_empatica_archive(zip_file_path: str,
                                      window_length: int = 180,
                                      window_step_size: float = 1.0,
                                      hrv_features: bool = True,
                                      eda_features: bool = True,
                                      acc_features: bool = True,
                                      bvp_features: bool = True,
                                      temp_features: bool = True,
                                      debug: bool = False) -> pd.DataFrame:
    """
    This function provides a standard set of HRV, EDA, ACC features for a given Empatica archive \
    (e.g. downloaded from E4 connect)

    Parameters
    ----------
    zip_file_path : str
        path to the Empatica zip file
    window_length : int
        the epoch width (window size) in seconds, for which features should be calculated
    window_step_size : int
        the step size for the sliding window in seconds
    hrv_features : bool
        whether HRV features should be calculated from the archive
    eda_features : bool
        whether EDA features should be calculated from the archive
    acc_features : bool
        whether ACC features should be calculated from the archive
    debug : bool
        whether debug output should be printed

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame containing a standard set of features for Empatica

    Examples
    --------
    >>> import flirt.simple
    >>> features = flirt.simple.get_features_for_empatica_archive("1560460372_A12345.zip")
    """

    if zip_file_path is None or not os.path.isfile(zip_file_path) or not zipfile.is_zipfile(zip_file_path):
        raise ValueError('zip file does note exist %s' % zip_file_path)

    if debug:
        print("Reading files")

    df_hrv_features = pd.DataFrame()
    df_eda_features = pd.DataFrame()
    df_acc_features = pd.DataFrame()
    df_bvp_features = pd.DataFrame()
    df_temp_features = pd.DataFrame()

    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        if hrv_features:
            with zip_file.open("IBI.csv", "r") as f:
                ibi_data = flirt.reader.empatica.read_ibi_file_into_df(f)

                if debug:
                    print("Calculating HRV features")
                df_hrv_features = flirt.hrv.get_hrv_features(ibi_data.iloc[:, 0], window_length=window_length,
                                                            window_step_size=window_step_size).add_prefix('hrv_')


        if eda_features:
            with zip_file.open("EDA.csv") as f:
                eda_data = flirt.reader.empatica.read_eda_file_into_df(f)

                if debug:
                    print("Calculating EDA features")
                # change for different subjects!!!
                #filepath = '/home/fefespinola/ETHZ_Fall_2020/project_data/WESAD/S4' # for S4!!!
                #preprocessor=eda_preprocess.MultiStepPipeline(eda_preprocess.MitExplorerDetector(filepath))
                df_eda_features = flirt.eda.get_eda_features(eda_data.iloc[:, 0], window_length=window_length,
                                                             window_step_size=window_step_size,
                                                             preprocessor=eda_preprocess.ExtendedKalmanFilter(),
                                                             signal_decomposition=eda_preprocess.CvxEda(),
                                                             scr_features=eda_preprocess.ComputeNeurokitPeaks()).add_prefix('eda_')
                                                    

        if acc_features:
            with zip_file.open("ACC.csv") as f:
                acc_data = flirt.reader.empatica.read_acc_file_into_df(f)

                if debug:
                    print("Calculating ACC features")
                df_acc_features = flirt.acc.get_acc_features(acc_data[:], window_length=window_length,
                                                             window_step_size=window_step_size).add_prefix('acc_')


        if bvp_features:
            with zip_file.open("BVP.csv") as f:
                bvp_data = flirt.reader.empatica.read_bvp_file_into_df(f)

                if debug:
                    print("Calculating BVP features")
                df_bvp_features = flirt.bvp.get_bvp_features(bvp_data.iloc[:, 0], window_length=window_length,
                                                             window_step_size=window_step_size).add_prefix('bvp_')


        if temp_features:
            with zip_file.open("TEMP.csv") as f:
                temp_data = flirt.reader.empatica.read_temp_file_into_df(f)

                if debug:
                    print("Calculating temperature features")
                df_temp_features = flirt.temp.get_temp_features(temp_data.iloc[:, 0], window_length=window_length,
                                                             window_step_size=window_step_size).add_prefix('temp_')
                zip_file.close()

        return __merge_features(df_hrv_features, df_eda_features, df_acc_features, df_bvp_features, df_temp_features, freq='%dms' % (window_step_size*1000))


def __merge_features(hrv_features: pd.DataFrame, eda_features: pd.DataFrame, acc_features: pd.DataFrame, bvp_features: pd.DataFrame,
                     temp_features: pd.DataFrame, freq: str = '1s') -> pd.DataFrame:
    if hrv_features.empty and eda_features.empty and acc_features.empty and bvp_features.empty and temp_features.empty:
        print("Received empty input features, returning empty df")
        return pd.DataFrame()

    merged_features = pd.concat([hrv_features, eda_features, acc_features, bvp_features, temp_features], axis=1, sort=True)

    target_index = pd.date_range(start=merged_features.iloc[0].name.ceil('s'),
                                 end=merged_features.iloc[-1].name.floor('s'), freq=freq, tz='UTC')

    merged_features = merged_features.reindex(index=merged_features.index.union(target_index))
    merged_features.interpolate(method='time', inplace=True)
    merged_features = merged_features.reindex(target_index)

    return merged_features
