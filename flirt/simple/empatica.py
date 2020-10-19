import os
import zipfile

import pandas as pd

# import flirt
import flirt.reader.empatica


def get_features_for_empatica_archive(zip_file_path: str,
                                      window_length: int = 180,
                                      window_step_size: int = 1,
                                      hrv_features: bool = True,
                                      eda_features: bool = True,
                                      acc_features: bool = True,
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

    with zipfile.ZipFile(zip_file_path) as zip_file:
        if hrv_features:
            with zip_file.open("IBI.csv") as f:
                ibi_data = flirt.reader.empatica.read_ibi_file_into_df(f)

            if debug:
                print("Calculating HRV features")
            df_hrv_features = flirt.hrv.get_hrv_features(ibi_data.iloc[:, 0], window_length=window_length,
                                                         window_step_size=window_step_size)

        if eda_features:
            with zip_file.open("EDA.csv") as f:
                eda_data = flirt.reader.empatica.read_eda_file_into_df(f)

                if debug:
                    print("Calculating EDA features")
                df_eda_features = flirt.eda.get_eda_features(eda_data.iloc[:, 0], window_length=window_length,
                                                             window_step_size=window_step_size).add_prefix('eda_')

        if acc_features:
            with zip_file.open("ACC.csv") as f:
                acc_data = flirt.reader.empatica.read_acc_file_into_df(f)

                if debug:
                    print("Calculating ACC features")
                df_acc_features = flirt.acc.get_acc_features(acc_data[:], window_length=window_length,
                                                             window_step_size=window_step_size).add_prefix('acc_')

        return __merge_features(df_hrv_features, df_eda_features, df_acc_features, freq='%ds' % window_step_size)


def __merge_features(hrv_features: pd.DataFrame, eda_features: pd.DataFrame, acc_features: pd.DataFrame,
                     freq: str = '1s') -> pd.DataFrame:
    if hrv_features.empty and eda_features.empty and acc_features.empty:
        print("Received empty input features, returning empty df")
        return pd.DataFrame()

    merged_features = pd.concat([hrv_features, eda_features, acc_features], axis=1, sort=True)

    target_index = pd.date_range(start=merged_features.iloc[0].name.ceil('s'),
                                 end=merged_features.iloc[-1].name.floor('s'), freq=freq, tz='UTC')

    merged_features = merged_features.reindex(index=merged_features.index.union(target_index))
    merged_features.interpolate(method='time', inplace=True)
    merged_features = merged_features.reindex(target_index)

    return merged_features
