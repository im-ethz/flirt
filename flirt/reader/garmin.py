import pandas as pd

__all__ = [
    "read_acc_file_into_df",
    "read_data_file_into_df",
]


def read_data_file_into_df(file: str, filter_type: str = None) -> pd.DataFrame:
    """
    Reads a Garmin data file (as exported from the Android data logger application) into a pandas DataFrame

    Parameters
    ----------
    file : str
        file path
    filter_type : str, optional
        filter the values by type and return only matching values (e.g. 'hrv', 'hr', 'steps'), by default None

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame representing the file contents
    """
    data = pd.read_csv(file, index_col=0, sep=';', names=['datetime', 'type', 'value'], parse_dates=True)
    data.index = data.index.tz_localize(None)

    if filter_type is not None:
        data = data.loc[data['type'] == filter_type]
        data.drop('type', axis=1, inplace=True)
        data.rename(columns={'value': filter_type}, inplace=True)

    return data


def read_acc_file_into_df(file: str) -> pd.DataFrame:
    """
    Reads a Garmin accelerometer data file (as exported from the Android data logger application) into a pandas DataFrame

    Parameters
    ----------
    file : str
        file path

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame representing the file contents
    """
    data = pd.read_csv(file, index_col=0, sep=';', names=['datetime', 'acc_x', 'acc_y', 'acc_z'], parse_dates=True)
    data.index = data.index.tz_localize(None)

    return data
